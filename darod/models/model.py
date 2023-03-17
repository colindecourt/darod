import tensorflow as tf
from tensorflow.keras import layers

from .darod_blocks import DARODBlock2D
from ..layers.frcnn_layers import RoIBBox, RoIPooling, RadarFeatures
from ..utils import bbox_utils


class Decoder(tf.keras.layers.Layer):
    """
    Generating bounding boxes and labels from faster rcnn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting top_n boxes by scores.
    """
    """
    inputs:
        roi_bboxes = (batch_size, roi_bbox_size, [y1, x1, y2, x2])
        pred_deltas = (batch_size, roi_bbox_size, total_labels * [delta_y, delta_x, delta_h, delta_w])
        pred_label_probs = (batch_size, roi_bbox_size, total_labels)
    outputs:
        pred_bboxes = (batch_size, top_n, [y1, x1, y2, x2])
        pred_labels = (batch_size, top_n)
            1 to total label number
        pred_scores = (batch_size, top_n)
    """

    def __init__(self, variances, total_labels, max_total_size=100, score_threshold=0.05, iou_threshold=0.5, **kwargs):
        """

        :param variances: bbox variances
        :param total_labels: number of classes
        :param max_total_size: max number of predictions
        :param score_threshold: score threshold
        :param iou_threshold: iou threshold
        :param kwargs: other args
        """
        super(Decoder, self).__init__(**kwargs)
        self.variances = variances
        self.total_labels = total_labels
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "variances": self.variances,
            "total_labels": self.total_labels,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold
        })
        return config

    def call(self, inputs):
        """
        Make final predictions from DAROD outputs
        :param inputs: DAROD outputs (roi boxes, deltas and probas)
        :return: final predictions (boxes, classes, scores)
        """
        roi_bboxes = inputs[0]
        pred_deltas = inputs[1]
        pred_label_probs = inputs[2]
        batch_size = tf.shape(pred_deltas)[0]
        #
        pred_deltas = tf.reshape(pred_deltas, (batch_size, -1, self.total_labels, 4))

        pred_deltas *= self.variances
        #
        expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, self.total_labels, 1))
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)
        pred_bboxes = tf.clip_by_value(pred_bboxes, clip_value_min=0.0, clip_value_max=1.0)
        #
        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        #
        final_bboxes, final_scores, final_labels, _ = bbox_utils.non_max_suppression(
            pred_bboxes, pred_labels,
            iou_threshold=self.iou_threshold,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold)
        #
        return tf.stop_gradient(final_bboxes), tf.stop_gradient(final_labels), tf.stop_gradient(final_scores)


class DAROD(tf.keras.Model):
    def __init__(self, config, anchors):
        """
        Implement DAROD model

        :param config: configuration dictionary to build the model
        :param anchors: generated anchors.
        """
        super(DAROD, self).__init__()
        self.config = config
        self.anchors = anchors
        #
        self.use_doppler = config["training"]["use_doppler"]
        self.use_dropout = config["training"]["use_dropout"]
        self.variances = config["fastrcnn"]["variances_boxes"]
        self.total_labels = config["data"]["total_labels"]
        self.frcnn_num_pred = config["fastrcnn"]["frcnn_num_pred"]
        self.box_nms_iou = config["fastrcnn"]["box_nms_iou"]
        self.box_nms_score = config["fastrcnn"]["box_nms_score"]
        self.dropout_rate = config["training"]["dropout_rate"]
        self.layout = config["model"]["layout"]
        self.use_bn = config["training"]["use_bn"]
        self.dilation_rate = config["model"]["dilation_rate"]

        #
        if self.use_bn:
            block_norm = "group_norm"  # By default for this model
        else:
            block_norm = None

        # Backbone
        self.block1 = DARODBlock2D(filters=64, padding="same", kernel_size=(3, 3),
                                   num_conv=2, dilation_rate=self.dilation_rate, activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 2),
                                   pooling_strides=(2, 2), name="darod_block1")

        self.block2 = DARODBlock2D(filters=128, padding="same", kernel_size=(3, 3),
                                   num_conv=2, dilation_rate=self.dilation_rate, activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 1),
                                   pooling_strides=(2, 1), name="darod_block2")

        self.block3 = DARODBlock2D(filters=256, padding="same", kernel_size=(3, 3),
                                   num_conv=3, dilation_rate=(1, 1), activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 1),
                                   pooling_strides=(2, 1), name="darod_block3")

        # RPN
        self.rpn_conv = layers.Conv2D(config["rpn"]["rpn_channels"], config["rpn"]["rpn_window"], padding="same",
                                      activation="relu", name="rpn_conv")
        self.rpn_cls_output = layers.Conv2D(config["rpn"]["anchor_count"], (1, 1), activation="linear", name="rpn_cls")
        self.rpn_reg_output = layers.Conv2D(4 * config["rpn"]["anchor_count"], (1, 1), activation="linear",
                                            name="rpn_reg")

        # Fast RCNN
        self.roi_bbox = RoIBBox(anchors, config, name="roi_bboxes")
        self.radar_features = RadarFeatures(config, name="radar_features")
        self.roi_pooled = RoIPooling(config, name="roi_pooling")

        #
        self.flatten = layers.Flatten(name="frcnn_flatten")
        if self.use_dropout:
            self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.fc1 = layers.Dense(config["fastrcnn"]["in_channels_1"], activation="relu", name="frcnn_fc1")
        self.fc2 = layers.Dense(config["fastrcnn"]["in_channels_2"], activation="relu", name="frcnn_fc2")
        self.frcnn_cls = layers.Dense(config["data"]["total_labels"], activation='linear', name="frcnn_cls")
        self.frcnn_reg = layers.Dense(config["data"]["total_labels"] * 4, activation="linear", name="frcnn_reg")

        self.decoder = Decoder(variances=self.variances, total_labels=self.total_labels,
                               max_total_size=self.frcnn_num_pred,
                               score_threshold=self.box_nms_score, iou_threshold=self.box_nms_iou)

    def call(self, inputs, step="frcnn", training=False):
        """
        DAROD forward pass

        :param inputs: RD spectrum
        :param step: RPN (for RPN pretraining) or Fast R-CNN for end to end training
        :param training: training or inference
        :return: RPN predictions (bbox regression and class) and/or Fast R-CNN predictions (bbox regression and object class)
        """

        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)

        # RPN forward pass
        rpn_out = self.rpn_conv(x)
        rpn_cls_pred = self.rpn_cls_output(rpn_out)
        rpn_delta_pred = self.rpn_reg_output(rpn_out)

        if step == "rpn":
            return rpn_cls_pred, rpn_delta_pred

        # Fast RCNN part
        roi_bboxes_out, roi_bboxes_scores = self.roi_bbox([rpn_delta_pred, rpn_cls_pred], training=training)

        roi_pooled_out = self.roi_pooled([x, roi_bboxes_out])

        output = layers.TimeDistributed(self.flatten)(roi_pooled_out)
        features = self.radar_features([roi_bboxes_out, roi_bboxes_scores], training=training)
        output = tf.concat([output, features], -1)

        output = layers.TimeDistributed(self.fc1)(output)
        if self.use_dropout:
            output = layers.TimeDistributed(self.dropout, name="dropout_1")(output)
        output = layers.TimeDistributed(self.fc2)(output)
        if self.use_dropout:
            output = layers.TimeDistributed(self.dropout, name="dropout_2")(output)
        #
        frcnn_cls_pred = layers.TimeDistributed(self.frcnn_cls)(output)
        frcnn_reg_pred = layers.TimeDistributed(self.frcnn_reg)(output)

        # Decoder part
        decoder_output = self.decoder([roi_bboxes_out, frcnn_reg_pred, tf.nn.softmax(frcnn_cls_pred)])

        return rpn_cls_pred, rpn_delta_pred, frcnn_cls_pred, frcnn_reg_pred, roi_bboxes_out, decoder_output
