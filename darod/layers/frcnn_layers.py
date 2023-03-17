import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..utils import bbox_utils, train_utils


class RoIBBox(Layer):
    """
    Generating bounding boxes from rpn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting "train or test nms_topn" boxes.
    """
    def __init__(self, anchors, config, **kwargs):
        """

        :param anchors: generated anchors
        :param config: configuration dictionary
        :param kwargs:
        """
        super(RoIBBox, self).__init__(**kwargs)
        self.config = config
        # self.mode = mode
        self.anchors = tf.constant(anchors, dtype=tf.float32)

    def get_config(self):
        config = super(RoIBBox, self).get_config()
        # config.update({"config": self.config, "anchors": self.anchors.numpy(), "mode": self.mode})
        return config

    def call(self, inputs, training=True):
        """

        :param inputs: rpn_bbox_deltas and rpn_labels
        :param training:
        :return: roi boxes
        """
        rpn_bbox_deltas = inputs[0]
        rpn_labels = inputs[1]
        anchors = self.anchors
        #
        pre_nms_topn = self.config["fastrcnn"]["pre_nms_topn_train"] if training else self.config["fastrcnn"][
            "pre_nms_topn_test"]
        post_nms_topn = self.config["fastrcnn"]["post_nms_topn_train"] if training else self.config["fastrcnn"][
            "post_nms_topn_test"]
        nms_iou_threshold = self.config["rpn"]["rpn_nms_iou"]
        variances = self.config["rpn"]["variances"]
        total_anchors = anchors.shape[0]
        batch_size = tf.shape(rpn_bbox_deltas)[0]
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, total_anchors, 4))
        # Apply softmax to rpn_labels to obtain probabilities
        rpn_labels = tf.nn.softmax(rpn_labels)

        rpn_labels = tf.reshape(rpn_labels, (batch_size, total_anchors))
        #
        rpn_bbox_deltas *= variances
        rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)

        _, pre_indices = tf.nn.top_k(rpn_labels, pre_nms_topn)
        #
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_labels = tf.gather(rpn_labels, pre_indices, batch_dims=1)
        #
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_labels = tf.reshape(pre_roi_labels, (batch_size, pre_nms_topn, 1))

        roi_bboxes, roi_scores, _, _ = bbox_utils.non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                                      max_output_size_per_class=post_nms_topn,
                                                                      max_total_size=post_nms_topn,
                                                                      iou_threshold=nms_iou_threshold)

        return tf.stop_gradient(roi_bboxes), tf.stop_gradient(roi_scores)


class RadarFeatures(Layer):
    """
    Extracting radar feature from RPN proposed boxes.
    This layer extracts range and Doppler values from RPN proposed
    boxes which have scores > 0.5. Otherwise, range and Doppler values
    are set to -1
    """
    def __init__(self, config, **kwargs):
        super(RadarFeatures, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(RadarFeatures, self).get_config()
        config.update({"config": self.config})
        return config

    def call(self, inputs):
        """

        :param inputs: roi_bboxes and roi_scores
        :return: radar_features
        """
        # Get inputs
        roi_bboxes = bbox_utils.denormalize_bboxes(inputs[0],
                                                   height=self.config["model"]["input_size"][0],
                                                   width=self.config["model"]["input_size"][1])
        roi_scores = inputs[1]

        # Get centers of roi boxes
        h = roi_bboxes[..., 2] - roi_bboxes[..., 0]
        w = roi_bboxes[..., 3] - roi_bboxes[..., 1]

        ctr_y = roi_bboxes[..., 0] + h / 2
        ctr_x = roi_bboxes[..., 1] + w / 2

        # Get radar feature
        range_values = self.config["data"]["range_res"] * tf.cast(ctr_y, tf.float32)
        doppler_values = tf.abs(32 - (self.config["data"]["doppler_res"] * tf.cast(ctr_x, tf.float32)))
        radar_features = tf.stack([range_values, doppler_values], axis=-1)

        # Get valid radar features (i.e. boxes with height > 0 and width > 0 and scores > 0.5)
        radar_features = tf.where(tf.expand_dims(tf.logical_and(roi_scores > 0.5, tf.logical_and(h > 0, w > 0)), -1),
                                  radar_features, tf.ones_like(radar_features) * -1)

        return radar_features


class RoIDelta(Layer):
    """
    Calculating faster rcnn actual bounding box deltas and labels.
    This layer only running on the training phase.
    """
    def __init__(self, config, **kwargs):
        super(RoIDelta, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(RoIDelta, self).get_config()
        config.update({"config": self.config})
        return config

    def call(self, inputs):
        """

        :param inputs: roi boxes, gt boxes and gt labels
        :return: roi boxes deltas and roi boxes labels
        """
        roi_bboxes = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]
        total_labels = self.config["data"]["total_labels"]
        total_pos_bboxes = int(self.config["fastrcnn"]["frcnn_boxes"] / 3)
        total_neg_bboxes = int(self.config["fastrcnn"]["frcnn_boxes"] * (2 / 3))
        variances = self.config["fastrcnn"]["variances_boxes"]
        adaptive_ratio = self.config["fastrcnn"]["adaptive_ratio"]
        positive_th = self.config["fastrcnn"]["positive_th"]

        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        # Calculate iou values between each bboxes and ground truth boxes
        iou_map = bbox_utils.generate_iou_map(roi_bboxes, gt_boxes)
        # Get max index value for each row
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        # IoU map has iou values for every gt boxes and we merge these values column wise
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        #
        pos_mask = tf.greater(merged_iou_map, positive_th)
        pos_mask = train_utils.randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32),
                                                        seed=self.config["training"]["seed"])
        #
        neg_mask = tf.logical_and(tf.less(merged_iou_map, positive_th), tf.greater_equal(merged_iou_map, 0.0))
        if adaptive_ratio:
            pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
            # Keep a 33%/66% ratio of positive/negative bboxes
            total_neg_bboxes = tf.multiply(pos_count + 1, 3)
            neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, total_neg_bboxes,
                                                            seed=self.config["training"]["seed"])
        else:
            neg_mask = train_utils.randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32),
                                                            seed=self.config["training"]["seed"])
        #
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        #
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)
        expanded_gt_labels = pos_gt_labels + neg_gt_labels
        #
        roi_bbox_deltas = bbox_utils.get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes) / variances
        #
        roi_bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
        scatter_indices = tf.tile(tf.expand_dims(roi_bbox_labels, -1), (1, 1, 1, 4))
        roi_bbox_deltas = scatter_indices * tf.expand_dims(roi_bbox_deltas, -2)
        roi_bbox_deltas = tf.reshape(roi_bbox_deltas, (batch_size, total_bboxes * total_labels, 4))
        #
        if self.config["fastrcnn"]["reg_loss"] == "sl1":
            return tf.stop_gradient(roi_bbox_deltas), tf.stop_gradient(roi_bbox_labels)
        elif self.config["fastrcnn"]["reg_loss"] == "giou":
            expanded_roi_boxes = scatter_indices * tf.expand_dims(roi_bboxes, -2)
            expanded_roi_boxes = tf.reshape(expanded_roi_boxes, (batch_size, total_bboxes * total_labels, 4))
            #
            expanded_gt_boxes = scatter_indices * tf.expand_dims(expanded_gt_boxes, -2)
            expanded_gt_boxes = tf.reshape(expanded_gt_boxes, (batch_size, total_bboxes * total_labels, 4))
            return tf.stop_gradient(expanded_roi_boxes), tf.stop_gradient(expanded_gt_boxes), tf.stop_gradient(
                roi_bbox_labels)


class RoIPooling(Layer):
    """
    Reducing all feature maps to same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    """
    def __init__(self, config, **kwargs):
        super(RoIPooling, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(RoIPooling, self).get_config()
        config.update({"config": self.config})
        return config

    def call(self, inputs):
        """

        :param inputs: feature map and roi boxes
        :return: pooled features
        """
        feature_map = inputs[0]
        roi_bboxes = inputs[1]
        pooling_size = self.config["fastrcnn"]["pooling_size"]
        batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
        #
        row_size = batch_size * total_bboxes
        # We need to arange bbox indices for each batch
        pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
        pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1,))
        pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
        # Crop to bounding box size then resize to pooling size
        pooling_feature_map = tf.image.crop_and_resize(
            feature_map,
            pooling_bboxes,
            pooling_bbox_indices,
            pooling_size
        )
        final_pooling_feature_map = tf.reshape(pooling_feature_map, (
        batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2],
        pooling_feature_map.shape[3]))
        return final_pooling_feature_map
