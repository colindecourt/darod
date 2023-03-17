import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ..utils import bbox_utils


def compute_eta(total_duration, epoch_duration, epoch, total_epochs):
    """Compute training ETA"""
    total_duration += epoch_duration
    eta = (total_epochs - epoch) * total_duration / (epoch)
    return time.strftime("%H:%M:%S", time.gmtime(eta)), total_duration


def randomly_select_xyz_mask(mask, select_xyz, seed):
    """
    Selecting x, y, z number of True elements for corresponding batch and replacing others to False
    :param mask: (batch_size, [m_bool_value])
    :param select_xyz: (batch_size, [m_bool_value])
    :param seed: seed
    :return:  selected_valid_mask = (batch_size, [m_bool_value])
    """
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32, seed=seed)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)


def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, config):
    """
    Generating one step data for training or inference. Batch operations supported.
    :param anchors: (total_anchors, [y1, x1, y2, x2]) these values in normalized format between [0, 1]
    :param gt_boxes: (batch_size, gt_box_size, [y1, x1, y2, x2]) these values in normalized format between [0, 1]
    :param gt_labels: (batch_size, gt_box_size)
    :param config: dictionary
    :return: bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
             bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    batch_size = tf.shape(gt_boxes)[0]
    anchor_count = config["rpn"]["anchor_count"]
    total_pos_bboxes = int(config["rpn"]["rpn_boxes"] / 2)
    total_neg_bboxes = int(config["rpn"]["rpn_boxes"] / 2)
    variances = config["rpn"]["variances"]
    adaptive_ratio = config["rpn"]["adaptive_ratio"]
    postive_th = config["rpn"]["positive_th"]
    #
    output_height, output_width = config["model"]["feature_map_shape"][0], config["model"]["feature_map_shape"][1]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # Get max index value for each column
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_mask = tf.greater(merged_iou_map, postive_th)
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0],), True), tf.shape(pos_mask))
    pos_mask = tf.logical_and(tf.logical_or(pos_mask, max_pos_mask), tf.reduce_sum(anchors, axis=-1) != 0.0)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32),
                                        seed=config["training"]["seed"])
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    # Keep a 50%/50% ratio of positive/negative samples
    if adaptive_ratio:
        neg_count = 2 * pos_count
    else:
        neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    #
    neg_mask = tf.logical_and(tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask)),
                              tf.reduce_sum(anchors, axis=-1) != 0.0)
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count, seed=config["training"]["seed"])
    #
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    bbox_labels = tf.reshape(bbox_labels, (batch_size, output_height, output_width, anchor_count))
    #
    return bbox_deltas, bbox_labels


def frcnn_cls_loss(*args):
    """
    Calculating faster rcnn class loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: CE loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]

    loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=True)
    loss_for_all = loss_fn(y_true, y_pred)
    #
    cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    #
    conf_loss = tf.reduce_sum(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    return tf.math.divide_no_nan(conf_loss, total_boxes)


def rpn_cls_loss(*args):
    """
    Calculating rpn class loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: CE loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy(from_logits=True)
    return lf(target, output)


def reg_loss(*args):
    """
    Calculating rpn / faster rcnn regression loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: regression loss val
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE, delta=1 / 9)
    loss_for_all = loss_fn(y_true, y_pred)
    # loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    #
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    #
    loc_loss = tf.reduce_sum(tf.math.multiply_no_nan(pos_mask, loss_for_all))
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))

    return tf.math.divide_no_nan(loc_loss, total_pos_bboxes)


def giou_loss(y_true, y_pred, roi_bboxes):
    """
    Calculating rpn / faster rcnn regression loss value.
    :param y_true: ground truth
    :param y_pred: predictied deltas
    :param roi_bboxes: predicted boxes
    :return: regression loss val
    """
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    y_pred = bbox_utils.get_bboxes_from_deltas(roi_bboxes, y_pred)
    #
    loss_fn = tfa.losses.GIoULoss(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    # loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    #
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    #
    loc_loss = tf.reduce_sum(tf.math.multiply_no_nan(pos_mask, loss_for_all))
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))

    return tf.math.divide_no_nan(loc_loss, total_pos_bboxes)


def darod_loss(rpn_cls_pred, rpn_delta_pred, frcnn_cls_pred, frcnn_reg_pred,
              bbox_labels, bbox_deltas, frcnn_reg_actuals, frcnn_cls_actuals):
    """
    Calculate loss function for DAROD model
    :param rpn_cls_pred: RPN classification pred
    :param rpn_delta_pred: RPN regression pred
    :param frcnn_cls_pred: FRCNN classification pred
    :param frcnn_reg_pred: FRCNN regression pred
    :param bbox_labels: bounding boxes labels
    :param bbox_deltas: bounding boxes deltas
    :param frcnn_reg_actuals: faster rcnn regression labels
    :param frcnn_cls_actuals: faster rcnn classification labels
    :return: faster rcnn loss
    """
    rpn_regression_loss = reg_loss(bbox_deltas, rpn_delta_pred)
    rpn_classif_loss = rpn_cls_loss(bbox_labels, rpn_cls_pred)
    frcnn_regression_loss = reg_loss(frcnn_reg_actuals, frcnn_reg_pred)
    frcnn_classif_loss = frcnn_cls_loss(frcnn_cls_actuals, frcnn_cls_pred)
    return rpn_regression_loss, rpn_classif_loss, frcnn_regression_loss, frcnn_classif_loss


def darod_loss_giou(rpn_cls_pred, rpn_delta_pred, frcnn_cls_pred, frcnn_reg_pred,
                   bbox_labels, bbox_deltas, frcnn_reg_actuals, frcnn_cls_actuals, roi_bboxes):
    rpn_regression_loss = reg_loss(bbox_deltas, rpn_delta_pred)
    rpn_classif_loss = rpn_cls_loss(bbox_labels, rpn_cls_pred)
    frcnn_regression_loss = giou_loss(frcnn_reg_actuals, frcnn_reg_pred, roi_bboxes) / 10
    frcnn_classif_loss = frcnn_cls_loss(frcnn_cls_actuals, frcnn_cls_pred)
    return rpn_regression_loss, rpn_classif_loss, frcnn_regression_loss, frcnn_classif_loss
