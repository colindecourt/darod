import tensorflow as tf


def anchors_generation(config, train=True):
    """
    Generating anchors boxes with different shapes centered on
    each pixel.
    :param config: config file with input data, anchor scales/ratios
    :param train: anchors for training or inference
    :return: base_anchors = (anchor_count * fm_h * fm_w, [y1, x1, y2, x2]
    """
    in_height, in_width = config["model"]["feature_map_shape"]
    scales, ratios = config["rpn"]["anchor_scales"], config["rpn"]["anchor_ratios"]
    num_scales, num_ratios = len(scales), len(ratios)
    scale_tensor = tf.convert_to_tensor(scales, dtype=tf.float32)
    ratio_tensor = tf.convert_to_tensor(ratios, dtype=tf.float32)
    boxes_per_pixel = (num_scales + num_ratios - 1)
    #
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # Generate all center points for the anchor boxes
    center_h = (tf.range(in_height, dtype=tf.float32) + offset_h) * steps_h
    center_w = (tf.range(in_width, dtype=tf.float32) + offset_w) * steps_w
    shift_y, shift_x = tf.meshgrid(center_h, center_w)
    shift_y, shift_x = tf.reshape(shift_y, shape=(-1)), tf.reshape(shift_x, shape=(-1))

    # Generate "boxes_per_pixel" number of heights and widths that are later
    # used to create anchor box corner coordinates xmin, xmax, ymin ymax
    w = tf.concat((scale_tensor * tf.sqrt(ratio_tensor[0]), scales[0] * tf.sqrt(ratio_tensor[1:])),
                  axis=-1) * in_height / in_width
    h = tf.concat((scale_tensor / tf.sqrt(ratio_tensor[0]), scales[0] / tf.sqrt(ratio_tensor[1:])),
                  axis=-1) * in_height / in_width

    # Divide by 2 to get the half height and half width
    anchor_manipulation = tf.tile(tf.transpose(tf.stack([-w, -h, w, h])), [in_height * in_width, 1]) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = tf.repeat(tf.stack([shift_x, shift_y, shift_x, shift_y], axis=1), boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulation

    if train:
        output = tf.where(
            tf.reduce_any(tf.logical_or(tf.less_equal(output, 0.0), tf.greater_equal(output, 1.0)), axis=-1,
                          keepdims=True) == False, output, 0.0)
    else:
        output = tf.clip_by_value(output, clip_value_min=0.0, clip_value_max=1.0)
    return output


def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    """
    Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression

    :param pred_bboxes: (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2]), total_labels should be 1 for binary operations like in rpn
    :param pred_labels: (batch_size, total_bboxes, total_labels)
    :param kwargs: other parameters
    :return: nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
            nmsed_scores = (batch_size, max_detections)
            nmsed_classes = (batch_size, max_detections)
            valid_detections = (batch_size)
                Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
                The rest of the entries are zero paddings.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )


def get_bboxes_from_deltas(anchors, deltas):
    """
    Calculating bounding boxes for given bounding box and delta values.
    :param anchors: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param deltas: (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    :return:  final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    #
    return tf.stack([y1, x1, y2, x2], axis=-1)


def get_bboxes_transforms(deltas):
    """
    Calculating bounding boxes for given delta values.
    :param deltas:  (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    :return:  final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    dy = deltas[..., 0]
    dx = deltas[..., 1]
    dh = deltas[..., 2]
    dw = deltas[..., 3]

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = tf.exp(dw)
    pred_h = tf.exp(dh)

    y1 = pred_ctr_y - 0.5 * pred_h
    x1 = pred_ctr_x - 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    x2 = pred_ctr_y + 0.5 * pred_w

    return tf.stack([y1, x1, y2, x2], axis=-1)


def get_deltas_from_bboxes(bboxes, gt_boxes):
    """
    Calculating bounding box deltas for given bounding box and ground truth boxes.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param gt_boxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :return:  final_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    #
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height),
                       tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    #
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)


def generate_iou_map(bboxes, gt_boxes):
    """
    Calculating iou values for each ground truth boxes in batched manner.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param gt_boxes: (batch_size, total_gt_boxes, [y1, x1, y2, x2])
    :return: iou_map = (batch_size, total_bboxes, total_gt_boxes)
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    #
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    ### Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    # Intersection over Union
    return tf.math.divide_no_nan(intersection_area, union_area)


def generate_delta_map(bboxes, gt_boxes):
    """
    Calculating delta values between center for each ground truth boxes in batched manner.
    :param bboxes:  (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param gt_boxes:  (batch_size, total_gt_boxes, [y1, x1, y2, x2])
    :return: delta_map = (batch_size, total_bboxes, total_gt_boxes)
    """
    # Denormalize bboxes for better computation
    bboxes = denormalize_bboxes(bboxes, 256, 64)
    gt_boxes = denormalize_bboxes(gt_boxes, 256, 64)
    #
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    #  Calculate bbox and groud truth centers
    bbox_center_x = bbox_x1 + (bbox_x2 - bbox_x1) / 2
    bbox_center_y = bbox_y1 + (bbox_y2 - bbox_y1) / 2
    #
    gt_center_x = gt_x1 + (gt_x2 - gt_x1) / 2
    gt_center_y = gt_y1 + (gt_y2 - gt_y1) / 2
    #
    deltas_x = tf.pow((tf.transpose(gt_center_x, [0, 2, 1]) - bbox_center_x), 2)
    deltas_y = tf.pow((tf.transpose(gt_center_y, [0, 2, 1]) - bbox_center_y), 2)
    deltas = tf.add(deltas_x, deltas_y)

    deltas = tf.math.sqrt(deltas)

    return deltas


def normalize_bboxes(bboxes, height, width):
    """
    Normalizing bounding boxes.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param height: image height
    :param width: image width
    :return: normalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2]) in normalized form [0, 1]
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)


def denormalize_bboxes(bboxes, height, width):
    """
    Denormalizing bounding boxes.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2]) in normalized form [0, 1]
    :param height: image height
    :param width: image width
    :return: denormalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))


def rotate_bboxes(bboxes, height, width):
    """
    Rotate bounding boxes
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2]) in denormalize form [0, heigth/width]
    :param height: image height
    :param width: image width
    :return: rotated_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    y1 = height - bboxes[..., 0]
    x1 = width - bboxes[..., 1]
    y2 = height - bboxes[..., 2]
    x2 = width - bboxes[..., 3]
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
