import json
import os

import numpy as np


def giou2d(bboxes1, bboxes2):
    """
    Calculate the gious between each bbox of bboxes1 and bboxes2.
    :param bboxes1: [x1, y1, x2, y2]
    :param bboxes2: [x1, y1, x2, y2]
    :return: Generalised IoU between boxes1 and boxes2
    """
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)

    area1 = (bboxes1[2] - bboxes1[0] + 1) * (
            bboxes1[3] - bboxes1[1] + 1)
    area2 = (bboxes2[2] - bboxes2[0] + 1) * (
            bboxes2[3] - bboxes2[1] + 1)

    x_start = np.maximum(bboxes1[0], bboxes2[0])
    x_min = np.minimum(bboxes1[0], bboxes2[0])
    y_start = np.maximum(bboxes1[1], bboxes2[1])
    y_min = np.minimum(bboxes1[1], bboxes2[1])
    x_end = np.minimum(bboxes1[2], bboxes2[2])
    x_max = np.maximum(bboxes1[2], bboxes2[2])
    y_end = np.minimum(bboxes1[3], bboxes2[3])
    y_max = np.maximum(bboxes1[3], bboxes2[3])

    overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
    closure = np.maximum(x_max - x_min + 1, 0) * np.maximum(y_max - y_min + 1, 0)

    union = area1 + area2 - overlap
    ious = overlap / union - (closure - union) / closure

    return ious


def iou2d(box_xywh_1, box_xywh_2):
    """
    Numpy version of 3D bounding box IOU calculation
    :param box_xywh_1: [x1, y1, x2, y2]
    :param box_xywh_2: [x1, y1, x2, y2]
    :return:
    """
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    box1_w = box_xywh_1[..., 2] - box_xywh_1[..., 0]
    box1_h = box_xywh_1[..., 3] - box_xywh_1[..., 1]
    #
    box2_w = box_xywh_2[..., 2] - box_xywh_2[..., 0]
    box2_h = box_xywh_2[..., 3] - box_xywh_2[..., 1]
    ### areas of both boxes
    box1_area = box1_h * box1_w
    box2_area = box2_h * box2_w
    ### find the intersection box
    box1_min = [box_xywh_1[..., 0], box_xywh_1[..., 1]]
    box1_max = [box_xywh_1[..., 2], box_xywh_1[..., 3]]
    box2_min = [box_xywh_2[..., 0], box_xywh_2[..., 1]]
    box2_max = [box_xywh_2[..., 2], box_xywh_2[..., 3]]

    x_top = np.maximum(box1_min[0], box2_min[0])
    y_top = np.maximum(box1_min[1], box2_min[1])
    x_bottom = np.minimum(box1_max[0], box2_max[0])
    y_bottom = np.minimum(box1_max[1], box2_max[1])

    intersection_area = np.maximum(x_bottom - x_top, 0) * np.maximum(y_bottom - y_top, 0)
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


def init_tp_dict(n_classes, iou_threshold=[0.5]):
    """
    Init a dictionary containing true positive and false positive per
    class and total gt per class
    :param n_classes: the number of classes
    :param iou_threshold: a dictionary containing empty list for tp, fp and total_gts
    :return: empty score dictionary
    """
    tp_dict = dict()
    for class_id in range(n_classes):
        tp_dict[class_id] = {
            "tp": [[] for _ in range(len(iou_threshold))],
            "fp": [[] for _ in range(len(iou_threshold))],
            "scores": [[] for _ in range(len(iou_threshold))],
            "total_gt": 0
        }
    return tp_dict


def accumulate_tp_fp(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_classes, tp_dict, iou_thresholds):
    """
    Count the number of false positive and true positive for a single image

    :param pred_boxes: the predicted boxes for the image i
    :param pred_labels: the predicted labels for the image i
    :param pred_scores: the predicted scores for the image i
    :param gt_boxes: the ground truth boxes for the image i
    :param gt_classes: the ground truth labels for the image i
    :param tp_dict: dictionary containing accumulated statistics on the test set
    :param iou_thresholds: threshold to use for tp detections

    :return: the updated ap_dict
    """
    # Remove background detections
    keep_idx = pred_boxes[..., :].any(axis=-1) > 0
    pred_boxes = pred_boxes[keep_idx]
    pred_labels = pred_labels[keep_idx]
    pred_scores = pred_scores[keep_idx]
    detected_gts = [[] for _ in range(len(iou_thresholds))]
    # Count number of GT for each class
    for i in range(len(gt_classes)):
        gt_temp = gt_classes[i]
        if gt_temp == -2:
            continue
        tp_dict[gt_temp]["total_gt"] += 1

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = int(pred_labels[i])
        pred_score = pred_scores[i]
        iou = iou2d(pred_box, gt_boxes)
        max_idx_iou = np.argmax(iou)
        gt_box = gt_boxes[max_idx_iou]
        gt_label = gt_classes[max_idx_iou]
        for iou_idx in range(len(iou_thresholds)):
            tp_dict[pred_label]["scores"][iou_idx].append(pred_score)
            tp_dict[pred_label]["tp"][iou_idx].append(0)
            tp_dict[pred_label]["fp"][iou_idx].append(0)

            if pred_label == gt_label and iou[max_idx_iou] >= iou_thresholds[iou_idx] and \
                    list(gt_box) not in detected_gts[iou_idx]:
                tp_dict[pred_label]["tp"][iou_idx][-1] = 1
                detected_gts[iou_idx].append(list(gt_box))
            else:
                tp_dict[pred_label]["fp"][iou_idx][-1] = 1

    return tp_dict


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    :param recall: The recall curve (list).
    :param precision: The precision curve (list).
    :return: The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_ap_class(tp, fp, scores, total_gt):
    """
    Compute the AP given tp, fp, scores and number of ground truth of a class
    See: https://github.com/keshik6/KITTI-2d-object-detection for some part of this code.
    :param tp: true positives list
    :param fp: false positives list
    :param scores: scores list
    :param total_gt: number of total GT in the dataset
    :return: average precision of a class
    """
    ap = 0

    # Array manipulation
    tp = np.array(tp)
    fp = np.array(fp)
    scores = np.array(scores)
    # Sort detection by scores
    indices = np.argsort(-scores)
    fp = fp[indices]
    tp = tp[indices]
    # compute false positives and true positives
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # compute precision/recall
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # compute ap
    average_precision = compute_ap(recall, precision)
    if len(precision) == 0:
        mean_pre = 0.0
    else:
        mean_pre = np.mean(precision)
    if len(recall) == 0:
        mean_rec = 0.0
    else:
        mean_rec = np.mean(recall)
    return average_precision, mean_pre, mean_rec


def compute_f1(precision, recall):
    """
    Compute F1 score
    :param precision:
    :param recall:
    :return: F1-score
    """
    return 2 * (precision * recall) / (precision + recall + 1e-16)


def AP(tp_dict, n_classes, iou_th=[0.5]):
    """
    Create a dictionary containing ap per class

    :param tp_dict:
    :param n_classes:
    :return: A dictionary with the AP for each class
    """
    ap_dict = dict()
    # Ignore 0 class which is BG
    for class_id in range(n_classes):
        tp, fp, scores, total_gt = tp_dict[class_id]["tp"], tp_dict[class_id]["fp"], tp_dict[class_id]["scores"], \
            tp_dict[class_id]["total_gt"]
        ap_dict[class_id] = [[] for _ in range(len(iou_th))]
        ap_dict[class_id] = {
            "AP": [0.0 for _ in range(len(iou_th))],
            "precision": [0.0 for _ in range(len(iou_th))],
            "recall": [0.0 for _ in range(len(iou_th))],
            "F1": [0.0 for _ in range(len(iou_th))]
        }
        for iou_idx in range(len(iou_th)):
            ap_dict[class_id]["AP"][iou_idx], ap_dict[class_id]["precision"][iou_idx], ap_dict[class_id]["recall"][
                iou_idx] \
                = compute_ap_class(tp[iou_idx], fp[iou_idx], scores[iou_idx], total_gt)
            ap_dict[class_id]["F1"][iou_idx] = compute_f1(ap_dict[class_id]["precision"][iou_idx],
                                                          ap_dict[class_id]["recall"][iou_idx])
    ap_dict["mean"] = {
        "AP": [np.mean([ap_dict[class_id]["AP"][iou_th] for class_id in range(n_classes)])
               for iou_th in range(len(ap_dict[0]["AP"]))],
        "precision": [np.mean([ap_dict[class_id]["precision"][iou_th] for class_id in range(n_classes)])
                      for iou_th in range(len(ap_dict[0]["precision"]))],
        "recall": [np.mean([ap_dict[class_id]["recall"][iou_th] for class_id in range(n_classes)])
                   for iou_th in range(len(ap_dict[0]["recall"]))],
        "F1": [np.mean([ap_dict[class_id]["F1"][iou_th] for class_id in range(n_classes)])
               for iou_th in range(len(ap_dict[0]["F1"]))]
    }

    return ap_dict


def write_ap(ap_dict, save_path):
    with open(os.path.join(save_path, "test_results.json"), 'w') as ap_file:
        json.dump(ap_dict, ap_file, indent=2)
