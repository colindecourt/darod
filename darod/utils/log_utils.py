import os

import tensorflow as tf


def get_log_path(model_type, exp, backbone, custom_postfix=""):
    """
    Generating log path from model_type value for tensorboard.
    :param model_type: the model used
    :param exp: experiment name
    :param backbone: the backbone used
    :param custom_postfix:  any custom string for log folder name
    :return: tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    log_path = "logs/{}_{}{}/{}/".format(model_type, backbone, custom_postfix, exp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


def tensorboard_val_stats(writer, ap_dict, labels, step):
    """
    Write evaluation metric to tensorboard
    :param writer: TB writer
    :param ap_dict: dictionary with AP
    :param labels: labels list
    :param step: epoch number
    :return:
    """
    with writer.as_default():
        for class_id in ap_dict:
            if class_id != "mean":
                tf.summary.scalar("mAP@0.5/" + labels[class_id], ap_dict[class_id]["AP"][0], step=step)
                tf.summary.scalar("precision@0.5/" + labels[class_id], ap_dict[class_id]["precision"][0], step=step)
                tf.summary.scalar("recall@0.5/" + labels[class_id], ap_dict[class_id]["recall"][0], step=step)
                tf.summary.scalar("F1@0.5/" + labels[class_id], ap_dict[class_id]["F1"][0], step=step)
        tf.summary.scalar("mAP@0.5/Mean", ap_dict["mean"]["AP"][0], step=step)
        tf.summary.scalar("precision@0.5/Mean", ap_dict["mean"]["precision"][0], step=step)
        tf.summary.scalar("recall@0.5/Mean", ap_dict["mean"]["recall"][0], step=step)
        tf.summary.scalar("F1@0.5/Mean", ap_dict["mean"]["F1"][0], step=step)
