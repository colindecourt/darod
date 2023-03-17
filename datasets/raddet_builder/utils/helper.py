import numpy as np
import tensorflow_datasets as tfds


def readAndEncodeGtRD(gt_instance, rd_shape):
    """ read gt_instance and return bbox
    and class for RD """
    x_shape, y_shape = rd_shape[1], rd_shape[0]
    boxes = gt_instance["boxes"]
    classes = gt_instance["classes"]
    new_boxes = []
    new_classes = []
    for (box, class_) in zip(boxes, classes):

        yc, xc, h, w = box[0], box[2], box[3], box[5]
        y1, y2, x1, x2 = int(yc - h / 2), int(yc + h / 2), int(xc - w / 2), int(xc + w / 2)
        if x1 < 0:
            # Create 2 boxes
            x1 += x_shape
            box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
            box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
            #
            new_boxes.append(box1)
            new_classes.append(class_)
            #
            new_boxes.append(box2)
            new_classes.append(class_)
        elif x2 >= x_shape:
            x2 -= x_shape
            box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
            box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
            #
            new_boxes.append(box1)
            new_classes.append(class_)
            #
            new_boxes.append(box2)
            new_classes.append(class_)
        else:
            new_boxes.append([y1 / y_shape, x1 / x_shape, y2 / y_shape, x2 / x_shape])
            new_classes.append(class_)

    return new_boxes, new_classes


def buildTfdsBoxes(box):
    ymin, xmin, ymax, xmax = box
    area = (xmax - xmin) * (ymax - ymin)
    return tfds.features.BBox(
        ymin=ymin,
        xmin=xmin,
        ymax=ymax,
        xmax=xmax
    ), area


def complexTo2channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array


def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array


def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array


def getSumDim(target_array, target_axis):
    """ sum up one dimension """
    output = np.sum(target_array, axis=target_axis)
    return output
