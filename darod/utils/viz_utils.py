import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import tensorflow as tf
from PIL import Image
from matplotlib.patches import Rectangle

from .bbox_utils import denormalize_bboxes


def set_seaborn_style():
    sb.set_style(style='darkgrid')
    colors = ["#FCB316", "#6DACDE", "#BFD730", "#320E3B", "#E56399", "#393D3F", "#A97C73", "#AF3E4D", "#2F4B26"]
    sb.set_context("paper")
    sb.set_palette(sb.color_palette(colors))


def to_PIL(spectrum):
    spectrum = ((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)))
    cm = plt.get_cmap('plasma')
    colored_spectrum = cm(spectrum)
    pil_spectrum = Image.fromarray((colored_spectrum[:, :, :3] * 255).astype(np.uint8))
    return pil_spectrum


def norm2Image(array):
    """ normalize to image format (uint8) """
    norm_sig = plt.Normalize()
    img = plt.cm.plasma(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img


def drawRDboxes(boxes, labels, ax, use_facecolor=True, color="#E56399", class_names=["pedestrian", "bicyclist", "car"]):
    """Draw bounding boxes on RD spectrum
    Set labels to None for RPN boxes only."""
    if use_facecolor:
        facecolor = color
    else:
        facecolor = "none"

    if labels is not None:
        for (box, label) in zip(boxes, labels):
            if label != -1 or tf.reduce_sum(box, axis=0) != 0:
                print(box)
                y1, x1, y2, x2 = denormalize_bboxes(box, 256, 64)
                h, w = y2 - y1, x2 - x1
                rect = Rectangle((x1, y1), width=w, height=h, linewidth=1.0, alpha=0.9,
                                 linestyle="dashed", color=color, facecolor=facecolor, edgecolor=color, fill=False)
                ax.add_patch(rect)
                label = int(label.numpy())
                ax.text(x1 - 1 - (x2 - x1), y1 - 3 - (y2 - y1), class_names[label - 1], size=10,
                        verticalalignment='baseline',
                        color='w', backgroundcolor="none",
                        bbox={'facecolor': color, 'alpha': 0.5,
                              'pad': 2, 'edgecolor': 'none'})
    else:
        for box in boxes:
            if tf.reduce_sum(box, axis=0) != 0:
                y1, x1, y2, x2 = denormalize_bboxes(box, 256, 64)
                h, w = y2 - y1, x2 - x1
                rect = Rectangle((x1, y1), width=w, height=h, linewidth=1.0, alpha=0.9,
                                 linestyle="dashed", color=color, facecolor=facecolor, fill=False)
                ax.add_patch(rect)


def drawRDboxes_with_scores(boxes, labels, scores, ax, use_facecolor=True, color="#E56399",
                            class_names=["pedestrian", "bicyclist", "car"]):
    """Draw bounding boxes on RD spectrum with scores.
    This function must be use ONLY with predictions"""
    if use_facecolor:
        facecolor = color
    else:
        facecolor = "none"

    for (box, label, score) in zip(boxes, labels, scores):
        if label != 0 or tf.reduce_sum(box, axis=0) != 0:
            y1, x1, y2, x2 = denormalize_bboxes(box, 256, 64)
            h, w = y2 - y1, x2 - x1
            score = np.round(score.numpy(), 4)
            rect = Rectangle((x1, y1), width=w, height=h, linewidth=1.0, alpha=0.9,
                             linestyle="dashed", color=color, facecolor=facecolor, edgecolor=color, fill=False)
            ax.add_patch(rect)
            label = int(label.numpy())
            ax.text(x1 + 1, y1 - 3, class_names[label - 1] + " " + str(score), size=10, verticalalignment='baseline',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})


def showCameraRD(camera_image, rd_spectrum, boxes, labels, class_names, scores=None, gt_boxes=None, gt_labels=None,
                 dataset="carrada"):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    rd_spectrum = norm2Image(rd_spectrum.numpy().squeeze())
    ax1.imshow(rd_spectrum)
    title = "Range-Doppler"
    ax1.set_xticks([0, 16, 32, 48, 63])
    ax1.set_xticklabels([-13, -6.5, 0, 6.5, 13])
    ax1.set_yticks([0, 64, 128, 192, 255])
    ax1.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax1.set_xlabel("velocity (m/s)")
    ax1.set_ylabel("range (m)")
    ax1.set_title(title)
    if boxes is not None and labels is not None and scores is None:
        drawRDboxes(boxes, labels, ax1, class_names=class_names, color="#02D0F0")
    elif scores is not None and boxes is not None and labels is not None:
        drawRDboxes_with_scores(boxes, labels, scores, ax1, class_names=class_names, color="#02D0F0")
    else:
        raise ValueError("Please use at least boxes and labels as arguments.")
    if gt_boxes is not None and gt_labels is not None:
        drawRDboxes(gt_boxes, gt_labels, ax1, class_names=class_names, color="#ECE8EF")
    # Show only left camera image
    if dataset == "raddet":
        ax2.imshow(camera_image[:, :camera_image.shape[1] // 2, ...])
    else:
        ax2.imshow(camera_image)
    ax2.axis('off')
    ax2.set_title("camera image")
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
