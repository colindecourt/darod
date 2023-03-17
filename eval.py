import json
import os

import tensorflow as tf
from tqdm import tqdm

from darod.metrics import mAP
from darod.models import model
from darod.utils import io_utils, data_utils, bbox_utils

# Set memory growth to avoid GPU to crash
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    args = io_utils.handle_args_eval()
    # summary_writer = tf.summary.create_file_writer(args.path)

    config_pth = os.path.join(args.path, "config.json")
    with open(config_pth, 'r') as file:
        config = json.load(file)
    dataset = config["data"]["dataset"]
    labels = config["data"]["labels"]
    labels = ["bg"] + labels
    seed = config["training"]["seed"]

    # Prepare dataset
    if dataset == "carrada":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    elif dataset == "raddet":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    else:
        raise NotImplementedError("This dataset doesn't exist.")

    len_test_dataset = data_utils.get_total_item_size(dataset_info, "test")
    anchors = bbox_utils.anchors_generation(config, train=False)

    faster_rcnn_model = model.DAROD(config, anchors)
    faster_rcnn_model.build(input_shape=(None, 256, 64, config["model"]["input_size"][-1]))
    faster_rcnn_model.summary()

    def restore_ckpt(config, log_pth):
        optimizer = config["training"]["optimizer"]
        lr = config["training"]["lr"]
        use_scheduler = config["training"]["scheduler"]
        momentum = config["training"]["momentum"]
        if optimizer == "SGD":
            optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        elif optimizer == "adad":
            optimizer = tf.optimizers.Adadelta(learning_rate=1.0)
        elif optimizer == "adag":
            optimizer = tf.optimizers.Adagrad(learning_rate=lr)
        else:
            raise NotImplemented("Not supported optimizer {}".format(optimizer))
        global_step = tf.Variable(1, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=faster_rcnn_model, step=global_step)
        manager = tf.train.CheckpointManager(ckpt, log_pth, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)

    def test_step(batched_test_dataset, map_iou_threshold_list=None, num_classes=3):
        """
        Test step for model evaluation.

        :param batched_test_dataset: test dataset
        :param map_iou_threshold_list: list of threshold to test over the predictions
        :return: dictionary with AP per class at different thresholds
        """
        if map_iou_threshold_list is None:
            map_iou_threshold_list = [0.1, 0.3, 0.5, 0.7]

        pbar = tqdm(total=int(len_test_dataset))
        tp_dict = mAP.init_tp_dict(num_classes, iou_threshold=map_iou_threshold_list)
        import time

        total_time = 0
        for input_data in batched_test_dataset:
            spectrums, gt_boxes, gt_labels, is_same_seq, _, _, images = input_data
            # Take valid data (frames from the same record in a window)
            valid_idxs = tf.where(is_same_seq == 1)
            spectrums, gt_boxes, gt_labels = tf.gather_nd(spectrums, valid_idxs), tf.gather_nd(gt_boxes,
                                                                                               valid_idxs), tf.gather_nd(
                gt_labels, valid_idxs)
            if spectrums.shape[0] != 0:
                start = time.time()
                _, _, _, _, _, decoder_output = faster_rcnn_model(spectrums, training=False)
                end = time.time() - start
                total_time += end
                pred_boxes, pred_labels, pred_scores = decoder_output
                pred_labels = pred_labels - 1
                gt_labels = gt_labels - 1
                for batch_id in range(gt_labels.shape[0]):
                    tp_dict = mAP.accumulate_tp_fp(pred_boxes.numpy()[batch_id], pred_labels.numpy()[batch_id],
                                                   pred_scores.numpy()[batch_id], gt_boxes.numpy()[batch_id],
                                                   gt_labels.numpy()[batch_id], tp_dict,
                                                   iou_thresholds=map_iou_threshold_list)
            pbar.update(1)

        print("Inference time on GPU: {} s".format(total_time / 1391))
        ap_dict = mAP.AP(tp_dict, num_classes, iou_th=map_iou_threshold_list)
        return ap_dict

    # ------ Start evaluation with latest ckpt ------ #
    print("Start evaluation of the model with latest checkpoint.")

    restore_ckpt(config, log_pth=args.path)
    out_ap = test_step(batched_test_dataset, num_classes=len(labels) - 1)
    mAP.write_ap(out_ap, args.path)


if __name__ == "__main__":
    main()
