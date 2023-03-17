import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from r2d2.models import model
from r2d2.utils import io_utils, data_utils, bbox_utils, viz_utils

# Set memory growth to avoid GPU to crash
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def main():
    args = io_utils.handle_args_viz()

    config_pth = os.path.join(args.path, "config.json")
    with open(config_pth, 'r') as file:
        config = json.load(file)
    dataset = config["data"]["dataset"]
    labels = config["data"]["labels"]
    labels = ["bg"] + labels
    seed = config["training"]["seed"]
    layout = config["model"]["layout"]
    target_id = args.seq_id
    eval_best = args.eval_best if args.eval_best is not None else False
    # Prepare dataset
    if dataset == "carrada":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    elif dataset == "raddet":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    else:
        raise NotImplementedError("This dataset doesn't exist.")

    anchors = bbox_utils.anchors_generation(config, train=False)
    faster_rcnn_model = model.R2D2(config, anchors)
    if layout == "2D":
        faster_rcnn_model.build(input_shape=(None, 256, 64, 1))
    else:
        fake_input = tf.zeros(shape=(config["training"]["batch_size"], config["model"]["sequence_len"], 256, 64, 1))
        _ = faster_rcnn_model(fake_input)
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

    # ------ Start evaluation with latest ckpt ------ #
    print("Start creation of animation using the latest checkpoint.")
    if eval_best:
        faster_rcnn_model.load_weights(os.path.join(args.path, "best-model.h5"))
    else:
        restore_ckpt(config, log_pth=args.path)

    idx = 0
    start_idx = 0
    end_idx = 250
    for data in batched_test_dataset:
        spectrums, gt_boxes, gt_labels, is_same_seq, seq_id, _, images = data
        if seq_id.numpy()[0] != target_id and dataset == "carrada":
            continue
        else:
            valid_idxs = tf.where(is_same_seq == 1)
            spectrums, gt_boxes, gt_labels = tf.gather_nd(spectrums, valid_idxs), tf.gather_nd(gt_boxes,
                                                                                               valid_idxs), tf.gather_nd(
                gt_labels, valid_idxs)
            if spectrums.shape[0] != 0:
                _, _, _, _, _, decoder_output = faster_rcnn_model(spectrums, training=False)
                pred_boxes, pred_labels, pred_scores = decoder_output
                if start_idx <= idx <= end_idx:
                    if layout == "2D":
                        fig = viz_utils.showCameraRD(camera_image=images[0], rd_spectrum=spectrums[0],
                                                     boxes=pred_boxes[0], labels=pred_labels[0], scores=pred_scores[0],
                                                     gt_boxes=None, gt_labels=None, dataset=dataset,
                                                     class_names=config["data"]["labels"])
                    else:
                        fig = viz_utils.showCameraRD(camera_image=images[0], rd_spectrum=spectrums[0][-1],
                                                     boxes=pred_boxes[0], labels=pred_labels[0], scores=pred_scores[0],
                                                     gt_boxes=None, gt_labels=None, dataset=dataset,
                                                     class_names=config["data"]["labels"])
                    plt.savefig("./images/" + "pred_" + str(idx) + ".png", bbox_inches="tight")
                    plt.clf()
                idx += 1
                if idx >= end_idx:
                    break

    import imageio

    png_dir = './images'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            os.remove(file_path)
    imageio.mimsave('./images/predictions_' + str(target_id) + '.gif', images, fps=4)


if __name__ == "__main__":
    main()
