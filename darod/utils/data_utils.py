import tensorflow as tf
import tensorflow_datasets as tfds


def make_window_dataset(ds, window_size=5, shift=1, stride=1, viz=False):
    """
    Create a window dataset of window size and flatten it to
    return a window of data for sequential processing. Only for temporal processing.
    :param ds: the dataset to window
    :param window_size: the size of the window
    :param shift: the shit between windows
    :param stride: the stride between windows
    :param viz: return camera images or not
    :return: a new dataset with windowed data
    """
    windows = ds.window(window_size, shift=shift, stride=stride)

    def sub_to_batch(sub):
        """Return a batched zip dataset containing the current sequence_id,
        window_size spectrums and the associated bounding boxes and labels
        inputs:
            sub: sub dataset to batch (or window)
        output: a zip dataset with sequence id, spectrum, bboxes and labels
        """
        if viz:
            return tf.data.Dataset.zip((sub['sequence/id'].padded_batch(window_size, padded_shapes=[],
                                                                        padding_values=tf.constant(-1, tf.int64),
                                                                        drop_remainder=True),
                                        sub['spectrum'].padded_batch(window_size, padded_shapes=[None, None],
                                                                     padding_values=tf.constant(0, tf.float32),
                                                                     drop_remainder=True),
                                        sub['objects']['bbox'].padded_batch(window_size, padded_shapes=[None, 4],
                                                                            padding_values=tf.constant(0, tf.float32),
                                                                            drop_remainder=True),
                                        sub['objects']['label'].padded_batch(window_size, padded_shapes=[None],
                                                                             padding_values=tf.constant(-2, tf.int64),
                                                                             drop_remainder=True),
                                        sub["image"].padded_batch(window_size, padded_shapes=[None, None, 3],
                                                                  padding_values=tf.constant(0, tf.uint8),
                                                                  drop_remainder=True),
                                        sub['spectrum/filename'].padded_batch(window_size, padded_shapes=[],
                                                                              padding_values=tf.constant("-1",
                                                                                                         tf.string),
                                                                              drop_remainder=True)))

        return tf.data.Dataset.zip((sub['sequence/id'].padded_batch(window_size, padded_shapes=[],
                                                                    padding_values=tf.constant(-1, tf.int64),
                                                                    drop_remainder=True),
                                    sub['spectrum'].padded_batch(window_size, padded_shapes=[None, None],
                                                                 padding_values=tf.constant(0, tf.float32),
                                                                 drop_remainder=True),
                                    sub['objects']['bbox'].padded_batch(window_size, padded_shapes=[None, 4],
                                                                        padding_values=tf.constant(0, tf.float32),
                                                                        drop_remainder=True),
                                    sub['objects']['label'].padded_batch(window_size, padded_shapes=[None],
                                                                         padding_values=tf.constant(-2, tf.int64),
                                                                         drop_remainder=True),
                                    sub['spectrum/filename'].padded_batch(window_size, padded_shapes=[],
                                                                          padding_values=tf.constant("-1", tf.string),
                                                                          drop_remainder=True)))

    windows = windows.flat_map(sub_to_batch)
    return windows


def data_preprocessing(sequence_id, spectrums, gt_boxes, gt_labels, filename, config, train=True, viz=False,
                       image=None):
    """
    Preprocess the data before training i.e. normalize spectrums between 0 and 1, standardize the
    data, augment it if necessary.
    :param sequence_id: id of the current sequence
    :param spectrums: serie of sequence_len spectrums
    :param gt_boxes: serie of ground truth boxes
    :param gt_labels: serie of ground truth labels
    :param filename: name of the file to process
    :param config: config file
    :param train: transform data for training (True) or for test/val (False)
    :param viz: return camera image if train is False and viz is True and only the last spectrum
    :param image: return camera image if True
    :return: spectrum, ground truth, labels, sequence id, filename, camera image (optional)
    """
    apply_augmentation = config["training"]["use_aug"]
    layout = config["model"]["layout"]

    # Only take the last label of the window i.e. the spectrum we want to detect objects
    gt_labels = tf.cast(gt_labels[-1] + 1, dtype=tf.int32)
    gt_boxes = gt_boxes[-1]

    if layout == "2D":
        spectrums = spectrums[-1]

    spectrums = tf.expand_dims(spectrums, -1)
    if config["training"]["pretraining"] == "imagenet":
        mean = [0.485, 0.456, 0.406]
        variance = [0.229, 0.224, 0.225]
        spectrums = (spectrums - tf.reduce_min(spectrums, axis=[0, 1])) / (
                    tf.reduce_max(spectrums, axis=[0, 1]) - tf.reduce_min(spectrums, axis=[0,
                                                                                           1]))  # Normalize spectrums between 0 and 1
        spectrums = (spectrums - mean) / variance  # Standardize spectrums
    elif config["model"]["backbone"] in ["resnet", "efficientnet", "mobilenet", "vgg16"]:
        spectrums = (spectrums - config["data"]["data_mean"]) / config["data"]["data_std"]
        spectrums = tf.image.grayscale_to_rgb(spectrums)
    else:
        # Standardize input
        spectrums = (spectrums - config["data"]["data_mean"]) / config["data"]["data_std"]

    if apply_augmentation and train:
        random_val = get_random_bool()
        if tf.greater_equal(random_val, 0) and tf.less(random_val, 0.25):
            spectrums, gt_boxes = flip_horizontally(spectrums, gt_boxes)
            print("Horizontal flipping")
        elif tf.greater_equal(random_val, 0.25) and tf.less(random_val, 0.5):
            spectrums, gt_boxes = flip_vertically(spectrums, gt_boxes)
            print("Vertical flipping")

    if viz and image is not None:
        image = image[-1]
    #
    is_same_seq = tf.cast(sequence_id[0] == sequence_id[-1], tf.int64)

    if viz:
        return spectrums, gt_boxes, gt_labels, is_same_seq, sequence_id[-1], filename, image

    return spectrums, gt_boxes, gt_labels, is_same_seq, sequence_id[-1], filename


def transforms(spectrum, config):
    """
    Standardize data for each frame of the sequence.
    :param spectrum: input data to standardize
    :param config: mean of the dataset
    :return:  standardized spectrums
    """
    spectrum = tf.expand_dims(spectrum, axis=-1)
    spectrum = (spectrum - config["data"]["data_mean"]) / config["data"]["data_std"]

    return spectrum


def get_random_bool():
    """
    Generating random boolean.
    :return: random boolean 0d tensor
    """
    return tf.random.uniform((), dtype=tf.float32)


def randomly_apply_operation(operation, img, gt_boxes):
    """
    Randomly applying given method to image and ground truth boxes.
    :param operation: callable method
    :param img: (height, width, depth)
    :param gt_boxes: (ground_truth_object_count, [y1, x1, y2, x2])
    :return: modified_or_not_img = (final_height, final_width, depth)
             modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )


def flip_horizontally(img, gt_boxes):
    """
    Flip image horizontally and adjust the ground truth boxes.
    :param img: (height, width, depth)
    :param gt_boxes: (ground_truth_object_count, [y1, x1, y2, x2])
    :return: modified_img = (height, width, depth)
             modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                 1.0 - gt_boxes[..., 3],
                                 gt_boxes[..., 2],
                                 1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def flip_vertically(img, gt_boxes):
    """
    Flip image horizontally and adjust the ground truth boxes.
    :param img: (height, width, depth)
    :param gt_boxes: (ground_truth_object_count, [y1, x1, y2, x2])
    :return: modified_img = (height, width, depth)
             modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_up_down(img)
    flipped_gt_boxes = tf.stack([1.0 - gt_boxes[..., 2],
                                 gt_boxes[..., 1],
                                 1.0 - gt_boxes[..., 0],
                                 gt_boxes[..., 3]], -1)
    return flipped_img, flipped_gt_boxes


def get_dataset(name, version, split, data_dir):
    """
    Get tensorflow dataset split and info.
    :param name: name of the dataset, carrada/raddet
    :param version:  version of the dataset
    :param split: data split string, should be one of ["train", "validation", "test"]
    :param data_dir: read/write path for tensorflow datasets
    :return: dataset: tensorflow dataset split
             info: tensorflow dataset info
    """
    dataset, info = tfds.load(name=name + ":" + version, data_dir=data_dir, split=split, as_supervised=False,
                              with_info=True)
    return dataset, info


def get_total_item_size(info, split):
    """
    Get total item size for given split.
    :param info: tensorflow dataset info
    :param split:  data split string, should be one of ["train", "test"]
    :return: total_item_size = number of total items
    """
    return info.splits[split].num_examples


def get_labels(info):
    """
    Get label names list.
    :param info: tensorflow dataset info
    :return:  labels = [labels list]
    """
    return info.features["objects"]["label"].names


def prepare_dataset(split, config, seed):
    """
    Prepare dataset for training
    :param split: train/test/val
    :param config: configuration dictionary with training settings
    :param seed: seed
    :return: batched dataset and dataset info
    """
    #
    train = True if split == "train" or split == "train[:90%]" else False
    viz = True if split == "test" else False
    buffer_size = 4000 if config["data"]["dataset"] == "carrada" else 10000
    dataset, dataset_info = get_dataset(name=config["data"]["dataset"], version=config["data"]["dataset_version"],
                                        split=split, data_dir=config["data"]["tfds_path"])

    dataset_w = make_window_dataset(dataset, window_size=config["model"]["sequence_len"], viz=viz)

    #
    if viz:
        dataset_w = dataset_w.map(
            lambda seq_ids, spectrums, gt_boxes, gt_labels, image, filename: data_preprocessing(seq_ids, spectrums,
                                                                                                gt_boxes, gt_labels,
                                                                                                filename, config,
                                                                                                train=train, viz=True,
                                                                                                image=image))
        padding_values = (
        tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32), tf.constant(-1, tf.int64),
        tf.constant(-1, tf.int64), tf.constant("-1", tf.string), tf.constant(0, tf.uint8))
    #
    else:
        dataset_w = dataset_w.map(
            lambda seq_ids, spectrums, gt_boxes, gt_labels, filename: data_preprocessing(seq_ids, spectrums, gt_boxes,
                                                                                         gt_labels, filename, config,
                                                                                         train=train))
        padding_values = (
        tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32), tf.constant(-1, tf.int64),
        tf.constant(-1, tf.int64), tf.constant("-1", tf.string))
    #
    if train:
        print("Shuffle the dataset")
        dataset_w = dataset_w.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)

    if config["model"]["layout"] == "2D":
        if viz:
            padded_shapes = (
            [None, None, config["model"]["input_size"][-1]], [None, 4], [None], [], [], [None], [None, None, 3])
        else:
            padded_shapes = ([None, None, config["model"]["input_size"][-1]], [None, 4], [None], [], [], [None])
    else:
        if viz:
            padded_shapes = (
            [None, None, None, config["model"]["input_size"][-1]], [None, 4], [None], [], [], [None], [None, None, 3])
        else:
            padded_shapes = ([None, None, None, config["model"]["input_size"][-1]], [None, 4], [None], [], [], [None])

    if split == "test":
        batched_dataset = dataset_w.padded_batch(1, padded_shapes=padded_shapes,
                                                 padding_values=padding_values,
                                                 drop_remainder=True)
    else:
        batched_dataset = dataset_w.padded_batch(config["training"]["batch_size"], padded_shapes=padded_shapes,
                                                 padding_values=padding_values,
                                                 drop_remainder=True)

    return batched_dataset, dataset_info
