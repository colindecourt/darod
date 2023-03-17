"""carrada dataset."""

import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
# CARRADA dataset 

The CARRADA dataset contains camera images, raw radar data and generated annotations for scene understanding
in autonomous driving. 
*Full dataset can be found here: https://github.com/valeoai/carrada_dataset*
"""

_CITATION = """
@misc{ouaknine2020carrada,
    title={CARRADA Dataset: Camera and Automotive Radar with Range-Angle-Doppler Annotations},
    author={A. Ouaknine and A. Newson and J. Rebut and F. Tupin and P. Pérez},
    year={2020},
    eprint={2005.01456},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
"""


# @dataclasses.dataclass
# class CarradaConfig(tfds.core.BuilderConfig):
#   sequence_length = 1


class Carrada(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carrada dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    # BUILDER_CONFIG = [
    #     # `name` (and optionally `description`) are required for each config
    #     CarradaConfig(name='sl_1', description='Sequence length of 1', sequence_length=1),
    #     CarradaConfig(name='sl_2', description='Sequence length of 2', sequence_length=2),
    #     CarradaConfig(name='sl_3', description='Sequence length of 3', sequence_length=3),
    #     CarradaConfig(name='sl_4', description='Sequence length of 4', sequence_length=4),
    #     CarradaConfig(name='sl_5', description='Sequence length of 5', sequence_length=5),
    #     CarradaConfig(name='sl_6', description='Sequence length of 6', sequence_length=6),
    #     CarradaConfig(name='sl_7', description='Sequence length of 7', sequence_length=7),
    #     CarradaConfig(name='sl_8', description='Sequence length of 8', sequence_length=8),
    #     CarradaConfig(name='sl_9', description='Sequence length of 9', sequence_length=9),
    #     CarradaConfig(name='sl_10', description='Sequence length of 10', sequence_length=10),
    # ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # No need to keep 'spectrum' and 'image' field. Serialize their filenames is
                # enough to load it and allows to save disk space.
                'spectrum': tfds.features.Tensor(shape=(256, 64), dtype=tf.float32),
                'image': tfds.features.Image(shape=(None, None, 3)),
                'spectrum/filename': tfds.features.Text(),
                'image/filename': tfds.features.Text(),
                'spectrum/id': tf.int64,
                'sequence/id': tf.int64,
                'objects': tfds.features.Sequence({
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    'id': tf.int64,
                    'label': tfds.features.ClassLabel(names=["pedestrian", "bicyclist", "car"])  # Keep 0 class for BG
                }),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('spectrum', 'objects'),  # Set to `None` to disable
            homepage="",
            citation=_CITATION,
            disable_shuffling=True,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(carrada): Downloads the data and defines the splits
        path = "/opt/dataset_ssd/radar1/Carrada/"

        # TODO(carrada): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path, 'train'),
            'test': self._generate_examples(path, 'test'),
            'val': self._generate_examples(path, 'val'),
        }

    def _generate_examples(self, path, split, input_type="RD"):
        """Yields examples."""

        exceptions = ['2019-09-16-12-52-12/camera_images/001015.jpg', '2019-09-16-12-52-12/camera_images/001016.jpg',
                      '2019-09-16-13-14-29/camera_images/000456.jpg', '2020-02-28-12-12-16/camera_images/000489.jpg',
                      '2020-02-28-12-12-16/camera_images/000490.jpg', '2020-02-28-12-12-16/camera_images/000491.jpg',
                      '2020-02-28-13-06-53/camera_images/000225.jpg', '2020-02-28-13-06-53/camera_images/000226.jpg',
                      '2020-02-28-13-10-51/camera_images/000247.jpg', '2020-02-28-13-10-51/camera_images/000248.jpg',
                      '2020-02-28-13-10-51/camera_images/000249.jpg', '2020-02-28-13-13-43/camera_images/000216.jpg',
                      '2020-02-28-13-13-43/camera_images/000217.jpg', '2020-02-28-13-15-36/camera_images/000169.jpg',
                      '2020-02-28-12-13-54/camera_images/000659.jpg', '2020-02-28-12-13-54/camera_images/000660.jpg',
                      '2020-02-28-13-07-38/camera_images/000279.jpg', '2020-02-28-13-07-38/camera_images/000280.jpg']

        annotations_path = os.path.join(path, 'annotations_frame_oriented.json')
        annotation, seq_ref = self._load_annotation(path, annotations_path)

        spectrum_type = 'range_doppler' if input_type == "RD" else 'range_angle'
        h, w = (256, 64) if input_type == "RD" else (256, 256)
        #
        # sequence_length = self.builder_config.sequence_length
        if split == "train":
            sequences = [seq for seq in seq_ref if seq_ref[seq]['split'] == "Train"]
        elif split == "val":
            sequences = [seq for seq in seq_ref if seq_ref[seq]['split'] == "Validation"]
        elif split == "test":
            sequences = [seq for seq in seq_ref if seq_ref[seq]['split'] == "Test"]

        s_id = 0
        a_id = 0
        sequence_id = 0
        for sequence in sequences:
            sequence_id += 1
            # Load all annotations for current sequence
            current_annotations = annotation[sequence]
            # Iterate over all frames in the current sequence
            for frame in current_annotations:
                objects = []
                # Load all annotations fdor all instances in the frame
                sequence_annot = current_annotations[frame]

                # Define file_name for camera and spectrum data
                spectrum_fn = os.path.join(sequence, spectrum_type + '_processed', frame.zfill(6) + '.npy')
                image_fn = os.path.join(sequence, 'camera_images', frame.zfill(6) + '.jpg')

                if image_fn not in exceptions:
                    if len(current_annotations[frame]) != 0:
                        instances_annot = current_annotations[frame]
                        for instance in instances_annot:
                            in_polygon = instances_annot[instance][spectrum_type]['dense']
                            bbox, area = self._build_bbox(in_polygon, h, w)
                            label = instances_annot[instance][spectrum_type]['label']
                            objects.append({
                                'bbox': bbox,
                                'label': label - 1,
                                'area': area,
                                'id': a_id
                            })
                            a_id += 1

                        example = {
                            'spectrum': np.load(os.path.join(path, spectrum_fn)).astype(np.float32),
                            'spectrum/filename': spectrum_fn,
                            'sequence/id': sequence_id,
                            'image': os.path.join(path, image_fn),
                            'image/filename': image_fn,
                            'spectrum/id': s_id,
                            'objects': objects
                        }
                        s_id += 1

                        yield s_id - 1, example

    def _build_bbox(self, polygon, h, w):
        x, y = [], []
        for y_, x_ in polygon:
            x.append(x_)
            y.append(y_)
        y1, x1, y2, x2 = min(y), min(x), max(y), max(x)
        bbox = [y1, x1, y2, x2]
        area = self._compute_area(bbox)
        return tfds.features.BBox(
            ymin=y1 / h,
            xmin=x1 / w,
            ymax=y2 / h,
            xmax=x2 / w,
        ), area

    def _compute_area(self, box):
        y1, x1, y2, x2 = box
        return (x2 - x1) * (y2 - y1)

    def _load_annotation(self, path, annotation_path):
        """
        Load annotations file and sequence distribution file
        """
        with open(annotation_path) as annotations:
            annotations = json.load(annotations)
        with open(os.path.join(path, "data_seq_ref.json")) as seq_ref:
            seq_ref = json.load(seq_ref)
        return annotations, seq_ref
