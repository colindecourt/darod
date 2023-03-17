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

# TODO(carrada): BibTeX citation
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


class Carrada(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for carrada dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(carrada): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'spectrum': tfds.features.Tensor(shape=(256, 64), dtype=tf.float64),
                'image': tfds.features.Image(shape=(None, None, 3)),
                'spectrum/filename': tfds.features.Text(),
                'image/filename': tfds.features.Text(),
                'spectrum/id': tf.int64,
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
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(carrada): Downloads the data and defines the splits
        path = "/datasets/radar1/Carrada/"

        # TODO(carrada): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path, 'train'),
            'test': self._generate_examples(path, 'test')
        }

    def _generate_examples(self, path, split, input_type="RD"):
        """Yields examples."""
        annotations_path = os.path.join(path, 'annotations_frame_oriented.json')
        annotation, seq_ref = self._load_annotation(path, annotations_path)

        spectrum_type = 'range_doppler' if input_type == "RD" else 'range_angle'
        h, w = (256, 64) if input_type == "RD" else (256, 256)
        if split == "train":
            sequences = [seq for seq in seq_ref if seq_ref[seq]['split'] == "Train"]
        else:
            sequences = [seq for seq in seq_ref if
                         (seq_ref[seq]['split'] == "Validation" or seq_ref[seq]['split'] == "Test")]

        s_id = 0
        a_id = 0
        for sequence in sequences:
            # Load all annotations for current sequence
            current_annotations = annotation[sequence]
            # Iterate over all frames in the current sequence
            for frame in current_annotations:
                objects = []
                # Load all annotations fdor all instances in the frame
                sequence_annot = current_annotations[frame]

                # Define file_name for camera and spectrum data
                spectrum_fn = os.path.join(sequence, spectrum_type + '_numpy', frame.zfill(6) + '.npy')
                image_fn = os.path.join(sequence, 'camera_images', frame.zfill(6) + '.jpg')

                if len(current_annotations[frame]) != 0:
                    instances_annot = current_annotations[frame]
                    for instance in instances_annot:
                        in_box = instances_annot[instance][spectrum_type]['box']
                        bbox = self._build_bbox(in_box, h, w)
                        area = self._compute_area(in_box)
                        label = instances_annot[instance][spectrum_type]['label']
                        objects.append({
                            'bbox': bbox,
                            'label': label - 1,
                            'area': area,
                            'id': a_id
                        })
                        a_id += 1
                    example = {
                        'spectrum': np.load(os.path.join(path, spectrum_fn)),
                        'spectrum/filename': spectrum_fn,
                        'image': os.path.join(path, image_fn),
                        'image/filename': image_fn,
                        'spectrum/id': s_id,
                        'objects': objects
                    }
                    s_id += 1
                    yield spectrum_fn, example

    def _build_bbox(self, box, h, w):
        x1, y1, x2, y2 = box[0][1], box[0][0], box[1][1], box[1][0]

        return tfds.features.BBox(
            ymin=y1 / h,
            xmin=x1 / w,
            ymax=y2 / h,
            xmax=x2 / w,
        )

    def _compute_area(self, box):
        x1, y1, x2, y2 = box[0][1], box[0][0], box[1][1], box[1][0]
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
