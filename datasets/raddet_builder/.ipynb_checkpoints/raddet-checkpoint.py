"""raddet dataset."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from utils import loader, helper

# TODO(raddet): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(raddet): BibTeX citation
_CITATION = """
"""


class Raddet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for raddet dataset."""
    VERSION = tfds.core.Version('3.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        # '2.0.0': "Sorted sequences",
        '2.0.0': "Min max normalization",
        '3.0.0': "No normalization"
    }

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
                'spectrum/id': tf.int64,
                'sequence/id': tf.int64,
                'objects': tfds.features.Sequence({
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    'id': tf.int64,
                    'label': tfds.features.ClassLabel(names=["person", "bicycle", "car", "motorcycle", "bus", "truck"])
                    # Keep 0 class for BG
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
        train_path = "/opt/dataset_ssd/radar1/RADDet dataset/train/"
        test_path = "/opt/dataset_ssd/radar1/RADDet dataset/test/"
        # TODO(carrada): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(train_path, 'train'),
            'test': self._generate_examples(test_path, 'test'),
        }

    def _generate_examples(self, path, input_type="RD"):
        classes_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        RAD_sequences = loader.readSequences(path)
        count = 0
        a_id = 0
        s_id = 0
        global_mean_log = 3.2438383
        global_variance_log = 6.8367246
        global_max_log = 10.0805629
        global_min_log = 0.0
        while count < len(RAD_sequences):
            objects = []
            RAD_filename = RAD_sequences[count]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path.")
            RAD_data = helper.complexTo2channels(RAD_complex)
            # Normalize data
            # RAD_data = (RAD_data - global_mean_log) / global_variance_log
            # RAD_data = (RAD_data - global_min_log) / (global_max_log - global_min_log)
            # Load GT instances
            gt_filename = loader.gtfileFromRADfile(RAD_filename, path)
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")
            # Get RD spectrum
            RD_data = helper.getSumDim(RAD_data, target_axis=1)
            # Get RD bboxes
            bboxes, classes = helper.readAndEncodeGtRD(gt_instances, RD_data.shape)
            seq_id = RAD_filename.split('/')[-2].split('_')[-1]
            for (box, class_) in zip(bboxes, classes):
                bbox, area = helper.buildTfdsBoxes(box)
                objects.append({
                    'bbox': bbox,
                    'label': classes_list.index(class_),
                    'area': area,
                    'id': a_id
                })
                a_id += 1
            image_filename = loader.imgfileFromRADfile(RAD_filename, path)
            example = {
                'spectrum': RD_data.astype(np.float32),
                'spectrum/filename': RAD_filename,
                'sequence/id': int(seq_id),
                'image': image_filename,
                'spectrum/id': count,
                'objects': objects
            }
            count += 1

            yield count, example
