import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils
from waymo_open_dataset.utils import frame_utils


class DataLabelReader():
    def __init__(self, parquetName):
        # parquetName: /path/to/context_name.parquet
        self.cam_segmentation_df = dd.read_parquet(parquetName)
        self.frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
        self.camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                                            open_dataset.CameraName.FRONT_LEFT,
                                            open_dataset.CameraName.FRONT,
                                            open_dataset.CameraName.FRONT_RIGHT,
                                            open_dataset.CameraName.SIDE_RIGHT]
    def ungroup_row(self, key_names: Sequence[str],
                key_values: Sequence[str],
                row: dd.DataFrame) -> Iterator[Dict[str, Any]]:
        """Splits a group of dataframes into individual dicts."""
        keys = dict(zip(key_names, key_values))
        cols, cells = list(zip(*[(col, cell) for col, cell in row.items()]))
        for values in zip(*cells):
            yield dict(zip(cols, values), **keys)
    
    def addCamSegList(self):
        cam_segmentation_per_frame_df = self.cam_segmentation_df.groupby(self.frame_keys, group_keys=False).agg(list)
        self.cam_segmentation_list = []
        for i, (key_values, r) in enumerate(cam_segmentation_per_frame_df.iterrows()):
            # Read three sequences of 5 camera images for this demo.
            if i >= 10:
                break
            # Store a segmentation label component for each camera.
            self.cam_segmentation_list.append(
            [v2.CameraSegmentationLabelComponent.from_dict(d) 
                for d in self.ungroup_row(self.frame_keys, key_values, r)])
        

    def addCamSegOrderedProtos(self):
        self.segmentation_protos_ordered = []
        for it, label_list in enumerate(self.cam_segmentation_list):
            segmentation_dict = {label.key.camera_name: label for label in label_list}
            self.segmentation_protos_ordered.append([segmentation_dict[name] for name in self.camera_left_to_right_order])

    def readLabel(self):
        # Decode a single panoptic label.
        panoptic_label_front = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
            self.segmentation_protos_ordered[0][open_dataset.CameraName.FRONT]
        )

        # Separate the panoptic label into semantic and instance labels.
        semantic_label_front, instance_label_front = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic_label_front,
            self.segmentation_protos_ordered[0][open_dataset.CameraName.FRONT].panoptic_label_divisor
        )

        # The dataset provides tracking for instances between cameras and over time.
        # By setting remap_to_global=True, this function will remap the instance IDs in
        # each image so that instances for the same object will have the same ID between
        # different cameras and over time.
        segmentation_protos_flat = sum(self.segmentation_protos_ordered, [])
        panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
            segmentation_protos_flat, remap_to_global=True
        )

        # We can further separate the semantic and instance labels from the panoptic
        # labels.
        NUM_CAMERA_FRAMES = 5
        self.semantic_labels_multiframe = []
        self.instance_labels_multiframe = []
        for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
            semantic_labels = []
            instance_labels = []
            for j in range(NUM_CAMERA_FRAMES):
                semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                    panoptic_labels[i + j], panoptic_label_divisor)
                semantic_labels.append(semantic_label)
                instance_labels.append(instance_label)
            self.semantic_labels_multiframe.append(semantic_labels)
            self.instance_labels_multiframe.append(instance_labels)

    def stackFrame(self):
        def _pad_to_common_shape(label):
            return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])

        # Pad labels to a common size so that they can be concatenated.
        instance_labels = [[_pad_to_common_shape(label) for label in instance_labels] for instance_labels in self.instance_labels_multiframe]
        semantic_labels = [[_pad_to_common_shape(label) for label in semantic_labels] for semantic_labels in self.semantic_labels_multiframe]
        instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]
        semantic_labels = [np.concatenate(label, axis=1) for label in semantic_labels]

        self.instance_label_concat = np.concatenate(instance_labels, axis=0)
        self.semantic_label_concat = np.concatenate(semantic_labels, axis=0)
        self.panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
            self.semantic_label_concat, self.instance_label_concat)
        
        # print(self.panoptic_label_rgb.shape)
        # _ = plt.hist(self.panoptic_label_rgb[:, :, 0], bins='auto')
        # plt.show()
        # plt.figure(figsize=(64, 60))
        # plt.imshow(self.panoptic_label_rgb)
        # plt.grid(False)
        # plt.axis('off')
        # plt.show()

if __name__ == '__main__':
    dir = '/media/kaiwenjon/Kevin-linux-dats/waymo/dataset/validation/camera_segmentation'
    parquetFile = f'{dir}/1024360143612057520_3580_000_3600_000.parquet' 
    camLabelReader = DataLabelReader(parquetName=parquetFile)
    camLabelReader.addCamSegList()
    camLabelReader.addCamSegOrderedProtos()
    camLabelReader.readLabel()
    camLabelReader.stackFrame()

    plt.figure(figsize=(64, 60))
    plt.imshow(camLabelReader.panoptic_label_rgb[:, :, 0])
    plt.grid(False)
    plt.axis('off')
    plt.show()



