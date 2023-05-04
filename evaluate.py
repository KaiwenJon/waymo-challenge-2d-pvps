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

#@title Metric computation utility functions

def _run_dummy_inference_from_protos(
    image_proto_list: Sequence[bytes]
) -> List[np.ndarray]:
  """Creates dummy predictions from protos."""
  panoptic_preds = []
  for image_proto in image_proto_list:
    image_array = tf.image.decode_jpeg(image_proto).numpy()
    # Creates a dummy prediction by setting the panoptic labels to 0 for all pixels.
    panoptic_pred = np.zeros(
        (image_array.shape[0], image_array.shape[1], 1), dtype=np.int32)
    panoptic_preds.append(panoptic_pred)
  return panoptic_preds


def _compute_metric_for_dataset(filename: str):
  """Computes metric for the dataset frames."""
  eval_config = camera_segmentation_metrics.get_eval_config()
  new_panoptic_label_divisor = eval_config.panoptic_label_divisor

  dataset = tf.data.TFRecordDataset(filename, compression_type='')
  # Load first 3 frames in the demo.
  frames_with_seg, sequence_id = (
      camera_segmentation_utils.load_frames_with_labels_from_dataset(dataset, 3))

  segmentation_protos_ordered = []
  image_protos_ordered = []
  # Only aggregates frames with camera segmentation labels.
  for frame in frames_with_seg:
    segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
    segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_left_to_right_order])
    image_proto_dict = {image.name: image.image for image in frame.images}
    image_protos_ordered.append([image_proto_dict[name] for name in camera_left_to_right_order])

  # The dataset provides tracking for instances between cameras and over time.
  # By setting remap_to_global=True, this function will remap the instance IDs in
  # each image so that instances for the same object will have the same ID between
  # different cameras and over time.
  segmentation_protos_flat = sum(segmentation_protos_ordered, [])
  image_protos_flat = sum(image_protos_ordered, [])
  decoded_elements = (
      camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
          segmentation_protos_flat, remap_to_global=True,
          new_panoptic_label_divisor=new_panoptic_label_divisor)
  )
  panoptic_labels, num_cameras_covered, is_tracked_masks = decoded_elements[0:3]
  
  # We provide a dummy inference function in the demo. Please replace this with 
  # your own method. It is recommended to generate your own panoptic labels first
  # and implement a function to load the generated panoptic labels from the disk.
  panoptic_preds = _run_dummy_inference_from_protos(image_protos_flat)
  return camera_segmentation_metrics.get_metric_object_by_sequence(
    true_panoptic_labels=panoptic_labels,
    pred_panoptic_labels=panoptic_preds,
    num_cameras_covered=num_cameras_covered,
    is_tracked_masks=is_tracked_masks,
    sequence_id=sequence_id,
  )


multi_sequence_result = camera_segmentation_metrics.aggregate_metrics(
    [_compute_metric_for_dataset(filename) for filename in eval_filenames]
)
print('Metrics:')
print(multi_sequence_result)
