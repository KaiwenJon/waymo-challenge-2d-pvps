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

camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT, #4
                            open_dataset.CameraName.FRONT_LEFT, #2
                            open_dataset.CameraName.FRONT, #1
                            open_dataset.CameraName.FRONT_RIGHT, #3
                            open_dataset.CameraName.SIDE_RIGHT] #5

camera_name_list = {
   1 : "FRONT",
   2 : "FRONT_LEFT",
   3 : "FRONT_RIGHT",
   4 : "SIDE_LEFT",
   5 : "SIDE_RIGHT"
}
camera_name_num = 1

# Path to the directory with all components
dataset_dir = '/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_v2/validation'

context_name = '17065833287841703_2980_000_3000_000'

def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)


camera_image_df = read('camera_image')
# Filter the images from camera=1
# NOTE: We could also use push down filters while reading the parquet files as well
# Details https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html#dask.dataframe.read_parquet
camera_image_df = camera_image_df[camera_image_df['key.camera_name'] == camera_name_num]

camera_box_df = read('camera_box')
# Inner join the camera_image table with the camera_box table.
df = camera_image_df.merge(
    camera_box_df,
    on=[
        'key.segment_context_name',
        'key.frame_timestamp_micros',
        'key.camera_name',
    ],
    how='inner',
)

# Create corresponding components from the raw
_, row = next(iter(df.iterrows()))

camera_image = v2.CameraImageComponent.from_dict(row)
camera_box = v2.CameraBoxComponent.from_dict(row)
print(
    f'Loaded image ({len(camera_image.image)} bytes) for'
    f' {camera_image.key.camera_name=} {camera_image.key.frame_timestamp_micros} {camera_image.key.camera_name=}'
)
print(
    'Loaded bounding box for'
    f' {camera_box.key.camera_object_id=} {camera_box.box=}'
)


plt.imshow(tf.image.decode_jpeg(camera_image.image))
plt.title(camera_name_list[camera_name_num])
plt.grid(False)
plt.axis('off')
plt.show()

