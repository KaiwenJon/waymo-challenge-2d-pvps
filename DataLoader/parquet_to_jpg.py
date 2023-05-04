
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import dask.dataframe as dd
import tensorflow as tf
from waymo_open_dataset import v2
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from waymo_open_dataset.utils import camera_segmentation_utils
import cv2
import os
tf.compat.v1.enable_eager_execution()

from pvps_dataset import PanopticSegmentationDataset


# I want to know what context_name is in this txt file

def getFramesList(txtFile, start_context, context_with_issue):
    context_name_list = []
    frames_dict = {}
    list_ = open(txtFile).read().split()
    start_fetching = False
    for line in list_:
        frames_line = line.split(',')
        context_name, time_stamp = frames_line
        if(context_name == start_context):
            start_fetching = True
        if(not start_fetching):
            continue
        if not(context_name in context_name_list) and not(context_name in context_with_issue):
            context_name_list.append(context_name)
        if not(context_name in frames_dict):
            frames_dict[context_name] = [time_stamp]
        else:
            frames_dict[context_name].append(time_stamp)

    for context_name in frames_dict:
        time_stamp_list = frames_dict[context_name]
        frames_dict[context_name] = sorted(time_stamp_list)
    #     print("======", context_name)
    #     print(len(frames_dict[context_name]))
    #     for time in frames_dict[context_name]:
    #         print(time)
    return context_name_list, frames_dict

context_with_issue = [
    # Training:
    # "10724020115992582208_7660_400_7680_400", # Don't append data from this context
    # "11139647661584646830_5470_000_5490_000",
    # "11379226583756500423_6230_810_6250_810",
    # "13181198025433053194_2620_770_2640_770",
    # "1422926405879888210_51_310_71_310",
    # "14276116893664145886_1785_080_1805_080",
    # "15367782110311024266_2103_310_2123_310",
    # "15832924468527961_1564_160_1584_160",
    # "2590213596097851051_460_000_480_000",
    # "2692887320656885771_2480_000_2500_000",
    # "3919438171935923501_280_000_300_000",
    # "5200186706748209867_80_000_100_000",
    # "5861181219697109969_1732_000_1752_000",
    # "6193696614129429757_2420_000_2440_000",
    # "6559997992780479765_1039_000_1059_000",
    # "6694593639447385226_1040_000_1060_000",
    # "7038362761309539946_4207_130_4227_130",
    # "809159138284604331_3355_840_3375_840",
    # "8806931859563747931_1160_000_1180_000"


    # Testing:
    # "10149575340910243572_2720_000_2740_000",
    # "10649066155322078676_1660_000_1680_000"
    # Can't find context name based on txt...
]
start_context = "10149575340910243572_2720_000_2740_000" # only append data starting from this context
txtFile = "./2d_pvps_test_frames.txt"
context_name_list, frames_dict = getFramesList(txtFile, start_context, context_with_issue)
for key in frames_dict:
    print(key)

my_transform = transforms.Compose([
    transforms.ToTensor()
])
frames_id_list_path = txtFile
frames_path = "/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_v2/testing/camera_image"
labels_path = "/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_v2/testing/camera_segmentation"
datasetNum = None
desired_timestamp = None
dataset_folder = "/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_jpg/testing"
# For each context name, read all the data and store it in a folder
for context_name in context_name_list:
    print("Fetching context:", context_name)
    context_folder = f"{dataset_folder}/{context_name}"
    desired_context_name = context_name
    dataset = PanopticSegmentationDataset(frames_path, labels_path, frames_id_list_path, datasetNum, desired_context_name, desired_timestamp, transform=my_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    for batch_idx, (frames, labels) in enumerate(dataloader):
        time_stamp = frames_dict[context_name][batch_idx]
        
        img_FRONT = frames[0][0][0]
        img_FRONT_LEFT = frames[1][0][0]
        img_FRONT_RIGHT = frames[2][0][0]
        img_SIDE_LEFT = frames[3][0][0]
        img_SIDE_RIGHT = frames[4][0][0]

        # torch.Size([batch_size, 1, 1280, 1920])
        semantic_labels_FRONT = labels[0][0][0]
        semantic_labels_FRONT_LEFT = labels[1][0][0]
        semantic_labels_FRONT_RIGHT = labels[2][0][0]
        semantic_labels_SIDE_LEFT = labels[3][0][0]
        semantic_labels_SIDE_RIGHT = labels[4][0][0]

        # torch.Size([batch_size, 1, 1280, 1920])
        instance_labels_FRONT = labels[0][1][0]
        instance_labels_FRONT_LEFT = labels[1][1][0]
        instance_labels_FRONT_RIGHT = labels[2][1][0]
        instance_labels_SIDE_LEFT = labels[3][1][0]
        instance_labels_SIDE_RIGHT = labels[4][1][0]

        # torch.Size([batch_size, 3, 1280, 1920])
        panoptic_rgb_labels_FRONT = labels[0][2][0]
        panoptic_rgb_labels_FRONT_LEFT = labels[1][2][0]
        panoptic_rgb_labels_FRONT_RIGHT = labels[2][2][0]
        panoptic_rgb_labels_SIDE_LEFT = labels[3][2][0]
        panoptic_rgb_labels_SIDE_RIGHT = labels[4][2][0]
        
        os.makedirs(f"{dataset_folder}/{context_name}/{time_stamp}", exist_ok=True)
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT.jpg", cv2.cvtColor(np.float32(img_FRONT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT_LEFT.jpg", cv2.cvtColor(np.float32(img_FRONT_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT_RIGHT.jpg", cv2.cvtColor(np.float32(img_FRONT_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_SIDE_LEFT.jpg", cv2.cvtColor(np.float32(img_SIDE_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_SIDE_RIGHT.jpg", cv2.cvtColor(np.float32(img_SIDE_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/semantic_labels_FRONT.jpg", cv2.cvtColor(np.float32(semantic_labels_FRONT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/semantic_labels_FRONT_LEFT.jpg", cv2.cvtColor(np.float32(semantic_labels_FRONT_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/semantic_labels_FRONT_RIGHT.jpg", cv2.cvtColor(np.float32(semantic_labels_FRONT_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/semantic_labels_SIDE_LEFT.jpg", cv2.cvtColor(np.float32(semantic_labels_SIDE_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/semantic_labels_SIDE_RIGHT.jpg", cv2.cvtColor(np.float32(semantic_labels_SIDE_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/instance_labels_FRONT.jpg", cv2.cvtColor(np.float32(instance_labels_FRONT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/instance_labels_FRONT_LEFT.jpg", cv2.cvtColor(np.float32(instance_labels_FRONT_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/instance_labels_FRONT_RIGHT.jpg", cv2.cvtColor(np.float32(instance_labels_FRONT_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/instance_labels_SIDE_LEFT.jpg", cv2.cvtColor(np.float32(instance_labels_SIDE_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/instance_labels_SIDE_RIGHT.jpg", cv2.cvtColor(np.float32(instance_labels_SIDE_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/panoptic_rgb_labels_FRONT.jpg", cv2.cvtColor(np.float32(panoptic_rgb_labels_FRONT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/panoptic_rgb_labels_FRONT_LEFT.jpg", cv2.cvtColor(np.float32(panoptic_rgb_labels_FRONT_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/panoptic_rgb_labels_FRONT_RIGHT.jpg", cv2.cvtColor(np.float32(panoptic_rgb_labels_FRONT_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/panoptic_rgb_labels_SIDE_LEFT.jpg", cv2.cvtColor(np.float32(panoptic_rgb_labels_SIDE_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/panoptic_rgb_labels_SIDE_RIGHT.jpg", cv2.cvtColor(np.float32(panoptic_rgb_labels_SIDE_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))