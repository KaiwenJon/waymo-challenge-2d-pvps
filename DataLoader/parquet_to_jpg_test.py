
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
tf.compat.v1.enable_eager_execution()
import os

class PanopticSegmentationDataset(Dataset):
    def __init__(self, frames_path, labels_path, frames_id_list_path, datasetNum=None, desired_context_name=None, desired_timestamp=None, transform=None):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.frames_context_list = self.get_frames_context_list(frames_id_list_path, datasetNum, desired_context_name, desired_timestamp)
        self.transform = transform
        self.camera_name_list = {
            1 : "FRONT",
            2 : "FRONT_LEFT",
            3 : "FRONT_RIGHT",
            4 : "SIDE_LEFT",
            5 : "SIDE_RIGHT"
        }
        print("Data prepared to be imported: ", len(self.frames_context_list), " frames.")
        for line in self.frames_context_list:
            print("context_name: ", line[0], "Timestamp: ", line[1])
        
        self.frames_list = [] # a list, each item is a list[(img1, "FRONT"), ..., (img5, "SIDE_RIGHT")] at one time step
        self.labels_list = [] # a list, each item is a list[[img1_label_semantic, img1_label_instance, img1_panoptic_label, "FRONT"], ..., (img5_label_semantic, img5_label_instance, "SIDE_RIGHT")] at one time step
        # end  = 0
        # start = 0
        # for line in self.frames_context_list:
        #     context_name, time_stamp = line
        #     start = time.time()
        #     self.load_frames_from_parquet(context_name, time_stamp)
        #     self.load_labels_from_parquet(context_name, time_stamp)
        #     end = time.time()
        #     print(round(end-start, 2), "s, imported ", context_name, time_stamp)
            
        # print("Imported Done!")
        # print("Num of frames, ", len(self.frames_list))
        # assert(len(self.frames_list) == len(self.labels_list)) # the number of timestamps should match
        
        
    def __len__(self):
        return len(self.frames_context_list)
    
    def __getitem__(self, index):
        """
        get a frame (containing multiple cameras) for one time step
        """
        context_name, time_stamp = self.frames_context_list[index]
        self.load_frames_from_parquet(context_name, time_stamp)
#         self.load_labels_from_parquet(context_name, time_stamp)
    
        frames = self.frames_list[-1]# a list of tuple:(rgb input images, camera_num): [(img1, FRONT), (img2, FRONT_LEFT),...,(img5, SIDE_RIGHT)]
        # labels = self.labels_list[-1]# a list of 3-tems:(semantic_labels, instance_labels, camera_num):
        
        self.frames_list.pop(0)
        # self.labels_list.pop(0)
        # apply transform
        if self.transform:
            frames = [(self.transform(img), camera_name) for (img, camera_name) in frames]  
#             labels = [(self.transform(semantic_label), self.transform(instance_label), self.transform(panoptic_label), camera_name) for (semantic_label, instance_label, panoptic_label, camera_name) in labels]
        
        return frames
    
    def load_frames_from_parquet(self, context_name, desired_time_stamp):
        """
        Given context name and time_stamp, fetch parquet file, filter with desired timestamp
        
        In theory, in the filtered df, for each unique timestamp, there's 5 images corresponding to 5 cameras
        
        for each unique timestamp, append a list[(img1, "FRONT"), ..., (img5, "SIDE_RIGHT")] to self.frames_list
        
        """
        
        camera_image_df = dd.read_parquet(tf.io.gfile.glob(f'{self.frames_path}/{context_name}.parquet'))
        filtered_camera_image_df = camera_image_df[camera_image_df['key.frame_timestamp_micros'] == (desired_time_stamp)]
        frames_for_one_timestamp = []
        collected_cameras = [] # to determine if we need to move on and wrap next timestamp
        for i, (key_values, row) in enumerate(filtered_camera_image_df.iterrows()):
            # In theory, there's five iterations since there's 5 cameras for each desired timestamp
            camera_image = v2.CameraImageComponent.from_dict(row)
            data_time_stamp = camera_image.key.frame_timestamp_micros
            camera_name = self.camera_name_list[camera_image.key.camera_name]
            image = tf.image.decode_jpeg(camera_image.image)
            image = image.numpy()
            image = np.int32(image)
            collected_cameras.append(camera_name)
            frames_for_one_timestamp.append((image, camera_image.key.camera_name))
        # The end, append all frames in this timestamp to global frames_list
        frames_for_one_timestamp = sorted(frames_for_one_timestamp, key=lambda frames_for_one_timestamp: frames_for_one_timestamp[1])
        frames_for_one_timestamp = [(image, self.camera_name_list[camera_num]) for (image, camera_num) in frames_for_one_timestamp]
        self.frames_list.append(frames_for_one_timestamp)

    
    def get_frames_context_list(self, txtFile, datasetNum, desired_context_name, desired_timestamp):
        """
        input: txt file provided by Waymo
        
        output: a list = [
            [context_name1, timestamp1],
            [context_name1, timestamp2],
            [context_name1, timestamp3],
            ...,
            [context_name2, timestamp1],
            ...
        ]
        """
        list_ = open(txtFile).read().split()
        list_ = sorted(list_, key=lambda list_: list_[1])
        frames_context_list = []
        cutoff = datasetNum
        if(datasetNum == None):
            cutoff = len(list_)
        else:
            cutoff = datasetNum
            
        numFetchedData = 0
        for line in list_:
            if(numFetchedData >= cutoff):
                break
            frames_line = line.split(',')
            context_name, time_stamp = frames_line
            
            # if a specific context of timestamp is requested, only fetch the desired parts.
            if(desired_context_name != None and context_name != desired_context_name):
                continue
            if(desired_timestamp!= None and time_stamp != desired_timestamp):
                continue
                
            # To this point, data is allowed to be fetched.
            frames_context_list.append([context_name, int(time_stamp)])
            numFetchedData += 1
        return frames_context_list
    



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
    ## only at test: 
    # "10980133015080705026_780_000_800_000"

]
start_context = "11096867396355523348_1460_000_1480_000" # only append data starting from this context
txtFile = "./2d_pvps_test_frames.txt"
context_name_list, frames_dict = getFramesList(txtFile, start_context, context_with_issue)
for key in frames_dict:
    print(key)

my_transform = transforms.Compose([
    transforms.ToTensor()
])
frames_id_list_path = txtFile
frames_path = "/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_v2/testing_location/camera_image"
labels_path = None
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
    for batch_idx, frames in enumerate(dataloader):
        time_stamp = frames_dict[context_name][batch_idx]
        
        img_FRONT = frames[0][0][0]
        img_FRONT_LEFT = frames[1][0][0]
        img_FRONT_RIGHT = frames[2][0][0]
        img_SIDE_LEFT = frames[3][0][0]
        img_SIDE_RIGHT = frames[4][0][0]

        os.makedirs(f"{dataset_folder}/{context_name}/{time_stamp}", exist_ok=True)
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT.jpg", cv2.cvtColor(np.float32(img_FRONT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT_LEFT.jpg", cv2.cvtColor(np.float32(img_FRONT_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_FRONT_RIGHT.jpg", cv2.cvtColor(np.float32(img_FRONT_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_SIDE_LEFT.jpg", cv2.cvtColor(np.float32(img_SIDE_LEFT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dataset_folder}/{context_name}/{time_stamp}/img_SIDE_RIGHT.jpg", cv2.cvtColor(np.float32(img_SIDE_RIGHT.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
       