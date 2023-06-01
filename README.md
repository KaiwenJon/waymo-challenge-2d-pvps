# Waymo Open Dataset Challenge - 2D Video Panoptic Segmentation
This project aims to participate in the Waymo Open Dataset Challenge, specifically in the 2D Video Panoptic Segmentation Task.

## Goal
Given a panoramic sequence of camera images across five cameras and over time, produce a set of panoptic segmentation labels for each pixel, where the instance labels are consistent across all images in the sequence.
![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/926dc90c-98f9-4830-8fc1-4fea856a5891)

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/47cd84e2-a7b7-42a0-92c0-45e6e375a093)

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/ae9e6986-5a02-4b87-a0d7-4d6c2fbd465f)

## Proposed Approach
The task of panoptic segmentation aims to predict the semantic and instance segmentation
of a single image. However, in our task, that’s not enough. Even if we have the ability to have
an accurate result of panoptic segmentation in every single image, the instance IDs across five
cameras and timestamps wouldn’t be consistent.

We try to overcome this challenge by two phases:
1. Apply modern ML models to predict Video Panoptic Segmentation, which solves the
instance IDs problem across time.
2. For every timestamp, use image stitching technique to determine the overlapped
instance between adjacent camera images and reassign IDs to keep them consistent.

The figure below shows our pipeline:
![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/33e8ec64-ec52-4c7b-aed2-509c60be0b49)

### Phase 1: Video Panoptic Segmentation
We take advantage of the modern model ViP-DeepLab, which extends the state-of-the-art
panoptic segmentation model, Panoptic-DeepLab, to the video domain. To briefly review ViPDeepLab, during the training, the model takes a pair of image frames as input and their
panoptic segmentation ground truths as training target. During inference, ViP-DeepLab
performs two-frame image panoptic predictions at each timestamp and continues the inference
process for every two consecutive frames (with one overlapping frame at the next time step) in
a video sequence. The consecutive two predictions are then overlapped to propagate instance
IDs so that all time-steps will keep consistent. The last frame will be predicted with a copy of
itself since it’s there’s no more frame left in the sequence.
![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/e53754a1-12a6-4330-a2f1-3b22026ddb5b)


### Phase 2: Image Stitching and re-assign IDs across cameras
At the stage of Video Panoptic Segmentation, we see video from different cameras as
different sequence of images, meaning that even though the sequence of “FRONT” camera
across time is consistent, it has no knowledge of the instance IDs of the sequence of
“FRONT_LEFT”. Therefore, we need to perform some kind of instance matching and reassign the IDs.
To associate information from multiple camera views, we perform image stitching first
for the adjacent cameras, like the combination below:
- stitch(SIDE_LEFT, FRONT_LEFT)
- stitch(FRONT_LEFT, FRONT)
- stitch(FRONT, FRONT_RIGHT)
- stitch(FRONT_RIGHT, SIDE_RIGHT)

We could do SIFT feature matching between images to find the homography. However, since
we’re already given extrinsic parameters of cameras, we can directly estimate the homography between
two images, assuming that the two cameras are just related by a pure rotation.
![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/c634a060-bfb0-4101-9a94-a557e1f078d0)

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/1ba8c76b-36a8-4622-b22a-4d80c578f69c)

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/8dd5a91d-c7f7-4eca-bd1e-1868f521a99d)

There’s some error involved, which can be caused by the fact that two cameras might have some
translations between them.
After they are stitched, we then look at the overlapped area. Within the area, we find all the pixels
where two adjacent images have the same semantic label. For each of those pixels, we obtain the
instance that encloses the pixel. Having found this instance for the two images, we then compute the
IoU of these two instances. If they happen to overlap a lot, we consider them the same object in the
scene and re-assign the IDs.
The picture below shows the mask where pixels in img1 and img2 share the same semantic label
within the overlapped area. (yellow=True).
![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/d98a1333-7937-4dc0-a7e5-0d0b119f4b2b)

Among this mask, we pick the instance of the cyclist as an example. We first find the cyclist
instance in left and right images. Then we compute the IoU in the stitched image within the overlapped
area to determine if they are the same objects in the scene.

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/72d249be-7120-446e-9b14-fef1c10cf248)

It turns out that the IoU between these two instances in the overlapped area is 0.9995. And thus we
consider them the same object and re-assign the ID in the right image to be the same as that of the left.

## ViP-DeepLab Inference Results
We inferenced the validation set using ViP-DeepLab’s provided checkpoint as our
experiment. It was trained on Cityscapes-DVPS dataset using ResNet50 backbone. The
following figures are some random chosen examples from our validation dataset.

![image](https://github.com/KaiwenJon/waymo-challenge-2d-pvps/assets/70893513/999cc97b-7321-46ff-9bec-b7a7f50df317)

## Discussion and conclusions
The inference results are not as good as expected, we thought there are two possible
reasons. First, the checkpoint wasn’t trained based on Waymo Open Dataset, and there are
some discrepancies between two datasets, which could cause the bad performance.
Furthermore, our proposed approach’s phase two wouldn’t work well if the phase one results
are not good. From the prediction results above, we can see that a lot of semantic class are not
fully segmented. Hence, when we were trying to reassign the ID between consecutive frames,
there were a high possibility the two frames weren’t close to each other enough to do the
comparison.

In order to improve our result, we should keep on training the Waymo Open Dataset on
ViP-DeepLab2 pretrained model and make suer the semantic classes are good enough for us to
finish the phase 2 ID assignment. It’s a special opportunity to participate in and explore this
challenge during the whole semester, both of us learned a lot of know-how from the paers and
also from practically implementing the whole system.

## Reference
1. Waymo Open Dataset Challenge: https://waymo.com/intl/en_us/open/challenges/2023/2dvideo-panoptic-segmentation/
2. Waymo 2D Video Panoptic Segmentaion Dataset:
https://waymo.com/intl/en_us/open/data/perception/#2d-video-panoptic-segmentation
3. ViP-DeepLab:
https://github.com/google-research/deeplab2/blob/main/g3doc/projects/vip_deeplab.md
4. Waymo Open Dataset: Panoramic Video Panoptic Segmentation:
https://arxiv.org/pdf/2206.07704.pdf
5. MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers:
https://arxiv.org/pdf/2012.00759v3.pdf
6. kMaX-DeepLab:
https://github.com/google-research/deeplab2/blob/main/g3doc/projects/kmax_deeplab.md
7. Panoptic-DeepLab:
https://github.com/googleresearch/deeplab2/blob/main/g3doc/projects/panoptic_deeplab.md
8. Masked-attention Mask Transformer for Universal Image Segmentation:
https://arxiv.org/pdf/2112.01527v3.pdf
9. Focal Modulation Network:
https://arxiv.org/pdf/2203.11926v3.pdf

