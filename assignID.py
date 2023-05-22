import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def _pad_to_common_shape(img):
    if(len(img.shape) == 3):
        return np.pad(img, [[1280 - img.shape[0], 0], [0, 0], [0, 0]])
    elif(len(img.shape) == 2):
        return np.pad(img, [[1280 - img.shape[0], 0], [0, 0]])
    

context_name = "9243656068381062947_1297_428_1317_428"
pred_path = f'../{context_name}'
image_path = f'/media/kaiwenjon/Kevin-linux-dats/waymo/dataset_jpg/validation/{context_name}'

cam_extrinsic = np.load('./Parameters/Camera_Param/cam_extrinsic.npy')# shape (5, 4, 4)
cam_intrinsic = np.load('./Parameters/Camera_Param/cam_intrinsic.npy')# shape (3, 4, 4)
cam_distortion = np.load('./Parameters/Camera_Param/cam_distortion.npy') # k1, k2, p1, p2, k3

cam_dict = {
    "img_FRONT" : 0,
    "img_FRONT_LEFT" : 1,
    "img_FRONT_RIGHT" : 2,
    "img_SIDE_LEFT" : 3,
    "img_SIDE_RIGHT" : 4
}
n1 = "FRONT"
n2 = "FRONT_RIGHT"

cam_name1 = f"img_{n1}"
cam_name2 = f"img_{n2}"

def warpImage(img1, img2, H, addImg1=False):
    img_stitch = cv2.warpPerspective(img2, H, (img2.shape[1]*2, img2.shape[0]))
    if(addImg1):
        img_stitch[0:img1.shape[0], 0:img1.shape[1]] = img1
    return img_stitch

def computeIoU(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))

time_stamps = [f for f in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, f))]
time_stamps = sorted(time_stamps)
for i, time_stamp in enumerate(time_stamps):
    if(i == 5):
        pred1 = cv2.imread(f"{pred_path}/{time_stamp}/{cam_name1}_panoptic_prediction.png")
        img1 = cv2.imread(f"{image_path}/{time_stamp}/{cam_name1}.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        semantic_pred1 = pred1[:, :, 0]
        instance_pred1 = pred1[:, :, 1]
        panoptic_pred1 = semantic_pred1 * 1000 + instance_pred1
        img1 = _pad_to_common_shape(img1)
        semantic_pred1 = _pad_to_common_shape(semantic_pred1)
        
        pred2 = cv2.imread(f"{pred_path}/{time_stamp}/{cam_name2}_panoptic_prediction.png")
        img2 = cv2.imread(f"{image_path}/{time_stamp}/{cam_name2}.jpg")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        semantic_pred2 = pred2[:, :, 0]
        instance_pred2 = pred2[:, :, 1]
        panoptic_pred2 = semantic_pred2 * 1000 + instance_pred2
        img2 = _pad_to_common_shape(img2)
        semantic_pred2 = _pad_to_common_shape(semantic_pred2)
        
        H = np.load(f"./Parameters/Homography/H_{n1}_and_{n2}.npy")
        semantic_warp = warpImage(semantic_pred1, semantic_pred2, H, addImg1=False)
        panoptic_warp = warpImage(panoptic_pred1, panoptic_pred2, H, addImg1=False)
        img_stitch = warpImage(img1, img2, H, addImg1=True)
        img_warp = warpImage(img1, img2, H, addImg1=False)
        
        reassignIDList = []
        
        tl = H @ np.array([0, 0, 1])
        tl /= tl[2]
        bl = H @ np.array([0, img2.shape[0], 1])
        bl /= bl[2]
        left_most_of_warpedImg2 = np.floor(min(tl[0], bl[0])).astype(int)
        right_most_of_img1 = img1.shape[1]
        overlap_mask = np.zeros_like(semantic_warp)
        overlap_mask[:, left_most_of_warpedImg2:right_most_of_img1] = 1
        
        
        sem_overlap1 = np.zeros_like(semantic_warp)
        sem_overlap1[:, left_most_of_warpedImg2:right_most_of_img1] = semantic_pred1[:, left_most_of_warpedImg2:right_most_of_img1]
        sem_overlap2 = np.zeros_like(semantic_warp)
        sem_overlap2[:, left_most_of_warpedImg2:right_most_of_img1] = semantic_warp[:, left_most_of_warpedImg2:right_most_of_img1]
        
        
        panoptic_pred1_long = np.zeros_like(panoptic_warp)
        panoptic_pred1_long[:img1.shape[0], :img1.shape[1]] = panoptic_pred1
        panoptic_pred2_long = panoptic_warp
        
        plt.imshow(semantic_pred1)
        plt.show()
        plt.imshow(semantic_pred2)
        plt.show()
        overlap_same_semantic_mask = sem_overlap1 == sem_overlap2
        overlap_same_semantic_mask[overlap_mask == 0] = 0
        panoptic1_unique_ids, panoptic1_unique_ids_cnts = np.unique(panoptic_pred1_long[overlap_same_semantic_mask], return_counts=True)
        for i, panoptic1_id in enumerate(panoptic1_unique_ids):
            area_in_panoptic1 = panoptic1_unique_ids_cnts[i]
            print("=================Found ",panoptic1_id, " in the left semantic cnts:", area_in_panoptic1)
            if(i != -1):
                overlap_same_sem1_certain_pano_id_mask =  np.logical_and(overlap_same_semantic_mask, (panoptic_pred1_long == panoptic1_id))
                m = (panoptic_pred1_long == panoptic1_id)
                # for this certain panoptic1_id, we want to find every panoptic2_ids that appears within this area, for each
                # of them, compute the IoU. If IoU is big, it means panotic1_id is the same object as that specific panoptic2_id,
                # So we reassign the instance2_id[pantopic_pred == panoptic2] = instance1_id
                panoptic2_ids, panoptic2_ids_cnts = np.unique(panoptic_pred2_long[overlap_same_sem1_certain_pano_id_mask], return_counts=True)
                print("These panoptic_id2 is the candidate to compute IoU")
                print(panoptic2_ids)
                max_iou = 0
                best_panoptic2 = None
                for panoptic2_id in panoptic2_ids:
                    overlap_same_sem2_certain_pano_id_mask = np.logical_and(overlap_same_semantic_mask, (panoptic_pred2_long == panoptic2_id))
#                     print(overlap_same_sem2_certain_pano_id_mask.sum())
#                     plt.imshow(overlap_same_sem2_certain_pano_id_mask)
#                     plt.show()
                    iou = computeIoU(overlap_same_sem1_certain_pano_id_mask, overlap_same_sem2_certain_pano_id_mask)
                    print("-------candidate", panoptic2_id, "IoU: ",iou)
                    if(iou > max_iou):
                        max_iou = iou
                        best_panoptic2 = panoptic2_id
                if(max_iou > 0.7 and area_in_panoptic1 > 100):
                    # replace!!
                    print(panoptic1_id, " and ", best_panoptic2, " are the same objects!!!!!!!!!!")
                    reassignIDList.append([panoptic1_id, best_panoptic2])
        
        
        print("Our reassignIDList!")
        print(reassignIDList)
        fig, axs = plt.subplots(2*len(reassignIDList)+2+2+1, 1, figsize=(10, 30))
        
        axs[0].imshow(img_stitch)
        axs[0].set_title('img_stitch')
        
        axs[1].imshow(img_warp)
        axs[1].set_title('img_warp')
        
        axs[2].imshow(pred1)
        axs[2].set_title('pred1')
        
        axs[3].imshow(pred2)
        axs[3].set_title('pred2')
        
        axs[4].imshow(overlap_same_semantic_mask)
        axs[4].set_title("overlap_same_semantic_mask")
        
        for i, [panoptic1_id, panoptic2_id] in enumerate(reassignIDList):
            instance_pred2[panoptic_pred2 == panoptic2_id] = instance_pred1[panoptic_pred1 == panoptic1_id][0]
            img1_mask = np.zeros_like(img1)
            img2_mask = np.zeros_like(img2)
            for j in range(3):
                img1_mask[:, :, j] = panoptic_pred1 == panoptic1_id
                img2_mask[:, :, j] = panoptic_pred2 == panoptic2_id
                
            axs[2*i+5].imshow(img1 * img1_mask)
            axs[2*i+5].set_title(panoptic1_id)
            axs[2*i+6].imshow(img2 * img2_mask)
            axs[2*i+6].set_title(panoptic2_id)
        


plt.show()