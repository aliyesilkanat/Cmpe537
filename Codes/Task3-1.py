import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from utils import mask_by_range_and_plot,kMeans


def get_max_min_centroid_values(img):
    reshaped_img =img.reshape(img.shape[0] * img.shape[1],img.shape[2])
    cluster_centers,quantized_img_predicted=kMeans(reshaped_img,4)
    quantized_img = [ cluster_centers[cl] for cl in quantized_img_predicted ]
    quantized_img=np.array(quantized_img).reshape(img.shape[0] , img.shape[1],img.shape[2])

    values,counts=np.unique(quantized_img_predicted, return_counts=True,axis=0)
    centroid_having_max_instances=np.argmax(counts)

    max_centroid=reshaped_img[centroid_having_max_instances==quantized_img_predicted]
    return np.max(max_centroid,axis=0),np.min(max_centroid,axis=0)


min_ranges_bgr=[]
max_ranges_bgr=[]
min_ranges_hsv=[]
max_ranges_hsv=[]
original_images=glob.glob("Images/Original Images/img_01*.jpg")+glob.glob("Images/Original Images/img_020.jpg")
original_images.sort()
for img_path in original_images:
    bgr_img=cv2.imread(img_path)
    max_rgb,min_rgb=get_max_min_centroid_values(bgr_img)
    max_ranges_bgr.append(max_rgb)
    min_ranges_bgr.append(min_rgb)
    hsv_img=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
    max_hsv,min_hsv=get_max_min_centroid_values(hsv_img)
    max_ranges_hsv.append(max_hsv)
    min_ranges_hsv.append(min_hsv)

max_bgr_skin_range=np.max(max_ranges_bgr,axis=0).astype(np.uint8)
min_bgr_skin_range=np.min(min_ranges_bgr,axis=0).astype(np.uint8)
max_hsv_skin_range=np.max(max_ranges_hsv,axis=0).astype(np.uint8)
min_hsv_skin_range=np.min(min_ranges_hsv,axis=0).astype(np.uint8)

mask_by_range_and_plot(max_bgr_skin_range,min_bgr_skin_range,max_hsv_skin_range,min_hsv_skin_range)
