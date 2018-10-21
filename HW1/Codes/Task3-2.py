import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from utils import mask_by_range_and_plot,kMeans



def get_max_crowded_centroid_value(img):
    reshaped_img =img.reshape(img.shape[0] * img.shape[1],img.shape[2])
    cluster_centers,quantized_img_predicted=kMeans(reshaped_img,4)
    quantized_img = [ cluster_centers[cl] for cl in quantized_img_predicted ]
    quantized_img=np.array(quantized_img).reshape(img.shape[0] , img.shape[1],img.shape[2])

    values,counts=np.unique(quantized_img_predicted, return_counts=True,axis=0)
    return cluster_centers[values[np.argmax(counts)]]


bgr_centroids=[]
hsv_centroids=[]
original_images=glob.glob("Images/Original Images/img_01*.jpg")+glob.glob("Images/Original Images/img_020.jpg")
original_images.sort()
for img_path in original_images:
    bgr_img=cv2.imread(img_path)


    bgr_centroids.append(get_max_crowded_centroid_value(bgr_img))
    hsv_img=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
    hsv_centroids.append(get_max_crowded_centroid_value(bgr_img))


max_bgr_skin_range=np.max(bgr_centroids,axis=0).astype(np.uint8)
min_bgr_skin_range=np.min(bgr_centroids,axis=0).astype(np.uint8)
max_hsv_skin_range=np.max(hsv_centroids,axis=0).astype(np.uint8)
min_hsv_skin_range=np.min(hsv_centroids,axis=0).astype(np.uint8)

mask_by_range_and_plot(max_bgr_skin_range,min_bgr_skin_range,max_hsv_skin_range,min_hsv_skin_range)
