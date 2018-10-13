import matplotlib.pyplot as plt
import numpy as np 
import glob
import cv2
def mask_by_range_and_plot(max_bgr_skin_range,min_bgr_skin_range,max_hsv_skin_range,min_hsv_skin_range):
    print("Max bgr skin range",max_bgr_skin_range)
    print("Min bgr skin range",min_bgr_skin_range)
    print("Max hsv skin range",max_hsv_skin_range)
    print("Min hsv skin range",min_hsv_skin_range)

    original_images=glob.glob("Images/Original Images/img_01*.jpg")+glob.glob("Images/Original Images/img_020.jpg")
    original_images.sort()


    for img in original_images:
        plt.figure(figsize=(20,9))

        bgr_img=cv2.imread(img)
        plt.subplot(141)
        bgr_mask=cv2.inRange(bgr_img,min_bgr_skin_range,max_bgr_skin_range)
        plt.axis('off')
        plt.title("BGR Mask",fontsize=15)
        plt.imshow(bgr_mask, cmap=plt.cm.gray)

        masked_bgr=cv2.bitwise_and(bgr_img,bgr_img,mask=bgr_mask)
        plt.subplot(142)
        plt.axis('off')
        plt.title("Masked BGR",fontsize=15)
        plt.imshow(cv2.cvtColor(masked_bgr,cv2.COLOR_BGR2RGB))
        
        hsv_img=cv2.cvtColor(masked_bgr,cv2.COLOR_BGR2HSV)
        hsv_mask=cv2.inRange(hsv_img,min_hsv_skin_range,max_hsv_skin_range)
        masked_hsv=cv2.bitwise_and(hsv_img,hsv_img,mask=hsv_mask)
        plt.subplot(143)
        plt.axis('off')
        plt.title("HSV Mask",fontsize=15)
        plt.imshow(hsv_mask, cmap=plt.cm.gray)
        
        plt.subplot(144)
        plt.axis('off')
        plt.title("Masked HSV",fontsize=15)
        plt.imshow(cv2.cvtColor(masked_hsv,cv2.COLOR_HSV2RGB))

        plt.show()

def mask_by_dilated_range_and_plot(max_bgr_skin_range,min_bgr_skin_range,max_hsv_skin_range,min_hsv_skin_range):
    print("Max bgr skin range",max_bgr_skin_range)
    print("Min bgr skin range",min_bgr_skin_range)
    print("Max hsv skin range",max_hsv_skin_range)
    print("Min hsv skin range",min_hsv_skin_range)
    kernel = np.ones((4,4),np.uint8)


    original_images=glob.glob("Images/Original Images/img_01*.jpg")+glob.glob("Images/Original Images/img_020.jpg")
    original_images.sort()


    for img in original_images:
        plt.figure(figsize=(20,9))

        bgr_img=cv2.imread(img)
        plt.subplot(141)
        bgr_mask=cv2.inRange(bgr_img,min_bgr_skin_range,max_bgr_skin_range)
        bgr_mask = cv2.dilate(bgr_mask,kernel,iterations = 5)
        plt.axis('off')
        plt.title("BGR Mask",fontsize=15)
        plt.imshow(bgr_mask, cmap=plt.cm.gray)

        masked_bgr=cv2.bitwise_and(bgr_img,bgr_img,mask=bgr_mask)
        plt.subplot(142)
        plt.axis('off')
        plt.title("Masked BGR",fontsize=15)
        plt.imshow(cv2.cvtColor(masked_bgr,cv2.COLOR_BGR2RGB))
        
        hsv_img=cv2.cvtColor(masked_bgr,cv2.COLOR_BGR2HSV)
        hsv_mask=cv2.inRange(hsv_img,min_hsv_skin_range,max_hsv_skin_range)
        hsv_mask = cv2.dilate(bgr_mask,kernel,iterations = 1)
        
        masked_hsv=cv2.bitwise_and(hsv_img,hsv_img,mask=hsv_mask)
        plt.subplot(143)
        plt.axis('off')
        plt.title("HSV Mask",fontsize=15)
        plt.imshow(hsv_mask, cmap=plt.cm.gray)
        
        plt.subplot(144)
        plt.axis('off')
        plt.title("Masked HSV",fontsize=15)
        plt.imshow(cv2.cvtColor(masked_hsv,cv2.COLOR_HSV2RGB))

        plt.show()
def kMeans(X, K, maxIters = 2):

    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Ensure we have K clusters, otherwise reset centroids and start over
        # If there are fewer than K clusters, outcome will be nan.
        if (len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C #clusters, clustered image