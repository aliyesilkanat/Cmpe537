import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

def get_min_max_ranges(arr1,arr2,arr3):
    return [np.min(arr1[arr1>0]),np.min(arr2[arr2>0]),np.min(arr3[arr3>0])],[np.max(arr1[arr1>0]),np.max(arr2[arr2>0]),np.max(arr3[arr3>0])]
def get_binary_mask(bgr_img,param=None):
    ret, mask = cv2.threshold(bgr_img, 1, 255, cv2.THRESH_BINARY)
    if param=="morph":

        kernel = np.ones((4,4),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)

    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def seg_skin(param=None):
    min_ranges_bgr=[]
    max_ranges_bgr=[]
    min_ranges_hsv=[]
    max_ranges_hsv=[]
    for img in glob.glob("Images/Ground Truths/*.jpg"):

        bgr_img=cv2.imread(img)
        binary_mask=get_binary_mask(bgr_img,param)
        bgr_img=cv2.bitwise_and(bgr_img,bgr_img,mask=binary_mask)
        b,g,r=cv2.split(bgr_img)

        min_range,max_range=get_min_max_ranges(b,g,r)
        min_ranges_bgr.append(min_range)
        max_ranges_bgr.append(max_range)

        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv_img)
        min_range,max_range=get_min_max_ranges(h,s,v)
        min_ranges_hsv.append(min_range)
        max_ranges_hsv.append(max_range)


    min_ranges_bgr=np.array(min_ranges_bgr)
    max_ranges_bgr=np.array(max_ranges_bgr)
    min_ranges_hsv=np.array(min_ranges_hsv)
    max_ranges_hsv=np.array(max_ranges_hsv)


    max_bgr_skin_range=np.min(max_ranges_bgr,axis=0)
    min_bgr_skin_range=np.max(min_ranges_bgr,axis=0)
    max_hsv_skin_range=np.min(max_ranges_hsv,axis=0)
    min_hsv_skin_range=np.max(min_ranges_hsv,axis=0)

    print("Max bgr skin range",max_bgr_skin_range)
    print("Min bgr skin range",min_bgr_skin_range)
    print("Max hsv skin range",max_hsv_skin_range)
    print("Min hsv skin range",min_hsv_skin_range)

    #prediction on original images
    original_images=glob.glob("Images/Original Images/img_00*.jpg")+glob.glob("Images/Original Images/img_010.jpg")
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
        hsv_mask=cv2.inRange(bgr_img,min_hsv_skin_range,max_hsv_skin_range)
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

seg_skin()
seg_skin("morph")
