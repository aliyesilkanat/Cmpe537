import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_histogram(ch):
    hist=np.zeros(256,dtype=np.int32)
    for pix in ch.flatten():
        hist[pix]+=1
    return hist


bgr_img=cv2.imread("Images/Original Images/img_001.jpg")

b,g,r=cv2.split(bgr_img)
hsv_img=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv_img)
h=((h/180)*255).astype(np.uint8)

plt.figure(figsize=(20,9))
plt.subplot(131)
plt.title("B")
plt.axis('off')
plt.imshow(b)
plt.subplot(132)
plt.axis('off')
plt.title("G")
plt.imshow(g)
plt.subplot(133)
plt.title("R")
plt.axis('off')
plt.imshow(r)
plt.show()

plt.figure(figsize=(20,9))
plt.subplot(131)
plt.title("H")
plt.axis('off')
plt.imshow(h)
plt.subplot(132)
plt.axis('off')
plt.title("S")
plt.imshow(s)
plt.subplot(133)
plt.title("V")
plt.axis('off')
plt.imshow(v)
plt.show()


b_hist=get_histogram(b)
g_hist=get_histogram(g)
r_hist=get_histogram(r)
h_hist=get_histogram(h)
s_hist=get_histogram(s)
v_hist=get_histogram(v)
linspace=np.linspace(0,len(b_hist),num=len(b_hist))
plt.figure(figsize=(20,9))
plt.subplot(131)
plt.title("B")
plt.bar(linspace,b_hist)
plt.subplot(132)
plt.title("G")
plt.bar(linspace,g_hist)
plt.subplot(133)
plt.title("R")
plt.bar(linspace,r_hist)
plt.show()
plt.figure(figsize=(20,9))
plt.subplot(131)
plt.title("H")
plt.bar(linspace,h_hist)
plt.subplot(132)
plt.title("S")
plt.bar(linspace,s_hist)
plt.subplot(133)
plt.title("V")
plt.bar(linspace,v_hist)
plt.show()
