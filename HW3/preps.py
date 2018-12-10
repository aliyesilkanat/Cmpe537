from PIL import Image
import numpy as np
import cv2
img=Image.open("Images/kuzey.jpg")
qualities=[0,20,40,60,80,100] #https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html?highlight=quality 1-95 range, default 75
for i in range(len(qualities)):
    img.save("Images/jpeg_quality/"+str(i+1)+".jpg",quality=qualities[i])


def noise_generator (image,mean,var):

    sigma = var**0.5
    noised_img=(img+np.random.normal(loc=mean,scale=sigma,size=(img.size[1],img.size[0],3)))
    return  Image.fromarray(np.clip(noised_img,0,256).astype(np.uint8))
vars=[0,1,2,3,4,5]
for i in range(len(vars)):
    noise_generator(img,0,(i+1)*1000).save("Images/gaussian_noise/"+str(i+1)+".jpg")

