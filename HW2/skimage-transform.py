import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform

# plt.figure(figsize=(100, 100))
image_left = cv2.cvtColor(cv2.imread("cmpe-building/left-2.jpg"), cv2.COLOR_BGR2RGB)
# plt.imshow(image_left)
# points_left = np.array(plt.ginput(5))
# # plt.axis('off')
image_right = cv2.cvtColor(cv2.imread("cmpe-building/left-1.jpg"), cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(100, 100))
#
# plt.imshow(image_right)
# points_right = np.array(plt.ginput(5))
# # plt.axis('off')

points_left = np.array([[464.47258591, 792.24735438]
                           , [432.76472104, 746.58802896]
                           , [718.13550488, 612.1466819]
                           , [959.11527791, 555.07252514]
                           , [969.26179467, 402.87477375]])
points_right = np.array([[205.73640856, 811.2720733]
                            , [174.02854368, 766.88106248]
                            , [491.1071924, 609.61005271]
                            , [723.20876326, 543.65769378]
                            , [730.81865083, 397.80151537]])
print(points_left)
print(points_right)
p1 = points_left
p2 = points_right
A = []
for i in range(0, len(p1)):
    x, y = p1[i][0], p1[i][1]
    u, v = p2[i][0], p2[i][1]
    A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
A = np.asarray(A)
U, S, Vh = np.linalg.svd(A)
L = Vh[-1, :] / Vh[-1, -1]
H = L.reshape(3, 3)

cv2.imshow("warp", transform.warp(image_right, H))
