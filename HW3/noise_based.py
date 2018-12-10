import cv2
import numpy as np
PATH = "plots/noise_based"

np.seterr(invalid='ignore')
import matplotlib.pyplot as plt
import glob
from utils import *

file_names = glob.glob(r"Images/gaussian_noise/*.jpg")
harris_results = []
sift_results = []
manual_harris_results = []

for f, idx in zip(file_names, range(len(file_names))):
    fig = plt.figure(figsize=(25, 5))
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv_Harris = calc_harris(gray)
    cv_Sift = calc_sift(gray)

    manual_harris = myHarrisDetector(gray)
    #     plotharrispoints(gray,manual_harris)
    img = img[..., ::-1]
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.plot([p[1] for p in manual_harris], [p[0] for p in manual_harris], '*')
    plt.title("Implemented Harris")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.plot([p[1] for p in cv_Harris], [p[0] for p in cv_Harris], '*')
    plt.title("Harris")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.plot([p[1] for p in cv_Sift], [p[0] for p in cv_Sift], '*')
    plt.title("SIFT")
    plt.axis('off')
    fig.suptitle("Image " + str(idx + 1))
    plt.savefig(PATH+"/keypoints"+str(idx+1)+".jpg")

    plt.show()

    harris_results.append(cv_Harris)
    sift_results.append(cv_Sift)
    manual_harris_results.append(manual_harris)
sift_repeatability = []
harris_repeatability = []
manual_repeatability = []
for i in range(len(file_names))[1:]:
    sift_repeatability.append(calc_repeatability(sift_results[0], sift_results[i]))
    harris_repeatability.append(calc_repeatability(harris_results[0], harris_results[i]))
    manual_repeatability.append(calc_repeatability(manual_harris_results[0], manual_harris_results[i]))
plot_repeatability(file_names, sift_repeatability, harris_repeatability, manual_repeatability,
                   sift_results, harris_results, manual_harris_results,PATH)
