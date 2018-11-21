import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def computeH(points_left, points_right):
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
    return L.reshape(3, 3)


def center_and_normalize_points(points):
    centroid = np.mean(points, axis=0)
    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])
    norm_factor = math.sqrt(2) / rms
    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    new_pointsh = (matrix @ pointsh).T
    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]

    return matrix, new_points


def compute_H_with_normalization(points_left, points_right):
    t1, new_points_left = center_and_normalize_points(points_left)
    t2, new_points_right = center_and_normalize_points(points_right)
    H = computeH(np.array(new_points_left), np.array(new_points_right))
    return np.dot(np.dot(np.linalg.inv(t2), H), t1)


def warp(image, H):
    im = image
    # This part will will calculate the X and Y offsets
    bunchX = []
    bunchY = []

    tt = np.array([[1], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[im.shape[1]], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[1], [im.shape[0]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[im.shape[1]], [im.shape[0]], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    refX1 = int(np.min(bunchX))
    refX2 = int(np.max(bunchX))
    refY1 = int(np.min(bunchY))
    refY2 = int(np.max(bunchY))
    # print("Refx1: ", refX1)
    # print("Refx2: ", refX2)
    # print("Refy1: ", refY1)
    # print("Refy2: ", refY2)
    # Final image whose size is defined by the offsets previously calculated
    final = np.zeros((int(refY2 - refY1), int(refX2 - refX1), 3))


    # Iterate over the imagine to transform every pixel
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):

            tt = np.array([[j], [i], [1]])
            tmp = np.dot(H, tt)
            x1 = int(tmp[0] / tmp[2]) - refX1
            y1 = int(tmp[1] / tmp[2]) - refY1

            if x1 > 0 and y1 > 0 and y1 < refY2 - refY1 and x1 < refX2 - refX1:
                final[y1, x1, :] = im[i, j, :]

    return final.astype(np.uint8), refX1
    # Hi = np.linalg.inv(H)
    # for i in range(final.shape[0]):
    #         for j in range(final.shape[1]):
    #                 if sum(final[i,j,:])==0:
    #                         tt = np.array([[j+refX1],[i+refY1],[1]])
    #                         tmp = np.dot(Hi,tt)
    #                         x1=int(tmp[0]/tmp[2])
    #                         y1=int(tmp[1]/tmp[2])

    #                         if x1>0 and y1>0 and x1<im.shape[1] and y1<im.shape[0]:
    #                                 final[i,j,:] = im[y1,x1,:]


def get_common_points(left_img_path, right_img_path, number_of_points):
    plt.figure(figsize=(20, 20))
    image_left = cv2.imread(left_img_path)
    plt.imshow(image_left)
    points_left = np.array(plt.ginput(n=number_of_points, timeout=0))
    # points_left = np.array([[[198.8267492, 789.50695507],
    #                          [272.87927895, 749.44575045],
    #                          [543.5959041, 614.69442583],
    #                          [626.14626513, 577.061173],
    #                          [720.83638513, 543.06984787],
    #                          [730.54819231, 394.96478838],
    #                          [845.87590258, 214.08237965],
    #                          [900.50481797, 505.43659505],
    #                          [928.42626361, 552.78165505],
    #                          [1064.39156413, 562.49346223],
    #                          [1117.80650362, 589.20093198],
    #                          [1213.71059953, 634.11804019]]])
    draw_points(image_left, points_left, "middle-right1.jpg")
    image_right = cv2.imread(right_img_path)
    plt.figure(figsize=(20, 20))
    plt.imshow(image_right)
    points_right = np.array(plt.ginput(n=number_of_points, timeout=0))
    # points_right = np.array([[193.78257543, 789.03505709]
    #                             , [280.41497626, 753.87987994]
    #                             , [550.35651509, 610.74808726]
    #                             , [624.43349551, 578.10399419]
    #                             , [724.87685879, 545.45990113]
    #                             , [812.76480166, 499.00484561]
    #                             , [729.89902695, 394.7948562]
    #                             , [841.64226861, 217.76342841]])
    print("Selected points for left img:", points_left)
    print("Selected points for right img:", points_right)
    return (points_left, points_right), (image_left, image_right)


def drawStitch(imageA, imageB, dif, offsetX):
    # initialize the output visualization image
    hdif = int(dif[0])
    wdif = int(dif[1])
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    print(offsetX)
    print(imageB.shape)
    print(wA - offsetX)
    print(wA + wB - offsetX)
    print(2022 - 3302)
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[hdif:hB + hdif, wA + offsetX:wA + wB + offsetX] = imageB

    return vis


def combine(image_left, image_mid, image_right):
    hL = image_left.shape[0]
    hC = image_mid.shape[0]
    hR = image_right.shape[0]

    wL = image_left.shape[1]
    wC = image_mid.shape[1]
    wR = image_right.shape[1]

    result = np.zeros((max(hL, hC, hR), wL + wC + wR, 3), dtype=np.uint8)
    result[:hL, :wL] = image_left
    # result[hL - hC:hL, wL:wL + wC] = image_mid
    result[:hC, wL:wL + wC] = image_mid
    result[:hR, wL + wC:wL + wC + wR] = image_right
    return result


def draw_points(img, points, fileName=None):
    tmp = np.copy(img)
    for (y, x) in points:
        tmp = cv2.circle(tmp, (int(y), int(x)), 10, (0, 0, 255), -1)
    if fileName is not None:
        cv2.imwrite(fileName, tmp)
    return tmp


def draw_common_points_on_images(img1, points1, img2, points2, filename):
    drawed1 = draw_points(img1, points1)
    drawed2 = draw_points(img2, points2)
    cv2.imwrite(filename + ".jpg", np.concatenate((drawed1, drawed2), axis=1))

if __name__ == '__main__':

    # (points_left_2_middle, points_middle_left_2), (image_left_2, image_middle) = get_common_points(
    #     "cmpe-building/left-1.jpg", "cmpe-building/middle.jpg", 12)
    # H_left_2_middle = compute_H_with_normalization(points_left_2_middle, points_middle_left_2)
    # warped_left, offsetX = warp(image_left_2, H_left_2_middle)
    # cv2.imshow("yeter", warped_left)

    # draw_common_points_on_images(image_left_2, points_left_2_middle, image_middle, points_middle_left_2, "left-2_middle")
    #
    #
    (points_middle_right_2, points_right_2_middle), (image_middle, image_right_2) = get_common_points(
        "cmpe-building/middle.jpg", "cmpe-building/right-1.jpg", 12)
    H_middle_2_right = compute_H_with_normalization(points_middle_right_2, points_right_2_middle)
    warped_right, offsetX = warp(image_right_2, np.linalg.inv(H_middle_2_right))

    # cv2.imshow("yeter", combine(warped_left, image_middle, warped_right))
    # cv2.waitKey()
    # cv2.imwrite("yeter.jpg", combine(warped_left, image_middle, warped_right))
    cv2.imshow("yeter", warped_right)
