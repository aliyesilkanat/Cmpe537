import sys
import numpy as np
import cv2
import os

np.random.seed(1)
sys.path.append("../")
from hw2 import computeH, combine, compute_H_with_normalization, warp, draw_common_points_on_images


def get_common_points(left_img_path, right_img_path, number_of_points, warp, number_of_wrong_matches=None):
    # plt.figure(figsize=(20, 20))
    image_left = cv2.imread(left_img_path)

    # plt.imshow(image_left)
    # points_left = np.array(plt.ginput(n=number_of_points, timeout=0))
    if warp == "left":
        points_left = np.array([[284.37191344, 749.12465815],
                                [545.64471998, 614.68331109],
                                [624.28022486, 579.17050244],
                                [416.2766313, 353.41050455],
                                [459.39932753, 353.41050455],
                                [510.13191132, 355.94713374],
                                [732.08696543, 397.80151537],
                                [841.16202058, 218.9691575],
                                [929.94404222, 547.46263757],
                                [1063.11707469, 560.14578351],
                                [1117.65460226, 591.85364839],
                                [1212.77819688, 631.17140083]])[:number_of_points]
    else:
        points_left = np.array([[388.20698921, 617.12237762],
                                [473.18530204, 580.7031007],
                                [575.15927743, 545.49779967],
                                [580.01518102, 396.17876427],
                                [576.37325333, 340.33587299],
                                [689.2730118, 221.36623503],
                                [770.60939693, 546.71177557],
                                [938.13807079, 528.5021371],
                                [945.42192618, 579.4891248],
                                [1030.400239, 620.76430531],
                                [1203.99879235, 779.79514789],
                                [1270.76746671, 828.35418379]])[:number_of_points]
    # draw_points(image_left, points_left, "middle-right1.jpg")
    image_right = cv2.imread(right_img_path)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image_right)
    # points_right = np.array(plt.ginput(n=number_of_points, timeout=0))
    if warp == "left":
        points_right = np.array(
            [[91.99687022, 766.44141302],
             [388.20698921, 618.33635352],
             [473.18530204, 580.7031007],
             [252.24168869, 345.19177658],
             [302.01470049, 347.61972837],
             [353.00168819, 347.61972837],
             [580.01518102, 396.17876427],
             [686.84506, 215.29635554],
             [774.25132462, 543.06984787],
             [894.43493848, 553.99563095],
             [946.63590207, 581.91707659],
             [1032.8281908, 621.97828121]])[:number_of_points]
    else:
        points_right = np.array(
            [[156.33759279, 623.19225711],
             [254.66964049, 579.4891248],
             [368.78337485, 543.06984787],
             [373.63927844, 388.89490889],
             [371.21132665, 334.2659935],
             [486.53903691, 214.08237965],
             [569.08939794, 545.49779967],
             [729.33421642, 526.07418531],
             [731.76216821, 578.2751489],
             [808.24264976, 613.48044993],
             [952.70578156, 755.51562994],
             [1019.47445592, 799.21876225]])[:number_of_points]
    if number_of_wrong_matches is not None:
        points_left = make_wrong_selection(image_left, number_of_wrong_matches, points_left)
        points_right = make_wrong_selection(image_right, number_of_wrong_matches, points_right)
    print("Selected points for left img:", points_left)
    print("Selected points for right img:", points_right)
    return (points_left, points_right), (image_left, image_right)


def make_wrong_selection(image, number_of_wrong_matches, points):
    print("Number of wrong matches: ", number_of_wrong_matches)
    rand_ixs = np.random.choice(points.shape[0], number_of_wrong_matches, replace=False)

    # np.random.randint(0, points.shape[0], number_of_wrong_matches)
    print("Changed point indexes: ", rand_ixs)
    y_s = np.random.randint(0, image.shape[0], len(rand_ixs))
    x_s = np.random.randint(0, image.shape[1], len(rand_ixs))
    points[rand_ixs] = np.array([y_s, x_s]).T
    return points


def warp_and_combine(number_of_points, wrong_points=None, normalization=False):
    final_img_path = str(number_of_points) + "points/"
    if normalization:
        final_img_path += "norm/"

    (points_left_1_middle, points_middle_left_1), (image_left_1, image_middle) = get_common_points(
        "../cmpe-building/left-1.jpg", "../cmpe-building/middle.jpg", number_of_points, "left", wrong_points)

    if normalization:
        H_left_1_middle = compute_H_with_normalization(points_left_1_middle, points_middle_left_1)
    else:
        H_left_1_middle = computeH(points_left_1_middle, points_middle_left_1)

    warped_left, offsetX = warp(image_left_1, H_left_1_middle)
    draw_common_points_on_images(image_left_1, points_left_1_middle, image_middle, points_middle_left_1,
                                 final_img_path + "left-1_middle" + str(wrong_points) + "wrong")

    (points_middle_right_1, points_right_1_middle), (image_middle, image_right_1) = get_common_points(
        "../cmpe-building/middle.jpg", "../cmpe-building/right-1.jpg", number_of_points, "right", wrong_points)

    if normalization:
        H_middle_1_right = compute_H_with_normalization(points_middle_right_1, points_right_1_middle)
    else:
        H_middle_1_right = computeH(points_middle_right_1, points_right_1_middle)
    warped_right, offsetX = warp(image_right_1, np.linalg.inv(H_middle_1_right))
    draw_common_points_on_images(image_middle, points_middle_right_1, image_right_1, points_right_1_middle,
                                 final_img_path + "middle_left-1" + str(wrong_points) + "wrong")

    # os.chdir(final_img_path)

    cv2.imwrite(final_img_path+"final" + str(wrong_points) + "wrong.jpg", combine(warped_left, image_middle, warped_right))


warp_and_combine(5, 3, normalization=True)
warp_and_combine(5, 3, normalization=False)
warp_and_combine(12, 3, normalization=True)
warp_and_combine(12, 3, normalization=False)

warp_and_combine(12, None, normalization=True)
warp_and_combine(5, None, normalization=False)
warp_and_combine(12, None, normalization=False)
warp_and_combine(5, None, normalization=True)

warp_and_combine(5, 5, normalization=True)
warp_and_combine(5, 5, normalization=False)
warp_and_combine(12, 5, normalization=True)
warp_and_combine(12, 5, normalization=False)
