import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas

def image_path(file_name):
    image_ori_ = cv2.imread('./Dataset/OriginalData/testing/images/'+file_name)
    image_seg_ = cv2.imread('./ours/'+file_name)
    # image_seg_ = cv2.imread('./output_predictions/output_predictions/'+file_name)
    # image_seg_ = cv2.imread('./3.1分割结果/3.1分割结果/'+file_name)

    image_seg = cv2.cvtColor(image_seg_, cv2.COLOR_BGR2GRAY)
    image_ori = cv2.resize(image_ori_, image_seg.shape, interpolation=cv2.INTER_LINEAR)
    return image_ori

# def concave(image_ori):
#     image_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)

#     # 进行边缘检测
#     _, binary_image = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
#     # 边缘检测和轮廓提取
#     contours_all,_ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours_all
def concave(image_seg):
    image_gray = cv2.cvtColor(image_seg, cv2.COLOR_BGR2GRAY)

    # 进行边缘检测
    _, binary_image = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    # 边缘检测和轮廓提取
    contours_all,_ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours_all
# def compute_curvature(contour, index):
#     prev = contour[(index - 1) % len(contour)][0]
#     current = contour[index][0]
#     next = contour[(index + 1) % len(contour)][0]

#     # 计算三个点的向量
#     v1 = np.array(current) - np.array(prev)
#     v2 = np.array(next) - np.array(current)

#     # 计算曲率
#     cross_product = np.cross(v1, v2)
#     area = 0.5 * np.linalg.norm(cross_product)
#     side1 = np.linalg.norm(v1)
#     side2 = np.linalg.norm(v2)
#     radius = (side1 * side2) / (2.0 * area)
#     curvature = 1.0 / radius

#     return curvature
import numpy as np

import numpy as np

def compute_curvature(contour, index):
    pt1 = contour[index - 1][0]  # Previous point
    pt2 = contour[index][0]      # Current point
    pt3 = contour[(index + 1) % len(contour)][0]  # Next point
    
    v1 = pt2 - pt1
    v2 = pt3 - pt2
    
    # Check if v1 and v2 are 2D arrays
    if v1.ndim < 2:
        v1 = np.expand_dims(v1, axis=0)
    if v2.ndim < 2:
        v2 = np.expand_dims(v2, axis=0)
    
    # Compute cross product
    cross_product = np.cross(v1, v2, axisa=0, axisb=0, axisc=0)
    
    # Compute norms
    norm_v1 = np.linalg.norm(v1, axis=-1)
    norm_v2 = np.linalg.norm(v2, axis=-1)
    
    # Compute curvature
    curvature = 2 * np.linalg.norm(cross_product, axis=-1) / (norm_v1 * norm_v2 * (norm_v1 + norm_v2))
    
    return curvature

def compute_contour_length(contour, index1, index2):
    # 计算两个点之间的轮廓线长度
    length = 0
    if index1 < index2:
        for i in range(index1, index2):
            length += cv2.arcLength(contour[i:i+2], False)
    else:
        for i in range(index1, index2, -1):
            length += cv2.arcLength(contour[i-1:i+1], False)
    return length
def find_concave_points_with_curvature(contour, k, max_distance_threshold):
    if len(contour) == 0:
        return []
    
    concave_points = []  # 存储所有的凹点及其索引
    contour_length = len(contour)  # 轮廓的长度，即边界像素的数量

    for i in range(contour_length):
        pass_indicate = True  # 表示凹点是否通过测试
        for j in range(1, k):
            left_index = (i - j) % contour_length
            right_index = (i + j) % contour_length
            left_point = contour[left_index]
            right_point = contour[right_index]
            middle_x = int((left_point[0] + right_point[0]) / 2.)
            middle_y = int((left_point[1] + right_point[1]) / 2.)
            if cv2.pointPolygonTest(contour, (middle_x, middle_y), False) > 0:
                pass_indicate = False
                break
        if pass_indicate:
            concave_points.append((contour[i], i))  # 包括轮廓索引
    
    concave_points.sort(key=lambda x: x[1])

    filtered_concave_points = concave_points.copy()

    while True:
        i = 0  # 开始的索引
        filtered_concave_points = []
        while i < len(concave_points):
            point1, index1 = concave_points[i]
            point2, index2 = concave_points[(i + 1) % len(concave_points)]  # 下一个凹点
            length_between_points = compute_contour_length(contour, index1, index2)
            if length_between_points <= max_distance_threshold:
                curvature1 = compute_curvature(contour, index1)
                curvature2 = compute_curvature(contour, index2)

                # 保留曲率更大的凹点
                if curvature1 >= curvature2:
                    filtered_concave_points.append((point1, index1))
                else:
                    filtered_concave_points.append((point2, index2))
                i += 2  # 跳过下一个凹点
            else:
                filtered_concave_points.append((point1, index1))
                i += 1  # 继续下一个凹点

        if len(concave_points) == len(filtered_concave_points):
            break  # 如果没有凹点在距离范围内，退出循环
        concave_points = filtered_concave_points.copy()

    # 按照轮廓方向对筛选后的凹点进行排序
    filtered_concave_points.sort(key=lambda x: x[1])

    return filtered_concave_points
# def find_concave_points_with_curvature(contour, k, max_distance_threshold):
#     concave_points = []  # 存储所有的凹点及其索引
#     contour_length = len(contour)  # 轮廓的长度，即边界像素的数量

#     for i in range(contour_length):
#         pass_indicate = True  # 表示凹点是否通过测试
#         for j in range(1, k):
#             left_index = (i - j) % contour_length
#             right_index = (i + j) % contour_length
#             left_point = contour[left_index][0]
#             right_point = contour[right_index][0]
#             middle_x = int((left_point[0] + right_point[0]) / 2.)
#             middle_y = int((left_point[1] + right_point[1]) / 2.)
#             if cv2.pointPolygonTest(contour, (middle_x, middle_y), False) > 0:
#                 pass_indicate = False
#                 break
#         if pass_indicate:
#             concave_points.append((contour[i][0], i))  # 包括轮廓索引
#     concave_points.sort(key=lambda x: x[1])

#     filtered_concave_points = concave_points.copy()

#     while True:
#         i = 0  # 开始的索引
#         filtered_concave_points = []
#         while i < len(concave_points):
#             point1, index1 = concave_points[i]
#             point2, index2 = concave_points[(i + 1) % len(concave_points)]  # 下一个凹点
#             # 检查两个连续凹点之间的距离是否小于等于 max_distance_threshold
#             # distance_between_points = np.linalg.norm(np.array(point1) - np.array(point2))
#             length_between_points = compute_contour_length(contour, index1, index2)
#             if length_between_points <= max_distance_threshold:
#                 curvature1 = compute_curvature(contour, index1)
#                 curvature2 = compute_curvature(contour, index2)

#                 # 保留曲率更大的凹点
#                 if curvature1 >= curvature2:
#                     filtered_concave_points.append((point1, index1))
#                 else:
#                     filtered_concave_points.append((point2, index2))
#                 i += 2  # 跳过下一个凹点
#             else:
#                 filtered_concave_points.append((point1, index1))
#                 i += 1  # 继续下一个凹点

#         if len(concave_points) == len(filtered_concave_points):
#             break  # 如果没有凹点在距离范围内，退出循环
#         concave_points = filtered_concave_points.copy()

#     # 按照轮廓方向对筛选后的凹点进行排序
#     filtered_concave_points.sort(key=lambda x: x[1])

#     # # 首尾相接
#     # if len(filtered_concave_points) > 0:
#     #     filtered_concave_points.append(filtered_concave_points[0])

#     return filtered_concave_points

def is_ellipse_inside_contour(ellipse, contour,result_image):
    ellipse_points = np.array(cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                               (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                               int(ellipse[2]), 0, 360, 5))
    ellipse_contour_area = cv2.contourArea(ellipse_points)  # Use contourArea directly on ellipse contour
    if ellipse_contour_area > 0:

        ellipse_mask = np.zeros_like(result_image)
        cv2.fillPoly(ellipse_mask, [ellipse_points], 255)
        contour_mask = np.zeros_like(result_image)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        ellipse_mask_gray = cv2.cvtColor(ellipse_mask, cv2.COLOR_BGR2GRAY)
        contour_mask_gray = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)
        intersection_mask = cv2.bitwise_and(ellipse_mask_gray, contour_mask_gray)
        intersection_area = cv2.countNonZero(intersection_mask)
        overlap_percentage = intersection_area / ellipse_contour_area
    # Adjust the threshold for overlap percentage
        return overlap_percentage >= 3/4   # You can experiment with different threshold values
    return False  # If ellipse_contour_area <= 0, consider it as not overlapping

def sxbian(ellipse):
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    aspect_ratio = major_axis / minor_axis
    return aspect_ratio <=3

def combine_flattened_ellipses(ellipses):
    com_ellipses = []

    # Iterate over pairs of ellipses
    for i in range(len(ellipses)):
        for j in range(i + 1, len(ellipses)):
            ellipse1 = ellipses[i]
            ellipse2 = ellipses[j]

            # Get the points used for fitting each ellipse
            points_ellipse1 = cv2.ellipse2Poly((int(ellipse1[0][0]), int(ellipse1[0][1])),
                                                (int(ellipse1[1][0] / 2), int(ellipse1[1][1] / 2)),
                                                int(ellipse1[2]), 0, 360, 5)
            points_ellipse2 = cv2.ellipse2Poly((int(ellipse2[0][0]), int(ellipse2[0][1])),
                                                (int(ellipse2[1][0] / 2), int(ellipse2[1][1] / 2)),
                                                int(ellipse2[2]), 0, 360, 5)

            # Combine the points used for fitting from both ellipses
            combined_points = np.concatenate([points_ellipse1, points_ellipse2])

            # Fit an ellipse to the combined points
            combined_ellipse = cv2.fitEllipse(combined_points)

            com_ellipses.append(combined_ellipse)

    return com_ellipses

def generate_points_around_contour(contour, num_points=20, distance_threshold=5):
    points_around_contour = []

    for i in range(len(contour)):
        point1 = contour[i][0]
        next_index = (i + 1) % len(contour)
        point2 = contour[next_index][0]

        # 生成沿轮廓线的点
        t = np.linspace(0, 1, num_points)
        x = (1 - t) * point1[0] + t * point2[0]
        y = (1 - t) * point1[1] + t * point2[1]

        points_around_contour.extend(np.column_stack((x, y)))

    return np.array(points_around_contour, dtype=np.int32)
# def combine_flattened_ellipses(ellipses, contour):
#     com_ellipses = []

#     # Iterate over pairs of ellipses
#     for i in range(len(ellipses)):
#         for j in range(i + 1, len(ellipses)):
#             ellipse1 = ellipses[i]
#             ellipse2 = ellipses[j]
#             com_ellipse = combine_ellipses(ellipse1, ellipse2)
#             if com_ellipse is not None:
#                 com_ellipses.append(com_ellipse)

#     return com_ellipses

def draw(contours_all, file_name, image_ori):
    result_image = np.copy(image_ori)
    mask = np.zeros_like(result_image)
    contours = []
    contours1 = []
    concave_points= []
    ellipses_list = []
    ellipses_list1 = []
    bian_ellipse = []
    num_ellipses = 0
    num_contours = 0
    threshold_area = 50  # 设置轮廓面积的阈值，用于筛选掉太小的轮廓
    threshold_ellipse_area = 1000  # 设置椭圆面积的阈值，用于筛选掉太小的椭圆
    
    for contour in contours_all[0]:
        contour_area = cv2.contourArea(contour)
        contours.append(contour)
        # if contour_area > threshold_area:
        #     contours.append(contour)
    
    for contour in contours:
        # cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)  # 绘制轮廓线为红色线条
        concave_points = find_concave_points_with_curvature(contour, k=10, max_distance_threshold=25)
        for coor in concave_points:
            cv2.circle(result_image, (int(coor[0][0]), int(coor[0][1])), 1, (0, 255, 0), 2)  
        if len(concave_points) <= 1:  # 确保至少有两个凹点才进行处理
            contours1.append(contour)
            cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)  # 绘制轮廓线为红色线条
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1) # 创建椭圆掩码
            num_contours = len(contours1)
                # 计算轮廓的中心点
#——————————————————————————————————————————————————-重心拟合
        if len(concave_points) >= 4:
            M = cv2.moments(contour)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])

            # 计算每个凹点与中心点的距离
            distances = [math.sqrt((center_x - point[0][0]) ** 2 + (center_y - point[0][1]) ** 2) for point in concave_points]
            sorted_points = sorted(zip(concave_points, distances), key=lambda x: x[1])
            # selected_points_list = [point[0] for point in sorted_points]
            selected_points_list = [point[0] for point in sorted_points[:4]]
            # 将坐标提取为NumPy数组
            selected_points_array = np.array([point[0] for point in selected_points_list], dtype=np.float32)

            # 进行椭圆拟合
            if len(selected_points_array) == 4:  # 至少需要5个点来拟合椭圆
                ellipse1 = cv2.minAreaRect(np.array(selected_points_array))
                center, (major_axis, minor_axis), angle = ellipse1
                ellipses_list.append(ellipse1)

        if len(concave_points) >= 2:
            for i in range(len(concave_points)):
                point1 = concave_points[i][1]
                next_index = (i + 1) % len(concave_points)
                point2 = concave_points[next_index][1]

                # 生成沿轮廓线的点
                if point1 < point2:
                    contour_around_points = generate_points_around_contour(contour[point1:point2])
                else:
                    contour_around_points = generate_points_around_contour(np.concatenate((contour[point1:], contour[:point2]), axis=0))
                # if len(contour_around_points) >= 5:
                #     ellipse2 = cv2.fitEllipse(contour_around_points)
                #     ellipses_list.append(ellipse2)

                retries = 0
                max_retries = 10  # Set a maximum number of retries to avoid infinite loop
                while retries < max_retries:
                    if len(contour_around_points) >= 6:
                        ellipse2 = cv2.fitEllipse(contour_around_points)
                        if is_ellipse_inside_contour(ellipse2, contour,result_image) and sxbian(ellipse2):
                            # Check if the ellipse center is inside the contour
                            ellipse_center = (int(ellipse2[0][0]), int(ellipse2[0][1]))
                            if cv2.pointPolygonTest(contour, ellipse_center, False) != -1:
                                ellipses_list.append(ellipse2)
                                break
                            else:
                                # Move the center of the ellipse to a point inside the contour
                                centroid = np.mean(contour_around_points, axis=0).astype(int)
                                direction = centroid - ellipse_center
                                contour_around_points += direction
                                retries += 1
                        else:
                            break
 #——————————————————————————————————————限制椭圆中点在轮廓内     
        # ellipses_list = []

        # if len(concave_points) >= 2:
        #     for i in range(len(concave_points)):
        #         point1 = concave_points[i][1]
        #         next_index = (i + 1) % len(concave_points)
        #         point2 = concave_points[next_index][1]

        #         if point1 < point2:
        #             segment = contour[point1:point2]
        #         else:
        #             segment = np.concatenate((contour[point1:], contour[:point2]), axis=0)

        #         retries = 0
        #         max_retries = 10  # Set a maximum number of retries to avoid infinite loop
        #         while retries < max_retries:
        #             if len(segment) >= 6:
        #                 ellipse2 = cv2.fitEllipse(segment)
                        
        #                 # Check if the ellipse center is inside the contour
        #                 ellipse_center = (int(ellipse2[0][0]), int(ellipse2[0][1]))
        #                 if cv2.pointPolygonTest(contour, ellipse_center, False) != -1:
        #                     ellipses_list.append(ellipse2)
        #                     break
        #                 else:
        #                     # Move the center of the ellipse to a point inside the contour
        #                     centroid = np.mean(segment, axis=0).astype(int)
        #                     direction = centroid - ellipse_center
        #                     segment += direction
        #                     retries += 1
        #             else:
        #                 break

#—————————————————————————————————————————————————————限制椭圆在轮廓内并且椭圆中心在轮廓内 
        # if len(concave_points) >= 2:  # 确保至少有两个凹点才进行处理
        #     for i in range(len(concave_points) - 1):
        #         # 选取当前凹点
        #         start_point = concave_points[i][1]

        #         # 寻找下一个凹点，确保两个凹点之间没有其他凹点
        #         next_index = (i + 1) % len(concave_points)
        #         end_point = concave_points[next_index][1]

        #         # 获取凹点之间的轮廓线上的点
        #         intermediate_points = []
        #         between_points = False
        #         for point in contour:
        #             point = tuple(point[0])

        #             if point == start_point:
        #                 between_points = True

        #             if between_points:
        #                 intermediate_points.append(point)

        #             if point == end_point:
        #                 between_points = False
        #                 intermediate_points.append(point)

        #         # 进行椭圆拟合
        #         if len(intermediate_points) >= 5:
        #             # 拟合轮廓线上的椭圆，使用最小二乘法
        #             intermediate_points = np.array(intermediate_points, dtype=np.float32)
        #             ellipse2 = cv2.fitEllipseDirect(intermediate_points)
        #             ellipses_list.append(ellipse2)


        # if len(concave_points) >= 2:
        #     for i in range(len(concave_points)):
        #         point1 = concave_points[i][1]
        #         next_index = (i + 1) % len(concave_points)
        #         point2 = concave_points[next_index][1]

        #         if point1 < point2:
        #             segment = generate_points_around_contour(contour[point1:point2])
        #         else:
        #             segment = generate_points_around_contour(np.concatenate((contour[point1:], contour[:point2]), axis=0))
        #         # if len(contour_around_points) >= 5:
        #         #     ellipse2 = cv2.fitEllipse(contour_around_points)
        #         #     ellipses_list.append(ellipse2)

        #         retries = 0
        #         max_retries = 10  # 设置最大重试次数，以避免无限循环
        #         while retries < max_retries:
        #             if len(segment) >= 6:  # 至少需要6个点来拟合椭圆
        #                 ellipse2 = cv2.fitEllipse(segment)

        #                 # Check if the ellipse center is inside the contour
        #                 ellipse_center = (int(ellipse2[0][0]), int(ellipse2[0][1]))

        #                 # Check if the ellipse boundary is inside the contour
        #                 # ellipse_points = generate_points_around_contour(ellipse2, num_points=len(segment))
                        
        #                 if cv2.pointPolygonTest(contour, ellipse_center, False) != -1:
        #                     # ellipses_list.append(ellipse2)
        #                     break
        #                 else:
        #                     # Move the center of the ellipse to a point inside the contour
        #                     centroid = np.mean(segment, axis=0).astype(int)
        #                     direction = centroid - ellipse_center
        #                     segment += direction
        #                     retries += 1
        #             else:
        #                 break

        #             if retries == max_retries:
        #                 # If the maximum number of retries is reached, break the loop
        #                 break
                    #__________________-两个扁椭圆拟合成一个
        # bian_ellipse = [ellipse for ellipse in ellipses_list if not sxbian(ellipse)]
        # # Combine ellipses and draw
        # combined_ellipses = combine_flattened_ellipses(bian_ellipse)
        # for combined_ellipse in combined_ellipses:
        #     if is_ellipse_inside_contour(combined_ellipse, contour,result_image) and sxbian(combined_ellipse) and (cv2.pointPolygonTest(contour, (int(combined_ellipse[0][0]), int(combined_ellipse[0][1])), False) > 0):
        #         cv2.ellipse(result_image, combined_ellipse, (255, 255, 0), 2)
        #         cv2.ellipse(mask, combined_ellipse, (255, 255, 255), -1)  # 创建椭圆掩码
#_____________________________
        ellipses_list1.clear()  # 清空 ellipses_list1
        for ellipse in ellipses_list:
            # if sxbian(ellipse):
            if is_ellipse_inside_contour(ellipse, contour,result_image)and sxbian(ellipse): 
            # if is_ellipse_inside_contour(ellipse, contour) and (cv2.pointPolygonTest(contour, (int(ellipse[0][0]), int(ellipse[0][1])), False) > 0):
            # if is_ellipse_inside_contour(ellipse, contour) and sxbian(ellipse) and (cv2.pointPolygonTest(contour, (int(ellipse[0][0]), int(ellipse[0][1])), False) > 0):
                ellipses_list1.append(ellipse)
                num_ellipses += 1  # 每次符合条件时计数
                # num_ellipses = len(ellipses_list1)
                cv2.ellipse(result_image, ellipse, (255, 0, 0), 2)
                cv2.ellipse(mask, ellipse, (255, 255, 255), -1)  # 创建椭圆掩码
        # num_ellipses = len(ellipses_list1)

    cv2.imwrite("ori0627/" + file_name, result_image)
    cv2.imwrite("mask0627/" + file_name, mask)
    return result_image,mask,ellipses_list,num_ellipses, num_contours

#IOU
def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    return (intersection + 1e-15) / (union + 1e-15)
#Dice
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

#像素准确率
def pixel_accuracy(gt, pred):
    # 确保输入的图像具有相同的形状
    if gt.shape != pred.shape:
        raise ValueError("输入的图像形状不匹配")

    # 比较每个像素的类别
    correct_pixels = np.sum(gt == pred)

    # 总像素数量
    total_pixels = gt.size

    # 计算像素准确率
    accuracy = correct_pixels / total_pixels

    return accuracy
#二进制交叉熵
def binary_cross_entropy(y_true, y_pred):
    # 防止概率值为0或1，添加一个微小的偏移
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算二进制交叉熵损失
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    # 求平均损失
    mean_loss = np.mean(loss)

    return mean_loss

#F1—score
def compute_f1_score(y_true, y_pred):
    # 将预测和真实值展平
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    # 计算真正例、假正例和假负例
    true_positives = (y_true * y_pred).sum()
    false_positives = ((1 - y_true) * y_pred).sum()
    false_negatives = (y_true * (1 - y_pred)).sum()
    
    # 计算精确度（Precision）和召回率（Recall）
    precision = true_positives / (true_positives + false_positives + 1e-15)
    recall = true_positives / (true_positives + false_negatives + 1e-15)
    
    # 计算F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)
    
    return f1_score,precision,recall

if __name__ == '__main__':
    dir_path = './Dataset/OriginalData/testing/images/'
    dices_a = []
    dices_c = []
    dices_l = []
    dices_p = []
    ious_a = []
    ious_c = []
    ious_l = []
    ious_p = []
    accuracy_a = []
    accuracy_c = []
    accuracy_l = []
    accuracy_p = []
    bce_a = []
    bce_c = []
    bce_l = []
    bce_p = []
    f1_a= []
    f1_c= []
    f1_l= []
    f1_p= []
    precision_a = []  # 添加 Precision 列表
    precision_c = []  # 添加 Precision 列表
    precision_l = []  # 添加 Precision 列表
    precision_p = []  # 添加 Precision 列表
    recall_p = []
    recall_l = []
    recall_c = []
    recall_a = [] 
    results_list = []
    acc_count_a = []
    acc_count_c = []
    acc_count_l = []
    acc_count_p = []
    num_ellipses = 0
    num_contours = 0
    label_dict = {}
    with open('counts_labels.txt', 'r') as file:
        for line in file:
            # Check if the line is not empty
            if line.strip():
                # Split the line into image and count
                parts = line.strip().split()

                # Check if there are enough values
                if len(parts) == 2:
                    image, count = parts

                    # Convert count to integer
                    count = int(count)

                    # Store in the dictionary
                    label_dict[image] = count
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for file_name in filenames:
            image_ori_ = cv2.imread('./Dataset/OriginalData/testing/images/'+file_name)
            image_seg_ = cv2.imread('./ours/'+file_name)
            # image_seg_ = cv2.imread('./output_predictions/output_predictions/'+file_name)
            image_seg = cv2.cvtColor(image_seg_, cv2.COLOR_BGR2GRAY)
            image_ori = cv2.resize(image_ori_, (image_seg.shape[1], image_seg.shape[0]), interpolation=cv2.INTER_LINEAR)
            contours_all = cv2.findContours(image_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = concave(image_ori)
            # result_image = draw(contours_all, file_name, image_ori)
            result_image, mask, ellipses_list,num_ellipses, num_contours= draw(contours_all, file_name, image_ori)
            total_count = num_ellipses + num_contours
            # 检查二值化图像的尺寸
            binary_height, binary_width = mask.shape[:2]
            # 读取相应的 groundtruth 图像
            groundtruth_image = cv2.imread('./Dataset/OriginalData/testing/segmentations/' + file_name)
            # 确保预测图像和 groundtruth 图像具有相同的大小
            mask = cv2.resize(mask, (groundtruth_image.shape[1], groundtruth_image.shape[0]))
            mask = cv2.resize(mask, (512, 512))
            groundtruth_image = cv2.resize(groundtruth_image, (512, 512))
            mask = mask / 255.0
            groundtruth_image = groundtruth_image / 255.0
            # 将结果图像和 groundtruth 图像转换为NumPy数组
            mask = np.array(mask)
            groundtruth_image = np.array(groundtruth_image)

            iou = compute_iou(mask, groundtruth_image)
            dice_coefficient = single_dice_coef(mask, groundtruth_image)
            accuracy=pixel_accuracy(groundtruth_image, mask)
            bce=binary_cross_entropy(groundtruth_image, mask)
            f1_score, _, _ = compute_f1_score(groundtruth_image, mask)
            _, precision, _ = compute_f1_score(groundtruth_image, mask)
            _, _, recall = compute_f1_score(groundtruth_image, mask)
            # num_contours, num_ellipses, total_count = count_contours_and_ellipses(image_seg, ellipses_list)
            
            
            count = label_dict.get(file_name, 0)
            # print(f"Image: {file_name}, Total Count: {total_count}, Ground Truth Count: {count}")
            # Use your precalculated total_count to calculate accuracy
            accuracy = total_count / count
            difference = total_count-count
            # Append the results to the list
            results_list.append({
                "Image Name": file_name,
                "Num Contours": num_contours,
                "Num Ellipses": num_ellipses,
                "Prediction Count": total_count,
                "labels Count": count,
                "Difference":difference,
                "Accuracy":accuracy
            })
             # 将IOU和Dice系数添加到相应的列表中
            if file_name.startswith('A'):
                ious_a.append(iou)
                dices_a.append(dice_coefficient)
                accuracy_a.append(accuracy)
                bce_a.append(bce)
                f1_a.append(f1_score)
                precision_a.append(precision)
                recall_a.append(recall)
                acc_count_a.append(accuracy)
            elif file_name.startswith('C'):
                ious_c.append(iou)
                dices_c.append(dice_coefficient)
                accuracy_c.append(accuracy)
                bce_c.append(bce)
                f1_c.append(f1_score)
                precision_c.append(precision)
                recall_c.append(recall)
                acc_count_c.append(accuracy)
            elif file_name.startswith('L'):
                ious_l.append(iou)
                dices_l.append(dice_coefficient)
                accuracy_l.append(accuracy)
                bce_l.append(bce)
                f1_l.append(f1_score)
                precision_l.append(precision)
                recall_l.append(recall)
                acc_count_l.append(accuracy)
            elif file_name.startswith('P'):
                ious_p.append(iou)
                dices_p.append(dice_coefficient)
                accuracy_p.append(accuracy)
                bce_p.append(bce)
                f1_p.append(f1_score)
                precision_p.append(precision)
                recall_p.append(recall)
                acc_count_p.append(accuracy)
            else:
                raise ValueError('wrong file name')
    # 计算平均值等其他指标（示例）
    ts_res_mean_a = np.mean(ious_a)
    ts_res_mean_c = np.mean(ious_c)
    ts_res_mean_l = np.mean(ious_l)
    ts_res_mean_p = np.mean(ious_p)

    ts_res_dice_a = np.mean(dices_a)
    ts_res_dice_c = np.mean(dices_c)
    ts_res_dice_l = np.mean(dices_l)
    ts_res_dice_p = np.mean(dices_p)

    ts_res_acc_a = np.mean(accuracy_a)
    ts_res_acc_c = np.mean(accuracy_c)
    ts_res_acc_l = np.mean(accuracy_l)
    ts_res_acc_p = np.mean(accuracy_p)

    ts_res_bce_a = np.mean(bce_a)
    ts_res_bce_c = np.mean(bce_c)
    ts_res_bce_l = np.mean(bce_l)
    ts_res_bce_p = np.mean(bce_p)

    ts_res_f1_a = np.mean(f1_a)
    ts_res_f1_c = np.mean(f1_c)
    ts_res_f1_l = np.mean(f1_l)
    ts_res_f1_p = np.mean(f1_p)

    ts_res_pre_a = np.mean(precision_a)
    ts_res_pre_c = np.mean(precision_c)
    ts_res_pre_l = np.mean(precision_l)
    ts_res_pre_p = np.mean(precision_p)

    ts_res_recall_a = np.mean(recall_a)
    ts_res_recall_c = np.mean(recall_c)
    ts_res_recall_l = np.mean(recall_l)
    ts_res_recall_p = np.mean(recall_p)

    ts_res_count_a = np.mean(acc_count_a)
    ts_res_count_c = np.mean(acc_count_c)
    ts_res_count_l = np.mean(acc_count_l)
    ts_res_count_p = np.mean(acc_count_p)

    print("Average IOU (A):", ts_res_mean_a)
    print("Average IOU (C):", ts_res_mean_c)
    print("Average IOU (L):", ts_res_mean_l)
    print("Average IOU (P):", ts_res_mean_p)

    print("Average Dice (A):", ts_res_dice_a)
    print("Average Dice (C):", ts_res_dice_c)
    print("Average Dice (L):", ts_res_dice_l)
    print("Average Dice (P):", ts_res_dice_p)

    print("Average ACC (A):", ts_res_acc_a)
    print("Average ACC (C):", ts_res_acc_c)
    print("Average ACC (L):", ts_res_acc_l)
    print("Average ACC (P):", ts_res_acc_p)

    print("Average bce (A):", ts_res_bce_a)
    print("Average bce (C):", ts_res_bce_c)
    print("Average bce (L):", ts_res_bce_l)
    print("Average bce (P):", ts_res_bce_p)

    print("Average F1 (A):", ts_res_f1_a)
    print("Average F1 (C):", ts_res_f1_c)
    print("Average F1 (L):", ts_res_f1_l)
    print("Average F1 (P):", ts_res_f1_p)

    print("Average Pre (A):", ts_res_pre_a)
    print("Average Pre (C):", ts_res_pre_c)
    print("Average Pre (L):", ts_res_pre_l)
    print("Average Pre (P):", ts_res_pre_p)

    print("Average Recall (A):", ts_res_recall_a)
    print("Average Recall (C):", ts_res_recall_c)
    print("Average Recall (L):", ts_res_recall_l)
    print("Average Recall (P):", ts_res_recall_p)


    df = pd.DataFrame({
        "Category": ["A", "C", "L", "P"],
        "Average Dice": [ts_res_dice_a, ts_res_dice_c, ts_res_dice_l, ts_res_dice_p],
        "Average IOU": [ts_res_mean_a, ts_res_mean_c, ts_res_mean_l, ts_res_mean_p],
        "Average F1": [ts_res_f1_a, ts_res_f1_c, ts_res_f1_l, ts_res_f1_p],
        "Average Precision": [ts_res_pre_a, ts_res_pre_c, ts_res_pre_l, ts_res_pre_p],
        "Average Recall": [ts_res_recall_a, ts_res_recall_c, ts_res_recall_l, ts_res_recall_p],
        "Accuracy counts": [ts_res_count_a, ts_res_recall_c, ts_res_recall_l, ts_res_recall_p]
    })

    # Save DataFrame to CSV file
    df.to_csv('results_table0627.csv', index=False)

    print("CSV file saved successfully.")

     # Convert the list to a DataFrame
    df = pd.DataFrame(results_list)

    # Save DataFrame to CSV file
    df.to_csv('contours_and_ellipses_counts0627.csv', index=False)

    print("CSV file saved successfully.")