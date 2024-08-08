import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image

def calculate_metrics(true_labels_path, result_image_path, class_prefix=''):
    # # # Load true labels and predicted results
    # # true_labels = load_labels(true_labels_path)
    # # predicted_labels = load_labels(result_image_path)
    # true_labels_img = Image.open(true_labels_path).convert('L')
    # true_labels = np.array(true_labels_img) / 255  # 假设标签图像是二值化的，进行归一化

    # # Flatten the labels
    # true_labels_flat = true_labels.flatten()

    # result_image_img = Image.open(result_image_path).convert('L')
    # predicted_labels = np.array(result_image_img) / 255  # 如果需要，进行归一化

    # predicted_labels_flat = predicted_labels.flatten()
    ##---------------------new------------------------------##
    # 将真实标签和预测标签调整为相同尺寸
    true_labels_flat = resize_and_flatten_image(true_labels_path, (512, 512))
    predicted_labels_flat = resize_and_flatten_image(result_image_path, (512, 512))
    # true_labels_flat = flatten_image(true_labels_path)
    # predicted_labels_flat = flatten_image(result_image_path)
    # with Image.open(true_labels_path) as img:
    #     true_labels_size = img.size
    # with Image.open(result_image_path) as img:
    #     result_image_size = img.size
    # true_labels_flat = resize_and_flatten_image(true_labels_path, true_labels_size)
    # predicted_labels_flat = resize_and_flatten_image(result_image_path, result_image_size)

    ##---------------------new------------------------------##

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels_flat, predicted_labels_flat)

    # Calculate metrics
    # dice = calculate_dice(cm)
    dice = calculate_dice(true_labels_flat, predicted_labels_flat)
    iou = calculate_iou(cm)
    miou = calculate_miou(cm)
    # miou = calculate_miou(true_labels_flat, predicted_labels_flat)
    f1 = calculate_f1(cm)
    precision = calculate_precision(cm)
    recall = calculate_recall(cm)
    accuracy = calculate_accuracy(cm)

    # You can return the metrics if needed in your code
    return {
        'Class Prefix': class_prefix,
        'Dice': dice,
        'IOU': iou,
        'mIOU': miou,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Confusion Matrix': cm
    }
def flatten_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img) // 255
    return img_array.flatten()
def resize_and_flatten_image(image_path, new_size):
    # 打开图像并转换为灰度模式
    img = Image.open(image_path).convert('L')
    # 调整图像尺寸
    resized_img = img.resize(new_size)
    # 将图像转换为 NumPy 数组并进行整除操作
    img_array = np.array(resized_img) // 255
    # 扁平化数组
    return img_array.flatten()

def calculate_dice(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]

    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return dice

def calculate_dice(true_labels_flat, predicted_labels_flat):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(true_labels_flat * predicted_labels_flat)
    if (np.sum(true_labels_flat)==0) and (np.sum(predicted_labels_flat)==0):
        return 1
    return (2*intersection) / (np.sum(true_labels_flat) + np.sum(predicted_labels_flat))

def calculate_iou(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]

    iou = true_positive / (true_positive + false_positive + false_negative)
    return iou

def calculate_miou(confusion_matrices):
    num_classes = len(confusion_matrices)
    total_miou = 0.0

    for class_index in range(num_classes):
        # 计算 MIoU 的分子（对角线上的值）
        true_positive = confusion_matrices[class_index][class_index]
        
        # 计算 MIoU 的分母（第 i 行的值 + 第 i 列的值 - 对角线上的值）
        denominator = sum(confusion_matrices[class_index]) + sum(confusion_matrices[i][class_index] for i in range(num_classes)) - true_positive
        
        # 防止分母为零的情况
        if denominator == 0:
            iou_class = 0.0
        else:
            iou_class = true_positive / denominator

        # print(f'MIoU for Class {class_index}: {iou_class}')

        total_miou += iou_class

    miou = total_miou / num_classes
    return miou


# def calculate_miou(true_labels_flat, predicted_labels_flat):
#     y_pred = predicted_labels_flat.flatten()  # 将预测掩模展平为一维数组
#     y_true = true_labels_flat.flatten()  # 将真实掩模展平为一维数组
    
#     intersection = (y_true * y_pred).sum()  # 计算交集
#     union = y_true.sum() + y_pred.sum() - intersection  # 计算并集

#     return (intersection + 1e-15) / (union + 1e-15)  # 计算IoU，避免分母为零

def calculate_f1(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_precision(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]

    precision = true_positive / (true_positive + false_positive)
    return precision

def calculate_recall(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    false_negative = confusion_matrix[1, 0]

    recall = true_positive / (true_positive + false_negative)
    return recall

def calculate_accuracy(confusion_matrix):
    true_positive = confusion_matrix[1, 1]
    true_negative = confusion_matrix[0, 0]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return accuracy

