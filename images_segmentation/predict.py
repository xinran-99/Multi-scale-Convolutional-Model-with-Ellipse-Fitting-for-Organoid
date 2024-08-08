import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
# from src import UNetBC
from skimage import exposure
from evaluation_metrics import calculate_metrics
import pandas as pd
import warnings
# from src import MyNet
from src import SegNet
from src import AttU_Net
from src import OriUNet
from src import VGG16UNet
from src import MyNet
from src import MultiResUnet
from src import build_doubleunet
from src import ACCoNet_Res
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def predict_single_image(model, img_path, output_folder, device, mean, std):
    # 使用CLAHE进行对比度增强
    original_img = Image.open(img_path)
    original_array = np.array(original_img)
    enhanced_array = exposure.equalize_adapthist(original_array, clip_limit=0.03)
    enhanced_img = Image.fromarray((enhanced_array * 255).astype(np.uint8))
    # enhanced_img.save(os.path.join(output_folder, f"enhanced_{os.path.basename(img_path)}"))
    original_img = enhanced_img.convert('RGB')
    # original_img.save(os.path.join(output_folder, f"original_{os.path.basename(img_path)}"))

    # original_img = Image.open(img_path).convert('RGB')
    # original_img = Image.open(img_path)

    # 从PIL图像到张量并归一化
    data_transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    # data_transform = transforms.Compose([transforms.Resize((904, 1224)),
    #                                 transforms.ToTensor(),
    #                                  transforms.Normalize(mean=mean, std=std)])
    # data_transform = transforms.Compose([
    #     # 不进行尺寸调整
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])
    img = data_transform(original_img)
    # 增加批次维度
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        t_start = time_synchronized()
        output_dict = model(img.to(device))
        t_end = time_synchronized()
        # print(f"推理时间: {t_end - t_start}秒")

        # 提取模型输出张量
        # output_tensor = output_dict['out']
#_______________________________segnet
        output_tensor = output_dict['out'] if isinstance(output_dict, dict) else output_dict

        prediction = output_tensor.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将器官细胞对应的像素值设置为255（白色）
        prediction[prediction == 1] = 255

        # 保存预测结果
        result_image_path = os.path.join(output_folder, os.path.basename(img_path))
        result_image = Image.fromarray(prediction)
        result_image.save(result_image_path)

epoch_metrics_list = []

def test(model, epoch):
    classes = 2  # 背景和器官细胞，根据你的数据集调整
    img_folder = "./OriginalData/testing/images/"
    output_folder = "./output_predictions/"
    ground_truth_folder="./OriginalData/testing/segmentations"

    total_dice = 0.0
    total_iou = 0.0
    assert os.path.exists(img_folder), f"image folder {img_folder} not found."
    
    os.makedirs(output_folder, exist_ok=True)

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 设备.")
    weights_path = f"./save_weights/0304temp_weights_mod.pth"  # 在权重路径中使用当前 epoch

    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    # model = model.cuda()

    # df = pd.DataFrame(columns=['Class Prefix', 'Dice', 'IOU','mIOU', 'F1', 'Precision', 'Recall', 'Accuracy', 'Confusion Matrix'])
    df = pd.DataFrame(columns=['Class Prefix', 'Dice', 'IOU','F1', 'Precision', 'Recall', 'Accuracy', 'Confusion Matrix'])
    # 遍历图像文件夹中的每个图像
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            predict_single_image(model, img_path, output_folder, device, mean, std)
            true_labels_path = os.path.join(ground_truth_folder, img_name)  # 替换为真实标签的文件夹路径
            result_image_path = os.path.join(output_folder, os.path.basename(img_path))
            metrics_result = calculate_metrics(true_labels_path, result_image_path, class_prefix=img_name[0])
            # 将结果添加到 DataFrame
            # 将 Dice 和 IOU 添加到总和
            total_dice += metrics_result['Dice']
            total_iou += metrics_result['IOU']
            df = df.append(metrics_result, ignore_index=True)
            # print(f"Metrics for {img_name}: {metrics_result}")

            # 将指标添加到DataFrame
            df = df.append({
                'Class': img_name[0],
                'Dice': metrics_result['Dice'],
                'IOU': metrics_result['IOU'],
                # 'mIOU': metrics_result['mIOU'],
                'F1': metrics_result['F1'],
                'Precision': metrics_result['Precision'],
                'Recall': metrics_result['Recall'],
                'Accuracy': metrics_result['Accuracy'],
                'Confusion Matrix':metrics_result['Confusion Matrix'],
            }, ignore_index=True)
 # 计算每个类别的指标平均值
    avg_metrics = df.groupby('Class').mean().reset_index()
# 计算所有图片的 Dice 和 IOU 的平均值
    total = total_dice + total_iou
    # total_average=np.mean(total)
    total_average=total/56
    print(f"total_average:{total_average}")
    #————————————————————————————————————————————————————————————————
    # # 保存DataFrame到CSV文件
    # avg_metrics.to_csv('average_metrics_results.csv', index=False)
    #————————————————————————————————————————————————————————————————
    # 将 epoch 和指标添加到列表中
    epoch_metrics_list.append({'Epoch': epoch, 'Total Average': total_average, 'Metrics': avg_metrics})

    # 在整个训练循环结束后保存评价指标到表格文件
    final_metrics_df = pd.concat([entry['Metrics'] for entry in epoch_metrics_list], ignore_index=True)
    final_metrics_df.to_csv('average_metrics_results_test_unetBC.csv', index=False)
    #________________________________________________________________
    return total_average

#训练时可以把main注释掉
def main():
    classes = 2  # 背景和器官细胞，根据你的数据集调整
    # weights_path = "./duibi experiment/lr0.01_0326_model_ACCoNet_193_0.7243051877763677.pth"#需要改成要测的模型
    # img_folder = "./OriginalData/testing/images/"
    # output_folder = "./output_predictions_ACConet/"
    # ground_truth_folder="./OriginalData/testing/segmentations"
    weights_path = f"./duibi experiment/lr0.01_0326_model_ACCoNet_193_0.7243051877763677.pth"  # 在权重路径中使用当前 epoch
    img_folder = "./orgaextractor/images/"
    output_folder = "./output_predictions_orgaextractor/ACCoNet/"
    ground_truth_folder="./orgaextractor/segmentations"
    total_dice = 0.0
    total_iou = 0.0
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_folder), f"image folder {img_folder} not found."
    
    os.makedirs(output_folder, exist_ok=True)

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 设备.")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # model = MyNet(in_channels=3, classes=classes)
    # model = UNet(in_channels=3, num_classes=classes, base_c=64)
    # model = UNetB(in_channels=3, num_classes=classes, base_c=64)
    # model = SegNet(num_classes=2)
    model = ACCoNet_Res(channel=64,num_classes=2)
    # model = MultiResUnet(input_channels=3, num_classes=2, alpha=1.67)
    # model = AttU_Net(in_channel=3,num_classes=2,channel_list=[64, 128, 256, 512, 1024],checkpoint=False,convTranspose=True)
    # model = OriUNet(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])

 
    model.to(device)
    # df = pd.DataFrame(columns=['Class Prefix', 'Dice', 'IOU', 'mIOU', 'F1', 'Precision', 'Recall', 'Accuracy', 'Confusion Matrix'])
    df = pd.DataFrame(columns=['Class Prefix', 'Dice', 'IOU','F1', 'Precision', 'Recall', 'Accuracy', 'Confusion Matrix'])

    # 遍历图像文件夹中的每个图像
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            predict_single_image(model, img_path, output_folder, device, mean, std)
            true_labels_path = os.path.join(ground_truth_folder, img_name)  # 替换为真实标签的文件夹路径
            result_image_path = os.path.join(output_folder, os.path.basename(img_path))
            metrics_result = calculate_metrics(true_labels_path, result_image_path, class_prefix=None)
            print(f"Dice value for {img_name}: {metrics_result['Dice']}")

            # 将结果添加到 DataFrame
            df = df.append(metrics_result, ignore_index=True)

    # 计算所有图片的 Dice 和 IOU 的平均值
    avg_metrics = df.mean()
    # 创建 DataFrame
    df = pd.DataFrame()

    # # 从DataFrame中选择评价指标的平均值
    # avg_metrics = df.mean().reset_index()

    # # 选择需要保留的列
    # avg_metrics = avg_metrics[['Dice', 'IOU', 'F1', 'Precision', 'Recall', 'Accuracy']]

    # # 重命名列名
    # avg_metrics.columns = ['Average Dice', 'Average IOU', 'Average F1', 'Average Precision', 'Average Recall', 'Average Accuracy']

    # 保存DataFrame到CSV文件
    avg_metrics.to_csv('average_metrics_results_orgaextractor_ACCoNet.csv', index=False)

if __name__ == '__main__':
    main()
