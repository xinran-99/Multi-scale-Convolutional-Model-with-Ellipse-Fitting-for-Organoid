import os
import time
import datetime

import torch
import numpy as np
from PIL import Image
# from src import UNetBC
from src import UNet

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
# from torchviz import make_dot
# import hiddenlayer as hl
from torchsummary import summary
# from ann_visualizer.visualize import ann_viz
from skimage import exposure
    # 绘制训练损失和验证损失图表
import matplotlib.pyplot as plt
from predict import test
import warnings
from src import unet
# nohup python train.py > output.txt 2>&1 &

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

class SegmentationPresetTrain:
    # def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
    #              mean=0.5, std=0.5):
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.Resize((512, 512)),
            # T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)
# 添加CLAHE对比度增强操作
        # trans.append(CLAHE())
        # trans.extend([
        #     T.RandomCrop(crop_size),
        #     T.ToTensor(),
        #     T.Normalize(mean=mean, std=std),
        # ])
        # self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    # def __init__(self, mean=0.5, std=0.5):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RandomCrop(512),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
    
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
# def get_transform(train, mean = 0.5,std = 0.5):    
    base_size = 512
    crop_size = 512

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

 
def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # Manually inspecting the model's forward pass
    input_tensor = torch.randn(1, 3, 512, 512)  # Adjust the shape if needed
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    # # Print the shapes of intermediate layers
    # print("\nSummary of intermediate layers:")
    # for key, value in output.items():
    #     print(f"{key}: {value.shape}")

    return model



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # mean = 0.5
    # std = 0.5
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    subset_fraction = 1  # Use 10% of the dataset for testing

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            num_workers=num_workers,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            collate_fn=train_dataset.collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True,
                                           pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=1,
    #                                          num_workers=num_workers,
    #                                          pin_memory=True,
    #                                          collate_fn=val_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            pin_memory=True)
    # for img, mask in train_loader:
    #     print(f"Training Image shape: {img.shape}, Mask shape: {mask.shape}")

    model = create_model(num_classes=num_classes)
    # model = create_model()

    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0
    start_time = time.time()
    train_losses = []  # 用于保存训练损失
    val_losses = []
    temp_weights_path = "./save_weights/0304temp_weights_mod.pth"
    # Define a variable to keep track of how many epochs the Dice coefficient hasn't improved
    patience = 50
    count = 0
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        train_losses.append(mean_loss)
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        val_losses.append(dice)

        temp_save_file = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args}
        if args.amp:
            temp_save_file["scaler"] = scaler.state_dict()

        torch.save(temp_save_file, temp_weights_path)

        # Test using the temporary weights
        total_average = test(model, temp_weights_path)
        print(f"total_average:{total_average}")

        if total_average > best_dice:
            best_dice = total_average
            # torch.save(save_file, "save_weights/best_model5.pth")
            torch.save(temp_save_file, f"save_weights/lr0.01_0326_model_unetBC_{epoch}_{total_average}.pth")
            count = 0
        else:
            count += 1
            if count >= patience:
                print(f'Early stop at epoch {epoch + 1}')
                break
        print(f"best_dice:{best_dice}")
        print(f"count:{count}")

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"\
                         f"total_average:{total_average}\n"
            # train_info = f"[epoch: {epoch}]\n" \
            #              f"train_loss: {mean_loss:.4f}\n" \
            #              f"lr: {lr:.6f}\n" \
            #              f"total_average:{total_average}\n"

            f.write(train_info + val_info + "\n\n")
            # f.write(train_info  + "\n\n")

        torch.cuda.empty_cache()

        # if args.save_best is True:
        #     torch.save(save_file, "save_weights/best_model5.pth")
        # else:
        #     torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Dice Coefficient')
    plt.legend()
    plt.show()
    plt.savefig('figure_loss.png')  # 指定您想要保存的路径和文件名

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    # parser.add_argument("--device", default="cuda", help="training device")

    parser.add_argument("-b", "--batch-size", default=2 , type=int)
    parser.add_argument("--epochs", default=1000, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)