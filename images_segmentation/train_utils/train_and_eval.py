import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
#     losses = {}
#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
#         #______________________
#         # # 添加IoU损失项
#         # iou_loss_value = iou_loss(x, build_target(target, num_classes, ignore_index))
#         # loss += iou_loss_value
#         #______________________
#         if dice is True:
#             dice_target = build_target(target, num_classes, ignore_index)
#             # loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
#             loss += dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)

#         losses[name] = loss

#     if len(losses) == 1:
#         return losses['out']

#     return losses['out'] + 0.5 * losses['aux']

# # ————————————————————————————————————————————segnet模型修改
# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
#     losses = {}
#     if isinstance(inputs, dict):
#         for name, x in inputs.items():
#             # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#             loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)

#             #______________________
#             # # 添加IoU损失项
#             # iou_loss_value = iou_loss(x, build_target(target, num_classes, ignore_index))
#             # loss += iou_loss_value
#             #______________________
#             if dice is True:
#                 dice_target = build_target(target, num_classes, ignore_index)
#                 # loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
#                 loss += dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)

#             losses[name] = loss
#     else:
#         # 如果输入不是字典，则默认为单个输出，并且命名为 'out'
#         losses['out'] = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
#         if dice is True:
#             dice_target = build_target(target, num_classes, ignore_index)
#             losses['out'] += dice_loss(inputs, dice_target, multiclass=False, ignore_index=ignore_index)

#     if len(losses) == 1:
#         return losses['out']

#     return losses['out'] + 0.5 * losses['aux']

# #————————————————————————没加辅助损失的loss————————————————————————————————
# # def criterion(inputs, target, loss_weight=None, num_classes: int = 2, ignore_index: int = -100, weight_iou: float = 1.1, weight_dice: float = 1.6 ,weight_focal: float = 1.6 ):
# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, ignore_index: int = -100, weight_iou=None, weight_dice=None,weight_focal=None):

#     losses = {}

#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
#         # 添加 IoU 损失项
#         iou_loss_value = iou_loss(x, build_target(target, num_classes, ignore_index))
#         # 调整 IoU 损失的权重
#         loss += weight_iou * iou_loss_value

#         # 调整 Dice 损失的权重
#         dice_target = build_target(target, num_classes, ignore_index)
#         #____________________________
#         # loss += weight_dice * dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)
#         #____________________________
#         dice_loss_value = dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)
#         loss += weight_dice * dice_loss_value
#         #____________________________
#         # 使用 Focal Loss
#         focal_loss = FocalLoss(gamma=2, alpha=None, ignore_index=ignore_index)
#         focal_loss_value = focal_loss(x, target)
#         loss += weight_focal * focal_loss_value
       
#         losses[name] = loss

#     #     print(f"{name} Loss: {loss.item()}")
#     #     print(f"{name} IoU Loss: {iou_loss_value.item()}")
#     #     print(f"{name} Dice Loss: {dice_loss_value.item()}")


#     # print("Losses Dictionary:")
#     # for key, value in losses.items():
#     #     print(f"{key}: {value}")

#     # 如果只有一个损失项，直接返回该项损失；否则返回多项损失的加权和
#     if len(losses) == 1:
#         return losses['out']
#     # else:
#     #     # 这里是一个简单的加权和，你可能需要调整权重
#     #     return losses['out'] + 0.5 * losses['aux']


#——————————————————————加了辅助损失的loss————————————————————————
# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, ignore_index: int = -100, weight_iou: float = 1.1, weight_dice: float = 1.6 ,weight_focal: float = 1.6 ):
def criterion(inputs, target, loss_weight=None, num_classes: int = 2, ignore_index: int = -100, weight_iou=None, weight_dice=None,weight_focal=None):

    losses = {}
    for name, x in inputs.items():
        #————————————————————————————————————————二进制交叉熵损失
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        loss = nn.functional.cross_entropy(inputs['out'], target, ignore_index=ignore_index, weight=loss_weight)        #——————————————————————————————————————————-iouloss
        #___________________________________________iouloss
        # iou_loss_value = iou_loss(x, build_target(target, num_classes, ignore_index))
        iou_loss_value = iou_loss(inputs['out'], build_target(target, num_classes, ignore_index))
        # 调整 IoU 损失的权重
        loss += weight_iou * iou_loss_value
        #——————————————————————————————————————-——-——diceloss
        # 调整 Dice 损失的权重
        dice_target = build_target(target, num_classes, ignore_index)
        #____________________________
        # loss += weight_dice * dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)
        #____________________________
        # dice_loss_value = dice_loss(x, dice_target, multiclass=False, ignore_index=ignore_index)
        dice_loss_value = dice_loss(inputs['out'], dice_target, multiclass=False, ignore_index=ignore_index)
        loss += weight_dice * dice_loss_value
        #__________________________________________  focal Loss
        focal_loss = FocalLoss(gamma=2, alpha=None, ignore_index=ignore_index)
        # focal_loss_value = focal_loss(x, target)
        focal_loss_value = focal_loss(inputs['out'], target)
        loss += weight_focal * focal_loss_value
        # losses[name] = loss
        losses['out'] = loss
        #_________________________________
        # 计算辅助损失
        if 'aux1' in inputs:
            aux1_loss = focal_loss(inputs['aux1'], target)  
            # loss += aux1_loss
            losses['aux1'] = aux1_loss

        if 'aux2' in inputs:
            aux2_loss = focal_loss(inputs['aux2'], target)  
            # loss += aux2_loss
            losses['aux2'] = aux2_loss

        if 'aux3' in inputs:
            aux3_loss = focal_loss(inputs['aux3'], target)  
            # loss += aux3_loss
            losses['aux3'] = aux3_loss

        if 'aux4' in inputs:
            aux4_loss = focal_loss(inputs['aux4'], target)  
            # loss += aux4_loss
            losses['aux4'] = aux4_loss
        #______________________________________
        # losses[name] = loss

    # print(f"{name} Loss: {loss.item()}")
    # print(f"{name} IoU Loss: {iou_loss_value.item()}")
    # print(f"{name} Dice Loss: {dice_loss_value.item()}")
    # print(f"{name} aux1 Loss: {aux1_loss.item()}")
    # print(f"{name} aux2 Loss: {aux2_loss.item()}")
    # print(f"{name} aux3 Loss: {aux3_loss.item()}")
    # print(f"{name} aux4 Loss: {aux4_loss.item()}")

    # print("Losses Dictionary:")
    # for key, value in losses.items():
    #     print(f"{key}: {value}")

    # 如果只有一个损失项，直接返回该项损失；否则返回多项损失的加权和
    if len(losses) == 1:
        return losses['out']
    else:
        # 这里是一个简单的加权和，你可能需要调整权重
        # return losses['out'] + 0.5 * losses['aux']
        return losses['out'] + 1.5* losses['aux1'] + 0.8 * losses['aux2'] + 0.4 * losses['aux3'] + 0.8 * losses['aux4'] 

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'
    header = 'validation:'

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            # print("Target shape:", target.shape)
            output = model(image)
            output = output['out']
#————————————————————————————————————segnet
            # output = output['out'] if isinstance(output, dict) else output
#——————————————————————————————————————
            # print("Output shape:", output.shape)
            
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        # 假设你的目标张量是 target
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # print(output)
            # loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            ## loss = criterion(output, target, loss_weights=[ce_weight, dice_weight, iou_weight], num_classes=num_classes, ignore_index=255)
            # loss = criterion(output, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255, weight_iou=0, weight_dice=0)
            
            loss = criterion(output, target, loss_weight=loss_weight, num_classes=num_classes, ignore_index=255, weight_iou= 0, weight_dice= 0 ,weight_focal= 0 )

            # print("Loss calculation shape:", loss.shape)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    #_________________________________________________
            # del intermediate_result_detached
    #___________________________________________________
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# def iou_loss(predicted, target):
#     intersection = torch.sum(predicted * target)
#     union = torch.sum(predicted) + torch.sum(target) - intersection
#     iou = intersection / (union + 1e-8)  # 避免除零错误
#     return 1 - iou

def iou_loss(predicted, target, ignore_index=-100):
    # Apply ignore mask if specified
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        predicted[ignore_mask] = 0
        target[ignore_mask] = 0

    # Convert predicted values to binary mask using a threshold
    predicted_binary = (predicted > 0.5).float()

    # Convert target to binary mask (0 or 1)
    target_binary = (target > 0.5).float()

    intersection = torch.sum(predicted_binary * target_binary)
    union = torch.sum(predicted_binary) + torch.sum(target_binary) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)  # Avoid division by zero
    return 1 - iou

class IoULoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, predicted, target):
        return iou_loss(predicted, target, ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)

    def forward(self, logits, target):
        ce_loss = self.criterion(logits, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss
terion = FocalLoss(gamma=2, alpha=None, ignore_index=-100)

# import torch
# from torch import nn
# import train_utils.distributed_utils as utils
# from .dice_coefficient_loss import dice_loss, build_target

# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
#     losses = {}
#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
#         if dice is True:
#             dice_target = build_target(target, num_classes, ignore_index)
#             loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
#         losses[name] = loss

#     if len(losses) == 1:
#         return losses['out']

#     return losses['out'] + 0.5 * losses['aux']


# def evaluate(model, data_loader, device, num_classes):
#     model.eval()
#     confmat = utils.ConfusionMatrix(num_classes)
#     dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     # header = 'Test:'
#     header = 'validation:'

#     with torch.no_grad():
#         for image, target in metric_logger.log_every(data_loader, 100, header):
#             image, target = image.to(device), target.to(device)
#             print("Target shape:", target.shape)

#             output = model(image)
#             print("Output shape:", output.shape)

#             if isinstance(output, dict):
#                 output = output['out']

#             confmat.update(target.flatten(), output.argmax(1).flatten())
#             dice.update(output, target)

#         confmat.reduce_from_all_processes()
#         dice.reduce_from_all_processes()

#     return confmat, dice.value.item()


# def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
#                     lr_scheduler, print_freq=10, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)

#     if num_classes == 2:
#         # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
#         loss_weight = torch.as_tensor([1.0, 2.0], device=device)
#     else:
#         loss_weight = None

#     for image, target in metric_logger.log_every(data_loader, print_freq, header):
#         # 假设你的目标张量是 target
#         image, target = image.to(device), target.to(device)
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             output = model(image)
#             # print(type(output))
#             output_dict = {'out': output}  # 将 output 转换为字典形式
#             loss = criterion(output_dict, target, loss_weight, num_classes=num_classes, ignore_index=255)
#             print("Loss calculation shape:", loss.shape)

#             # loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
#             # loss = criterion(output['out'], target, loss_weight, num_classes=num_classes, ignore_index=255)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()

#         lr_scheduler.step()

#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(loss=loss.item(), lr=lr)

#     return metric_logger.meters["loss"].global_avg, lr


# def create_lr_scheduler(optimizer,
#                         num_step: int,
#                         epochs: int,
#                         warmup=True,
#                         warmup_epochs=1,
#                         warmup_factor=1e-3):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0

#     def f(x):
#         """
#         根据step数返回一个学习率倍率因子，
#         注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
#         """
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             # warmup后lr倍率因子从1 -> 0
#             # 参考deeplab_v2: Learning rate policy
#             return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
