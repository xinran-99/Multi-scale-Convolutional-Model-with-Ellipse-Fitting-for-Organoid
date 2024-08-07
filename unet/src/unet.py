from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# class Down(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__(
#             nn.MaxPool2d(2, stride=2),
#             DoubleConv(in_channels, out_channels)
#         )

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.double_conv = DoubleConv(in_channels * 2, out_channels)

    def forward(self, x):
        # Apply max pooling
        max_pooled = self.max_pool(x)
        
        # Apply average pooling
        avg_pooled = self.avg_pool(x)

        # Concatenate max_pooled and avg_pooled along the channel dimension
        concatenated = torch.cat([max_pooled, avg_pooled], dim=1)

        # Apply DoubleConv to the concatenated result
        output = self.double_conv(concatenated)

        return output


class ResConvUp1(nn.Module):
    def __init__(self, in_channels, k1, k2, k3):
        super(ResConvUp1, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, k1, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        self.branch2 = nn.Conv2d(in_channels, k2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.branch3 = nn.Conv2d(in_channels, k3, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.concat = nn.Sequential(
            nn.Conv2d(k1 + k2 + k3, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # self.up = nn.Upsample(scale_factor=0.5, mode='nearest')

    def forward(self, x):
        branch1 = self.relu1(self.branch1(x))
        branch2 = self.relu2(self.branch2(x))
        branch3 = self.relu3(self.branch3(x))

        y = torch.cat([branch1, branch2, branch3], dim=1)
        out = self.concat(y)

        return out

class ResConvUp(nn.Module):
    def __init__(self, in_channels, k1, k2, k3):
        super(ResConvUp, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, k1, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        self.branch2 = nn.Conv2d(in_channels, k2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.branch3 = nn.Conv2d(in_channels, k3, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.concat = nn.Sequential(
            nn.Conv2d(k1 + k2 + k3, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=0.5, mode='nearest')

    def forward(self, x):
        branch1 = self.relu1(self.branch1(x))
        branch2 = self.relu2(self.branch2(x))
        branch3 = self.relu3(self.branch3(x))

        y = torch.cat([branch1, branch2, branch3], dim=1)
        out = self.concat(y)
        out = self.up(out)

        return out

        # out = self.up(out)

        # return out

# class Conv1x1Module(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Conv1x1Module, self).__init__()
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv1x1(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
 
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
 
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        target_size = x1.size()[2:]
        g1 = F.interpolate(g1, size=target_size, mode='bilinear', align_corners=False)

        # print("g1 size:", g1.size())
        # print("x1 size:", x1.size())
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
 
        return x * psi

class up_conv1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv1, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=8),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.up(x)

class up_conv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv2, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.up(x)

class up_conv3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv3, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.up(x)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# class UNet(nn.Module):
#     def __init__(self,
#                  in_channels: int = None,
#                  num_classes: int = None,
#                  bilinear: bool = True,
#                  base_c: int = None):
#         super(UNet, self).__init__()
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16)
        # self.down1 = Down(base_c, base_c * 2, base_c)
        # self.down2 = Down(base_c * 2, base_c * 4, base_c * 2)
        # self.down3 = Down(base_c * 4, base_c * 8, base_c * 4)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(base_c * 8, base_c * 16, base_c * 8)
        # #self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.multi_conv1 = ResConvUp1(base_c , base_c//4 , base_c//2 , base_c//4)
        self.multi_conv2 = ResConvUp(base_c * 2 , base_c//2 , base_c ,base_c//2)
        self.multi_conv3 = ResConvUp(base_c * 4 , base_c , base_c * 2 , base_c)
        self.multi_conv4 = ResConvUp(base_c * 8 , base_c * 2 , base_c * 4 ,base_c * 2)
        self.multi_conv5 = ResConvUp(base_c * 16 , base_c * 4 , base_c * 8  , base_c * 4)
        self.conv1x1 = OutConv(base_c * 32 , base_c * 16)          
        self.att1 = Attention_block(base_c * 16, base_c * 16 ,base_c * 16)
        self.up1 = Up(base_c * 32, base_c * 16 // factor, bilinear)
        self.att2 = Attention_block(base_c * 8, base_c * 8 ,base_c * 8)
        self.up2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.att3 = Attention_block(base_c * 4, base_c * 4 ,base_c * 4)       
        self.up3 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.att4 = Attention_block(base_c * 2 , base_c * 2 ,base_c * 2)
        self.up4 = Up(base_c * 4, base_c * 4 // factor, bilinear)
        self.out1 = up_conv1(base_c * 8 , base_c * 2)
        self.out2 = up_conv2(base_c * 4 , base_c * 2)
        self.out3 = up_conv3(base_c * 2 , base_c * 2)
        self.out4 = Conv(base_c * 2 , base_c * 2)
        self.out_conv1 = DoubleConv(base_c * 8 , base_c)
        self.out_conv2 = OutConv(base_c, num_classes)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("x5 shape:", x5.shape)
        res1 =  self.multi_conv1(x1) 
        # print("res1 shape:", res1.shape)
        # merge1 = np.concatenate((x1.detach().numpy(), res1.detach().numpy()), axis=1)
        # print("merge1 shape:", merge1.shape)
        merge1 = torch.cat([x1, res1], dim=1)
        res2 =  self.multi_conv2(merge1)
        # print("res2 shape:", res2.shape)
        # merge2 = torch.cat([x2.view(-1), res2.view(-1)], dim=0)
        merge2 = torch.cat([x2, res2], dim=1) 
        res3 =  self.multi_conv3(merge2)
        # print("res3 shape:", res3.shape)
        # merge3 = torch.cat([x3.view(-1), res3.view(-1)], dim=0)
        merge3 = torch.cat([x3, res3], dim=1) 
        res4 =  self.multi_conv4(merge3) 
        # print("res4 shape:", res4.shape)
        # merge4 = torch.cat([x4.view(-1), res4.view(-1)], dim=0)
        merge4 = torch.cat([x4, res4], dim=1)
        # print("merge4 shape:", merge4.shape)
        res5 =  self.multi_conv5(merge4) 
        # print("res5 shape:", res5.shape)
        # merge5 = torch.cat([x5.view(-1), res5.view(-1)], dim=0)
        merge5 = torch.cat([x5, res5], dim=1)
        # print("merge5 shape:", merge5.shape)
        merge6 =self.conv1x1(merge5)
        # print("merge6 shape:", merge6.shape)

        merge4 = self.att1(merge6, merge4)
        # print("merge4 shape:", merge4.shape)
        d5 = self.up1(merge6, merge4)
        # print("d5 shape:", d5.shape)
        merge3 = self.att2(d5, merge3)
        d4 = self.up2(d5, merge3)
        merge2 = self.att3(d4, merge2)        
        d3 = self.up3(d4, merge2)
        merge1 = self.att4(d3, merge1)
        d2 = self.up4(d3, merge1)
        # x4 = self.att1(x5, x4)
        # d5 = self.up1(x5, x4)
        # x3 = self.att2(d5, x3)
        # d4 = self.up2(d5, x3)
        # x2 = self.att3(d4, x2)        
        # d3 = self.up3(d4, x2)
        # x1 = self.att4(d3, x1)
        # x = self.up4(d3, x1)
        out1 = self.out1(d5)
        # print("out1 shape:", out1.shape)
        out2 = self.out2(d4)
        # print("out2 shape:", out2.shape)
        out3 = self.out3(d3)
        # print("out3 shape:", out3.shape)
        out4 = self.out4(d2)
        # print("out4 shape:", out4.shape)
        final_concat = torch.cat([out1, out2, out3, out4], dim=1)
        # print("final_concat shape:", final_concat.shape)
        output = self.out_conv1(final_concat)
        logits = self.out_conv2(output)
        # logits = self.out_conv(x)

        # return {"out": logits}
        return {"out": logits, "aux1": out1, "aux2": out2, "aux3": out3, "aux4": out4}

if __name__ == '__main__':
    net = UNet(3, 2, 64)
    x = torch.randn(1, 3, 512, 512)
    y = net(x)['out']
    print(x.shape)
    print("Output shape (y):", y.shape)
