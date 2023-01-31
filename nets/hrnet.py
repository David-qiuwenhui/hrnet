import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone="hrnetv2_w18", pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x):
        # ------ module1 ------ Conv3x3 + Conv3x3 + layer1
        x = self.model.conv1(x)  # x(bs, 3, 480, 480) -> x(bs, 64, 240, 240)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)  # x(bs, 64, 120, 120)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)  # x(bs, 256, 120, 120)
        # ------ module2 ------ Transition1 + Stage2
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)
        # ------ module3 ------ Transition2 + Stage3
        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)
        # ------ module4 ------ Transition3 + Stage4
        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        # ylist[0] (bs, 32, 120, 120), ylist[1] (bs, 64, 60, 60), ylist[2] (bs, 128, 30, 30)  ylist[3] (bs, 256, 15, 15)
        return y_list


class HRnet(nn.Module):
    def __init__(self, num_classes=21, backbone="hrnetv2_w18", pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(
            backbone=backbone, pretrained=pretrained
        )  # backbone="hrnetv2_w32", pretrained=False

        last_inp_channels = np.sum(
            self.backbone.model.pre_stage_channels, dtype=int
        )  # pre_stage_channels = [32, 64, 128, 256]  channels_sum=480

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)  # inputs(bs, 3, H, W)
        x = self.backbone(
            inputs
        )  # x0(B, 32, H/4, W/4), x1(B, 64, H/8, W/8), x2(B, 128, H/16, W/16), x3(B, 256, H/32, W/32)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)  # H/4, W/4
        x1 = F.interpolate(
            input=x[1], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x1(B, 64, H/4, W/4)
        x2 = F.interpolate(
            input=x[2], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x2(B, 128, H/4, W/4)
        x3 = F.interpolate(
            input=x[3], size=(x0_h, x0_w), mode="bilinear", align_corners=True
        )  # x3(B, 256, H/4, W/4)
        # Concat feature maps
        x = torch.cat(tensors=[x[0], x1, x2, x3], dim=1)  # x(B, 480, H/4, W/4)
        # Convolution block90
        x = self.last_layer(x)  # x(B, N, H/4, W/4)
        # Upsampling
        x = F.interpolate(
            input=x, size=(H, W), mode="bilinear", align_corners=True
        )  # x(B, N, H, W)
        return x

        # # ---------- 测试脚本 ----------
        # x0, x1, x2, x3 = self.backbone(inputs)
        # # Upsampling
        # x0_h, x0_w = x0.size(2), x0.size(3)  # x0_h = 120, x0_w = 120
        # x1 = F.interpolate(input=x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # x2 = F.interpolate(input=x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # x3 = F.interpolate(input=x3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # # Concat feature maps
        # x = torch.cat(tensors=[x0, x1, x2, x3], dim=1)  # x(bs, 480, 120, 120)
        # # Convolution block
        # x = self.last_layer(x)
        # # Upsampling
        # x = F.interpolate(input=x, size=(H, W), mode='bilinear', align_corners=True)
        # return x  # x(bs, num_classes, 480, 480)
        # # -----------------------------
