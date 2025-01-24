import torch
import torch.nn as nn
import torch.nn.functional as F

class GDFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, padding=1):

        super(GDFN, self).__init__()

        # 初始卷积层，将输入图像转换为特征图
        self.initial_conv = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1)  # 输入假设为 RGB 图像

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)

        # 门机制
        self.gate_fc = nn.Linear(hidden_channels, 1)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 输出层
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)

        # 图像重建模块
        self.reconstruction = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),  # 转换回 RGB 通道
            nn.Sigmoid()  # 将像素值归一化到 [0, 1]
        )

    def forward(self, x):

        # 第一步：将原始图像转换为特征图
        x = self.initial_conv(x)  # (B, 3, H, W) -> (B, in_channels, H, W)

        # 深度可分离卷积
        x = self.depthwise_conv(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)

        # 门机制
        gate = F.adaptive_avg_pool2d(x, 1)  # 全局池化 -> (B, hidden_channels, 1, 1)
        gate = gate.view(gate.size(0), -1)   # 展平 -> (B, hidden_channels)
        gate = self.gate_fc(gate)            # 全连接层输出门值 -> (B, 1)
        gate = self.sigmoid(gate).view(-1, 1, 1, 1)  # 归一化到 [0, 1] -> (B, 1, 1, 1)

        # 根据门机制加权特征
        x = x * gate  # (B, hidden_channels, H, W)

        # 输出层
        x = self.output_conv(x)  # (B, out_channels, H, W)

        # 重建图像
        reconstructed_image = self.reconstruction(x)  # (B, 3, H, W)
        return reconstructed_image
