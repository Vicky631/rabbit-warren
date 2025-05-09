import torch
import torch.nn as nn


# CBAM 注意力模块
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        # 通道注意力 (Channel Attention)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 空间注意力 (Spatial Attention)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # ====== 通道注意力 (Channel Attention) ======
        avg_out = self.fc2(torch.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(torch.relu(self.fc1(self.max_pool(x))))
        scale = self.sigmoid(avg_out + max_out)
        x = x * scale

        # ====== 空间注意力 (Spatial Attention) ======
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        concat = torch.cat([avg_pool, max_pool], dim=1)  # 拼接
        scale = self.sigmoid(self.spatial_conv(concat))
        x = x * scale

        return x


# U-Net + CBAM 结构
class UNet_model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_model, self).__init__()

        # 编码层
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2)
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2)
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )

        # CBAM 注意力
        self.cbam1 = CBAMBlock(48)
        self.cbam2 = CBAMBlock(96)
        self.cbam3 = CBAMBlock(96 + in_channels)

        # 解码层
        self.decode2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # 输出层
        self.output_layer = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # 编码过程
        pool1 = self.encode1(x)
        pool1 = self.cbam1(pool1)  # 添加 CBAM 注意力

        pool2 = self.encode2(pool1)

        upsample2 = self.encode3(pool2)

        # 跳跃连接 + CBAM
        concat2 = torch.cat((upsample2, pool1), dim=1)
        concat2 = self.cbam2(concat2)  # 添加 CBAM 注意力
        upsample1 = self.decode2(concat2)

        concat1 = torch.cat((upsample1, x), dim=1)
        concat1 = self.cbam3(concat1)  # 添加 CBAM 注意力
        umsample0 = self.decode1(concat1)

        output = self.output_layer(umsample0)

        return output
