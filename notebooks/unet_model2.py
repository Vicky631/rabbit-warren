import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net
    """

    def __init__(self, F_g, F_l, F_int):
        """
        F_g: Encoder 高层特征通道数
        F_l: Decoder 传递过来的跳跃连接特征通道数
        F_int: Attention 内部降维通道数
        """
        super(AttentionGate, self).__init__()

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
        """
        g: Encoder 传来的高层语义特征
        x: Decoder 跳跃连接传来的浅层特征
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # **确保 g1 和 x1 形状一致**
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # **对跳跃连接进行加权**


class UNet_model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        Attention U-Net 主体
        """
        super(UNet_model, self).__init__()

        # 编码部分
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2)  # **下采样 2×2**
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2)  # **下采样 2×2**
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=1, padding=1),  # **增加通道数**
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)  # **上采样**
        )

        # **Attention 机制**
        self.attention1 = AttentionGate(F_g=96, F_l=48, F_int=48)  # **用于 encode1**
        self.attention2 = AttentionGate(F_g=96, F_l=3, F_int=24)  # **修正通道，适配最终层**

        # 解码部分
        self.decode2 = nn.Sequential(
            nn.Conv2d(96 + 48, 96, 3, stride=1, padding=1),  # **确保 concat 结果通道一致**
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)  # **上采样**
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(96+in_channels, 64, 3, stride=1, padding=1),  # **修正 99 → 144**
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.output_layer = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

        self._init_weights()

    def forward(self, x):
        """
        前向传播
        """
        # **Encoder**
        pool1 = self.encode1(x)  # [batch, 48, 512, 512]
        pool2 = self.encode2(pool1)  # [batch, 48, 256, 256]
        upsample2 = self.encode3(pool2)  # [batch, 96, 512, 512]

        # **跳跃连接前使用 Attention**
        attn1 = self.attention1(upsample2, pool1)  # **确保通道匹配**
        concat2 = torch.cat((upsample2, attn1), dim=1)  # **[batch, 144, 512, 512]**

        upsample1 = self.decode2(concat2)  # [batch, 96, 1024, 1024]

        # **跳跃连接前使用 Attention**
        pool2_up = F.interpolate(pool2, size=upsample1.shape[2:], mode='bilinear', align_corners=True)  # **确保 pool2 形状匹配**
        attn2 = self.attention2(upsample1, x)  # **修正：直接使用 `x`**
        concat1 = torch.cat((upsample1, attn2), dim=1)  # **[batch, 144, 1024, 1024]**

        upsample0 = self.decode1(concat1)  # [batch, 32, 1024, 1024]
        output = self.output_layer(upsample0)  # [batch, 3, 1024, 1024]

        return output

    def _init_weights(self):
        """
        He 初始化所有 Conv2d 和 ConvTranspose2d 层
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
