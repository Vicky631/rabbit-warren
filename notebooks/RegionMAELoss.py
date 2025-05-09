import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance

class RegionMAELoss(nn.Module):
    def __init__(self, weight=1, bkg_weight=1, ratio=0.9):
        super(RegionMAELoss, self).__init__()
        self.weight = weight
        self.bkg_weight = bkg_weight
        self.ratio = ratio

    def forward(self, pred, label, points=None):
        """
        pred: [B, 1, H, W] - predicted density map
        label: [B, 1, H', W'] - label map with person IDs
        points: optional - dict of point locations for each person ID per sample
        """
        B, C, H, W = pred.size()
        label = F.interpolate(label, size=(H, W), mode='nearest')
        loss = torch.mean(torch.abs(pred - pred))  # initialize zero

        device = pred.device

# 修改：
        min_value, _ = torch.min(pred.view(B, -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_value, _ = torch.max(pred.view(B, -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        normalized_pred = (pred - min_value) / (max_value - min_value)

        # 或者使用torch.nn.functional.normalize，不过这里需要注意其默认是L2归一化，我们需要稍作调整
        # normalized_pred = F.normalize(pred.view(batch_size, -1), p=2, dim=1).view(batch_size, channels, height, width)
        # 这种方法不太符合最小-最大归一化，仅作参考

        # 2. 二分类
        # 选择阈值0.5，使用torch.where进行二分类
        threshold = 0.5
        binary_pred = torch.where(normalized_pred >= threshold, 1, 0)


        # 背景惩罚
        if (label == 0).sum() > 0:
            loss += self.bkg_weight * (binary_pred[label == 0]).sum()

        for b in range(B):
            unique_ids = torch.unique(label[b])
            for person_id in unique_ids:
                if person_id <= 0:
                    continue

                mask = (label[b] == person_id).float()
                region_pred = pred[b] * mask

                # Get center
                if points is None:
                    center = self.localize_head(mask[0].cpu().numpy(), ratio=self.ratio)
                else:
                    center = points[b][int(person_id.item())] / 8
                    if min(center) < 0 or max(center) > 512:
                        center = self.localize_head(mask[0].cpu().numpy(), ratio=self.ratio)

                # Cost map
                coords = np.argwhere(mask[0].cpu().numpy() > 0)[:, ::-1] / 512
                cost = distance.cdist(coords, center.reshape(1, 2) / 512, 'euclidean')
                cost = np.exp(cost / 0.6) - 1
                cost = torch.from_numpy(cost).float().to(device)

                # Transport
                transport = pred[b][0][mask[0].cpu().numpy() > 0]
                transport = transport / (transport.sum() + 1e-8)

                total_cost = (cost.reshape(-1) * transport.reshape(-1)).sum()
                mae = (torch.abs(region_pred.sum() - 1)).sum()

                loss += (mae + self.weight * total_cost)

        return loss / B

    @staticmethod
    def localize_head(mask, ratio=0.9):
        """
        Given a binary mask of a person, localize the head center.
        We assume the head center is about 15% from the top of the bounding box.
        """
        points = np.argwhere(mask > 0)
        if len(points) == 0:
            return np.array([0, 0], dtype=np.int32)

        x_min, y_min = points[:, 1].min(), points[:, 0].min()
        x_max, y_max = points[:, 1].max(), points[:, 0].max()
        w, h = x_max - x_min, y_max - y_min

        y = y_min + h * (1 - ratio)
        y = int(np.clip(y, 0, mask.shape[0] - 1))  # clamp y within image bounds

        row_y = mask[y, :]
        mid = np.argwhere(row_y > 0)
        if len(mid) == 0:
            return np.array([x_min + w // 2, y], dtype=np.int32)
        else:
            x_mid = (mid.min() + mid.max()) // 2
            return np.array([x_mid, y], dtype=np.int32)



# import torch
# import numpy as np
#
# def test_region_mae_loss():
#     from RegionMAELoss import RegionMAELoss  # 如果你已经保存成 module 的话，否则直接用类名
#
#     # 模拟输入数据
#     B, C, H, W = 2, 1, 64, 64
#     pred = torch.rand(B, C, H, W, requires_grad=True)  # 模拟预测密度图
#
#     # 构造标签（person_id 1 和 2）
#     label = torch.zeros(B, 1, H, W)
#     label[0, 0, 10:20, 10:20] = 1  # 人1
#     label[0, 0, 30:40, 30:40] = 2  # 人2
#     label[1, 0, 20:30, 20:30] = 1  # 第二张图中一个人
#
#     # 模拟点（可选）
#     points = {
#         0: {1: torch.tensor([15, 15]), 2: torch.tensor([35, 35])},
#         1: {1: torch.tensor([25, 25])}
#     }
#
#     # 实例化损失类
#     criterion = RegionMAELoss(weight=1.0, bkg_weight=0.1, ratio=0.9)
#
#     # 计算损失
#     loss = criterion(pred, label, points=points)
#
#     # 打印结果
#     print("Region MAE Loss:", loss.item())
#
#     # 验证是否可以反向传播
#     loss.backward()
#     print("Backward successful, grad of pred:", pred.grad.shape)
#
# # 调用测试函数
# test_region_mae_loss()
