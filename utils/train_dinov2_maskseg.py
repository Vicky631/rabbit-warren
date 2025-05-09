import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import utils



# ========= 1. 加载 DINOv2 ==========
def load_dinov2_model():
    print("🔧 Using DINOv2 as backbone")

    dino_repo = "/home/zy/wjj/Prompt_sam_localization/dinov2"
    dino_model_name = "dinov2_vitl14"
    dino_ckpt_path = "/home/zy/wjj/CrowdSAM-main/weights/dinov2_vitl14_pretrain.pth"

    # 加载模型结构
    dino_model = torch.hub.load(dino_repo, dino_model_name, source='local', pretrained=False).cuda()

    # 加载预训练权重
    dino_model.load_state_dict(torch.load(dino_ckpt_path))
    return dino_model



# ========= 2. 分类器 ==========
# ========= 2. 上下文感知分类器 ==========

class SpatialAwareClassifier(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):  # x: [B, C, H, W]
        return self.conv(x)


# ========= 3. 数据集 ==========
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # 读取灰度 mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 将 mask 转换为二值 (0, 1)
        mask = (mask > 0.5).float()

        return image, mask


# ========= 3. 多尺度融合的主模型 ==========
class DinoSegModel(nn.Module):
    def __init__(self, dino_model, patch_size=14):
        super().__init__()
        self.dino = dino_model
        self.patch_size = patch_size
        self.classifier = SpatialAwareClassifier(in_dim=2048, hidden_dim=512)

    def forward(self, x):
        with torch.no_grad():
            feats_all = self.dino.get_intermediate_layers(x, n=2)  # 最后一层和倒数第二层
            feat1 = feats_all[0]  # [B, N, C]
            feat2 = feats_all[1]  # [B, N, C]

        # reshape 成 feature map
        B, N, C = feat1.shape
        H = W = int(N ** 0.5)
        feat1 = feat1.transpose(1, 2).view(B, C, H, W)
        feat2 = feat2.transpose(1, 2).view(B, C, H, W)

        # 多尺度融合：直接拼接
        fused_feat = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]

        # 分类器
        mask = self.classifier(fused_feat)
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return mask


# ========= 5. 可视化保存 ==========
def visualize(image_tensor, pred_mask, gt_mask, save_path, idx):
    os.makedirs(save_path, exist_ok=True)
    image = image_tensor[0].cpu()
    pred = torch.sigmoid(pred_mask[0]).cpu()
    gt = gt_mask[0].cpu()

    save_image(image, f"{save_path}/img_{idx}.png")
    save_image(pred, f"{save_path}/pred_{idx}.png")
    save_image(gt, f"{save_path}/gt_{idx}.png")


# ========= 6. 主训练流程 ==========
def train_dino_seg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 路径 ===
    image_dir = "/home/zy/wjj/Prompt_sam_localization/dataset/UCF-QNRF/QNRF/images"
    mask_dir = "/home/zy/wjj/Prompt_sam_localization/dataset/UCF-QNRF/QNRF/labels"
    vis_dir = "./outputs2"
    save_model_dir="./outputs2"

    # === 数据加载 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageMaskDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # === 模型初始化 ===
    dino = load_dinov2_model()
    model = DinoSegModel(dino).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    epoch_loss = 0.0
    count = 0

    # === 训练 ===
    for epoch in range(50):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (img, mask) in enumerate(pbar):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

            if i % 10 == 0:
                visualize(img, pred, mask, vis_dir, idx=f"e{epoch}_i{i}")
        avg_loss = epoch_loss / count
        print(f"📉 Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

    # ✅ 保存模型
        save_path = os.path.join(save_model_dir, f"model_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"✅ Model saved to {save_path}")

def test_single_image(image_path, model_path='/home/zy/wjj/dinol/outputs3/model_epoch46.pth', output_dir='./test_outputs'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载模型 ===
    dino = load_dinov2_model()
    model = DinoSegModel(dino).to(device)

    # === 加载训练好的权重 ===
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # === 加载并预处理图片 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(image_tensor)

    # === 可视化保存 ===
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(image_tensor[0].cpu(), f"{output_dir}/{image_name}_input.png")
    save_image(torch.sigmoid(pred_mask[0].cpu()), f"{output_dir}/{image_name}_pred.png")

    print(f"✅ Done. Saved to {output_dir}/{image_name}_pred.png")


def predict_patchwise_mask_batched(image_tensor, model, patch_size=196, stride=196):
    """
    输入：带 Batch 的图像张量 [B, 3, H, W]
    输出：预测出的前景 mask，形状为 [B, 1, H, W]

    参数：
        image_tensor: [B, 3, H, W] 的图像张量（不需要做 padding）
        model: DinoSegModel，已加载权重、.eval() 且放到对应 device
        patch_size: patch 尺寸
        stride: 滑动窗口步长
    """
    device = image_tensor.device
    B, _, H, W = image_tensor.shape

    # === 对每张图分别 padding，确保 H/W 为 patch_size 的整数倍 ===
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    x_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')  # [B, 3, H_pad, W_pad]
    _, _, H_pad, W_pad = x_padded.shape

    pred_mask_full = torch.zeros((B, 1, H_pad, W_pad), device=device)
    count_mask = torch.zeros((B, 1, H_pad, W_pad), device=device)

    # === 滑窗处理所有 batch 样本 ===
    for top in range(0, H_pad - patch_size + 1, stride):
        for left in range(0, W_pad - patch_size + 1, stride):
            patch = x_padded[:, :, top:top + patch_size, left:left + patch_size]  # [B, 3, patch, patch]
            with torch.no_grad():
                pred_patch = model(patch)  # [B, 1, patch, patch]
                pred_patch = torch.sigmoid(pred_patch)

            pred_mask_full[:, :, top:top + patch_size, left:left + patch_size] += pred_patch
            count_mask[:, :, top:top + patch_size, left:left + patch_size] += 1

    # === 平均重叠区域 ===
    pred_mask_full = pred_mask_full / count_mask

    # === 去掉 padding，恢复原始尺寸 ===
    pred_mask_crop = pred_mask_full[:, :, :H, :W]  # [B, 1, H, W]

    return pred_mask_crop



# 用于加载上述的模型
def load_trained_dino_seg_model(ckpt_path):
    # 加载 DINOv2（结构）
    dino = load_dinov2_model()  # 你已有的函数，会返回初始化好的 DINO 模型

    # 构建 DinoSegModel（结构）
    model = DinoSegModel(dino)

    # 加载你训练的权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ DinoSegModel loaded from {ckpt_path}")
    return model



if __name__ == "__main__":
    # train_dino_seg()
    test_single_image("/home/zy/wjj/dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg")
    # dino = load_dinov2_model()
    #
    # dummy_input = torch.randn(1, 3, 224, 224).cuda()
    # try:
    #     out = dino.get_intermediate_layers(dummy_input, n=2)
    #     print(f"✔ get_intermediate_layers works. Got {len(out)} layers")
    # except AttributeError as e:
    #     print("❌ Your DINOv2 model doesn't support get_intermediate_layers.")
    #     print(e)
