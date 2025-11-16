import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import os


# 自注意力机制：用于提取全局上下文信息
class SelfAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query_fc = nn.Linear(in_dim, hidden_dim)  # 查询（Q）
        self.key_fc = nn.Linear(in_dim, hidden_dim)  # 键（K）
        self.value_fc = nn.Linear(in_dim, hidden_dim)  # 值（V）
        self.attn_fc = nn.Linear(hidden_dim, in_dim)  # 输出（Attention 输出）

    def forward(self, x):
        """
        x: 输入特征 [B, N, F] -> [batch_size, seq_len, feature_dim]
        """
        query = self.query_fc(x)  # [B, N, hidden_dim]
        key = self.key_fc(x)  # [B, N, hidden_dim]
        value = self.value_fc(x)  # [B, N, hidden_dim]

        # 计算注意力得分
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / query.size(-1) ** 0.5  # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]

        # 计算加权值
        attn_output = torch.matmul(attn_weights, value)  # [B, N, hidden_dim]

        # 输出
        output = self.attn_fc(attn_output)  # [B, N, F]
        return output


# 生成二维位置编码
class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, embedding_dim):
        super(PositionalEncoding2D, self).__init__()

        # 生成位置编码
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim

        # 初始化位置编码矩阵
        pe = torch.zeros(height, width, embedding_dim)
        for i in range(height):
            for j in range(width):
                for k in range(0, embedding_dim, 2):
                    pe[i, j, k] = math.sin(i / 10000 ** (k / embedding_dim))  # 使用sin生成编码
                    pe[i, j, k + 1] = math.cos(j / 10000 ** ((k + 1) / embedding_dim))  # 使用cos生成编码
        self.pe = pe.permute(2, 0, 1).unsqueeze(0)  # 转置并加一个batch维度 [1, embedding_dim, height, width]

    def forward(self, x):
        return x + self.pe.to(x.device)  # 将位置编码加到输入特征上


# 上下文感知分类器：结合自注意力机制与局部特征
class ContextAwareClassifier(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, output_dim=1):
        super(ContextAwareClassifier, self).__init__()
        self.self_attention = SelfAttention(in_dim=in_dim, hidden_dim=hidden_dim)
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        x: 输入特征 [B, C, H, W] -> [batch_size, channels, height, width]
        """
        # 先将输入特征展平成 [B, N, F]，适用于自注意力计算
        B, C, H, W = x.shape
        x_flat = x.view(B, -1, C)  # [B, N, C]

        # 应用自注意力
        attn_output = self.self_attention(x_flat)  # [B, N, C]

        # 重新将自注意力输出形状恢复成 [B, C, H, W]
        attn_output = attn_output.view(B, C, H, W)

        # 最后的卷积层进行分类，输出分割掩码
        mask = self.fc(attn_output)  # [B, 1, H, W]
        return mask


# 模型：DINOv2特征提取器和上下文感知分类器结合
class DinoSegModelWithContextAwareClassifier(nn.Module):
    def __init__(self, dino_model, patch_size=14, feature_dim=2048, hidden_dim=512, height=224, width=224):
        super(DinoSegModelWithContextAwareClassifier, self).__init__()
        self.dino = dino_model
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 上下文感知分类器
        self.context_aware_classifier = ContextAwareClassifier(in_dim=feature_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        """
        x: 输入图像特征 [B, C, H, W]
        """
        with torch.no_grad():
            feats_all = self.dino.get_intermediate_layers(x, n=2)  # 提取最后两层特征
            feat1 = feats_all[0]  # [B, N, C]
            feat2 = feats_all[1]  # [B, N, C]

        # 将特征转换为适合卷积的形状
        B, N, C = feat1.shape
        H = W = int(N ** 0.5)
        feat1 = feat1.transpose(1, 2).view(B, C, H, W)
        feat2 = feat2.transpose(1, 2).view(B, C, H, W)

        # 特征拼接
        fused_feat = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]

        # 上下文感知分类器处理
        mask = self.context_aware_classifier(fused_feat)

        # 对输出进行插值，恢复到原始尺寸
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return mask


# 数据集处理类
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


# 测试单张图像
def test_single_image(image_path, model_path='/home/zy/wjj/dinol/outputs3/model_epoch46.pth',
                      output_dir='./test_outputs'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载模型 ===
    dino = load_dinov2_model()
    model = DinoSegModelWithContextAwareClassifier(dino).to(device)

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


# 训练函数
def train_dino_seg_with_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 数据路径 ===
    image_dir = "/home/zy/wjj/dataset/UCF-QNRF/QNRF/images"
    mask_dir = "/home/zy/wjj/dataset/UCF-QNRF/QNRF/labels"
    vis_dir = "./outputs_with_attention"
    save_model_dir = "./outputs_with_attention"

    # === 数据加载 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageMaskDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # === 模型初始化 ===
    dino = load_dinov2_model()
    model = DinoSegModelWithContextAwareClassifier(dino).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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


# 主程序入口
if __name__ == "__main__":
    # 如果要训练模型，取消下面的注释并运行训练过程
    # train_dino_seg_with_attention()

    # 测试单张图像
    test_single_image("/home/zy/wjj/dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg")

    # 如果需要检查 DINOv2 是否支持 get_intermediate_layers
    # 加载 DINOv2 模型
    dino = load_dinov2_model()

    # 创建一个 dummy_input，确保输入的形状和模型所需的一致
    dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 假设使用 CUDA（GPU）

    # 测试 DINOv2 是否支持 get_intermediate_layers
    try:
        out = dino.get_intermediate_layers(dummy_input, n=2)  # 提取两个中间层的特征
        print(f"✔ get_intermediate_layers works. Got {len(out)} layers")
    except AttributeError as e:
        print("❌ Your DINOv2 model doesn't support get_intermediate_layers.")
        print(e)
