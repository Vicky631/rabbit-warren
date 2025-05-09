import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ========== 配置参数 ==========
GRID_SIZE = 3  # 划分的网格大小（3x3 区域）
POINTS_PER_REGION = 10  # 每个子区域最多提几个点
THRESHOLD_RATIO = 0.3  # 提点阈值比例
MIN_REGION_ACTIVATION = 0.05  # 前景最小激活阈值（跳过纯背景块）


# ========== 主要函数部分 ==========
def load_image(image_path):
    """加载图像，转换为 Tensor"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img).float().permute(2, 0, 1) / 255.0
    return img_tensor


def extract_peak_points(mask, threshold_ratio=0.3, kernel_size=3, max_points=100):
    """从前景 mask 中提取 peak points 作为正提示点"""
    assert mask.dim() == 3, "Expecting shape (1, H, W)"
    mask = torch.sigmoid(mask)

    max_val = mask.max().item()
    threshold = threshold_ratio * max_val

    pad = (kernel_size - 1) // 2
    pooled = torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    peak_mask = (pooled == mask) & (mask >= threshold)

    coords = peak_mask.nonzero(as_tuple=False)  # (N, 3)
    coords = coords[:, [2, 1]]  # (x, y)

    if coords.size(0) == 0:
        h, w = mask.shape[1:]
        coords = torch.tensor([[w // 2, h // 2]], dtype=torch.float32, device=mask.device)
    else:
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0])[:max_points]
            coords = coords[idx]

    labels = torch.ones(coords.shape[0], device=mask.device)
    return coords, labels


def extract_points_with_region_guidance(
    fg_mask, grid_size=3, points_per_region=10, threshold_ratio=0.3, min_region_activation=0.05
):
    """区域引导式提点"""
    _, H, W = fg_mask.shape
    region_H = H // grid_size
    region_W = W // grid_size

    coords_all, labels_all = [], []

    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * region_H
            y1 = (i + 1) * region_H if i < grid_size - 1 else H
            x0 = j * region_W
            x1 = (j + 1) * region_W if j < grid_size - 1 else W

            sub_mask = fg_mask[:, y0:y1, x0:x1]

            # ✅ 跳过纯背景区域
            if torch.sigmoid(sub_mask).mean() < min_region_activation:
                continue

            # 提点
            coords, labels = extract_peak_points(
                sub_mask,
                threshold_ratio=threshold_ratio,
                max_points=points_per_region,
            )
            coords[:, 0] += x0
            coords[:, 1] += y0

            coords_all.append(coords)
            labels_all.append(labels)

    if len(coords_all) == 0:
        coords_all = [torch.tensor([[W // 2, H // 2]], device=fg_mask.device)]
        labels_all = [torch.ones(1, device=fg_mask.device)]

    coords_all = torch.cat(coords_all, dim=0).unsqueeze(0)  # [1, N, 2]
    labels_all = torch.cat(labels_all, dim=0).unsqueeze(0)  # [1, N]
    return coords_all, labels_all


def visualize_points_on_image(image_tensor, points, save_path, filename="point_vis.png", point_color="red"):
    """在原图上可视化提示点，并保存图像"""
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.detach().cpu().numpy()
    else:
        img_np = image_tensor

    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.axis("off")

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    for x, y in points:
        plt.scatter(x, y, c=point_color, s=20)

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, filename)
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"✅ 可视化已保存：{out_file}")

def test_patchwise_image(image_path, model_path='/home/zy/wjj/dinol/outputs2/model_epoch46.pth',
                         output_dir='./test_outputs_patchwise', patch_size=224, stride=224):
    """
    将整张图像切成 patch，逐个预测前景，最后拼接恢复整图
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载模型 ===
    dino = load_dinov2_model()
    model = DinoSegModel(dino).to(device)

    # === 加载训练好的权重 ===
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # === 读取原图 ===
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # === 对图像做 padding，使得长宽都为 patch_size 的倍数 ===
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    pad_h = (patch_size - orig_h % patch_size) % patch_size

    padded_image = Image.new("RGB", (orig_w + pad_w, orig_h + pad_h))
    padded_image.paste(image, (0, 0))

    transform = transforms.ToTensor()
    image_tensor = transform(padded_image).unsqueeze(0).to(device)  # [1, 3, H, W]
    _, _, H, W = image_tensor.shape

    # === 分 patch，分别预测 ===
    pred_mask_full = torch.zeros((1, 1, H, W), device=device)  # 最终的前景图
    count_mask = torch.zeros((1, 1, H, W), device=device)       # 用于平均边缘重叠区域

    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image_tensor[:, :, top:top + patch_size, left:left + patch_size]  # [1, 3, patch, patch]
            with torch.no_grad():
                pred_patch = model(patch)  # [1, 1, patch, patch]
                pred_patch = torch.sigmoid(pred_patch)

            pred_mask_full[:, :, top:top + patch_size, left:left + patch_size] += pred_patch
            count_mask[:, :, top:top + patch_size, left:left + patch_size] += 1

    # === 平均重叠区域 ===
    pred_mask_full = pred_mask_full / count_mask

    # === 裁去 padding ===
    pred_mask_crop = pred_mask_full[:, :, :orig_h, :orig_w]

    # === 可视化保存 ===
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    save_image(transforms.ToTensor()(image), f"{output_dir}/{image_name}_input.png")
    save_image(pred_mask_crop[0].cpu(), f"{output_dir}/{image_name}_pred.png")

    print(f"✅ Patch-wise 前景预测完成，结果保存在 {output_dir}/{image_name}_pred.png")

# ========== 主函数：执行流程 ==========
def main(image_path):
    """主程序：比较两种提点方式"""
    save_path = "./vis_points"
    os.makedirs(save_path, exist_ok=True)

    # 加载图像
    image = load_image(image_path)
    H, W = image.shape[1:]

    # 生成模拟前景 mask（可替换为 DINOv2 输出）
    gt_points = np.array([[150, 300], [500, 400], [700, 800], [300, 700]])  # 真实点
    fg_mask = np.zeros((H, W))
    for x, y in gt_points:
        fg_mask[y, x] = 1
    fg_mask = gaussian_filter(fg_mask, sigma=10)
    fg_mask_tensor = torch.tensor(fg_mask).float().unsqueeze(0)

    # 1️⃣ 方法1：原始全图提点
    coords1, _ = extract_peak_points(fg_mask_tensor, threshold_ratio=THRESHOLD_RATIO)
    visualize_points_on_image(image, coords1[0], save_path, filename="original_method.png")

    # 2️⃣ 方法2：区域引导提点
    coords2, _ = extract_points_with_region_guidance(
        fg_mask_tensor,
        grid_size=GRID_SIZE,
        points_per_region=POINTS_PER_REGION,
        threshold_ratio=THRESHOLD_RATIO,
    )
    visualize_points_on_image(image, coords2[0], save_path, filename="region_guided_method.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="可视化两种提点方式")
    parser.add_argument("image_path", type=str, help="输入图像路径")
    args = parser.parse_args()

    main(args.image_path)
