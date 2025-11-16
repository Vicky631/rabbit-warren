import argparse

# from notebooks.RegionMAELoss import RegionMAELoss
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch
import logging
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss
from unet_model import UNet_model
from tqdm import tqdm
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import scipy.io
from datetime import datetime
from GDFN import GDFN
import sys
from utils.train_dinov2_maskseg import DinoSegModel, load_trained_dino_seg_model, predict_patchwise_mask_batched
from utils import utils
from utils.utils import PatchAlignedResizer, update_and_save_model, normalize_image_to_uint8
from utils.wjjimage import TensorVisualizer
from utils.utils import save_image

sys.path.append('/home/zy/wjj/Prompt_sam_localization')
from segment_anything.build_sam_adapter import sam_model_registry
# from segment_anything.build_sam_jj import sam_model_registry_jj
from segment_anything.predictor_jj import SamPredictor

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# 设置随机种子，确保实验可重复
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file="training.log"):
    """
    Set up a logger to output messages to a file and the console.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("saml")  # 创建自定义 logger
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


import torch.nn.functional as F
import numpy as np
import torch


def extract_peak_points(mask, threshold_ratio=0.3, kernel_size=3, max_points=100):
    """
    从前景 mask 中提取局部极大值位置作为正提示点，用于 SAM。
    支持从 logits（未 sigmoid）中提取。
    """
    assert mask.dim() == 3, "Expecting shape (1, H, W)"

    mask = mask.clone().detach()

    # ✅ 显式做 sigmoid 归一化（避免负值造成误提点）
    mask = torch.sigmoid(mask)

    max_val = mask.max().item()
    threshold = threshold_ratio * max_val

    # MaxPool2d to find local peaks
    pad = (kernel_size - 1) // 2
    pooled = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)

    peak_mask = (pooled == mask) & (mask >= threshold)

    coords = peak_mask.nonzero(as_tuple=False)  # (N, 3)
    coords = coords[:, [2, 1]]  # (x, y)

    if coords.size(0) == 0:
        h, w = mask.shape[1:]
        coords = torch.tensor([[w // 2, h // 2]], dtype=torch.float32, device=mask.device)
    else:
        coords = coords.float()
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0])[:max_points]
            coords = coords[idx]

    labels = torch.ones(coords.shape[0], device=mask.device)
    return coords, labels

    # SAM 模型


class Sam_model(nn.Module):
    def __init__(self, args, model_type, sam_checkpoint):
        super(Sam_model, self).__init__()
        self.sam = sam_model_registry[model_type](args, checkpoint=sam_checkpoint).cuda()
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

    def forward(self, x, points, text=None):
        """
        Forward method to include text as an additional prompt.

        Args:
            x (torch.Tensor): Input image.
            points (torch.Tensor): Input points.
            text (torch.Tensor, optional): Text embeddings. Default is None.

        Returns:
            torch.Tensor: Predicted masks.
        """
        # Encode the input image
        image = self.sam.image_encoder(x)

        # Generate sparse and dense embeddings, now including text
        try:
            se, de = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
                text=text,  # Pass text embeddings
            )
        except Exception:
            se, de = self.sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

        # Decode the mask
        pred, _ = self.sam.mask_decoder(
            image_embeddings=image,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
        )
        return pred


class Model(nn.Module):
    def __init__(self, args, model_type, sam_checkpoint):
        super().__init__()
        # Load SAM
        self.sam = Sam_model(args, model_type=model_type, sam_checkpoint=args.sam_ckpt)
        # if args.fine_tuning_configuration:
        #     sam_no_freeze_block = [f"blocks.{idx}." for idx, i in enumerate(args.fine_tuning_configuration) if i == 1]
        #
        # for n, value in self.sam.named_parameters():
        #     if "Adapter" not in n:
        #         if args.fine_tuning_configuration:
        #             if 1 not in [1 for val in sam_no_freeze_block if val in n]:
        #                 value.requires_grad = False
        #                 # value.requires_grad =True
        #         else:
        #             value.requires_grad = True
        for name, param in self.sam.named_parameters():
            if 'Adapter' in name:
                # print(f"[Trainable] {name}")
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Freeze SAM except mask decoder
        # for name, param in self.sam.named_parameters():
        #     param.requires_grad = 'mask_decoder' in name

        # Load frozen DINOv2 + classifier
        self.dino_seg = load_trained_dino_seg_model(
            "/home/zy/wjj/dinol/outputs2/model_epoch44.pth"
        )
        self.dino_seg.eval()
        for param in self.dino_seg.parameters():
            param.requires_grad = False

        # Add resizer
        self.resizer = PatchAlignedResizer(patch_size=14)

    def forward(self, x, points=None, text=None):
        """
        x: 输入图像 (B, 3, H, W)
        返回：
            x: 原图（可视化或增强用）
            pred_mask: SAM输出mask（用于计算loss）
        """
        device = next(self.parameters()).device
        x = x.to(device)

        B, _, H, W = x.shape

        # 1. Resize for DINOv2 (patch-aligned)
        x_dino = self.resizer.resize_input(x)  # [B, 3, H', W']

        # 2. DINOv2 前景预测
        with torch.no_grad():
            # fg_mask_logit = self.dino_seg(x_dino)
            fg_mask_logit = self.dino_seg(x_dino)  # [B, 1, H', W']

            fg_mask = self.resizer.restore_output(fg_mask_logit)  # [B, 1, H, W]

            coords_list, labels_list = [], []
            for i in range(B):
                coord, label = extract_peak_points(
                    fg_mask[i]
                )
                coords_list.append(coord)
                labels_list.append(label)
        # print(self.sam)

        # 3. 喂给 SAM
        image_embeddings = self.sam.image_encoder(x)  # 用原图喂入 SAM

        pred_masks = []
        for i in range(B):
            coord = coords_list[i].unsqueeze(0).to(x.device)  # [1, N, 2]
            label = labels_list[i].unsqueeze(0).to(x.device)  # [1, N]

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(coord, label),
                boxes=None,
                masks=None,
                # text_embeds=None,
            )

            pred, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_masks.append(pred)

        pred_masks = torch.cat(pred_masks, dim=0)  # [B, 1, H_sam, W_sam]
        # pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False)

        return x, pred_masks


# 可视化保存函数
def save_visualization(denoised_img, pred, label, save_path, filenames, idx, epoch):
    os.makedirs(save_path, exist_ok=True)

    # 获取当前批次的文件名
    batch_filenames = filenames[idx * pred.size(0):(idx + 1) * pred.size(0)]

    def save_image(image, path):

        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 1)
        elif np.issubdtype(image.dtype, np.integer):
            image = np.clip(image, 0, 255).astype(np.uint8)

        """保存图像到指定路径"""
        if image.ndim == 4:  # 批量图像 (N, C, H, W)
            for i in range(image.shape[0]):
                single_image = image[i].transpose(1, 2, 0) if image.shape[1] == 3 else image[i, 0]
                plt.imsave(f"{path}_sample{i + 1}.png", single_image, cmap='gray' if single_image.ndim == 2 else None)
        elif image.ndim == 3 and image.shape[0] == 3:  # 单张 RGB 图像 (C, H, W)
            image = image.transpose(1, 2, 0)  # 转换为 (H, W, C)
            plt.imsave(path, image)
        elif image.ndim == 3 and image.shape[0] == 1:  # 单通道灰度图像 (1, H, W)
            image = image.squeeze(0)  # 转换为 (H, W)
            plt.imsave(path, image, cmap='gray')
        elif image.ndim == 2:  # 2D 图像 (H, W)
            plt.imsave(path, image, cmap='gray')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    # 处理每种图像类型并保存
    for i, filename in enumerate(batch_filenames):
        base_name = os.path.splitext(filename)[0]  # 去除扩展名

        # 保存 denoised_img
        denoised_img_np = denoised_img[i].detach().cpu().numpy()
        save_image(
            denoised_img_np,
            os.path.join(save_path, f"epoch{epoch}_{base_name}_denoised.png")
        )

        # 保存 pred
        pred_np = pred[i].detach().cpu().numpy()  # Normalize the mask

        save_image(
            pred_np,
            os.path.join(save_path, f"epoch{epoch}_{base_name}_pred.png")
        )

        # 保存 label
        label_np = label[i].detach().cpu().numpy()
        base_name = str(base_name) if isinstance(base_name, np.ndarray) else base_name

        save_image(
            label_np,
            os.path.join(save_path, f"epoch{epoch}_{base_name}_label.png")
        )


def visualize_and_save(density_map, coordinates, save_path, filename):
    """
    在密度图上可视化定位点，并保存结果。

    Args:
        density_map (numpy.ndarray): 密度图，形状为 (H, W)。
        coordinates (list): 点的坐标列表，每个点为 (x, y) 格式。
        save_path (str): 保存可视化结果的路径。
        filename (str): 保存文件的名称。
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(density_map, cmap="jet")  # 显示密度图
    plt.colorbar(label="Density Value")  # 添加颜色条
    for x, y in coordinates:
        plt.scatter(x, y, c="red", s=10)  # 在定位点处绘制红点
    plt.title("Density Map with Local Maxima")
    plt.axis("off")

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, bbox_inches="tight")  # 保存图片
    plt.close()


# 保存最优模型
def save_best_model(model, save_path, best_loss, val_loss, epoch, optimizer=None, model_name="best_model",
                    max_models=1):
    """
    Save the best model based on the validation loss, allowing to keep up to `max_models` best models.

    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the model.
        best_loss (float): Current best loss, updated after each save.
        val_loss (float): Validation loss for the current epoch.
        epoch (int): Current epoch number.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save (if provided).
        model_name (str, optional): The name for the saved model file.
        max_models (int, optional): Maximum number of models to keep. Default is 1.

    Returns:
        float: Updated best_loss.
    """

    # If the current validation loss is better (smaller), save the model
    if val_loss < best_loss:
        best_loss = val_loss

        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # 删除旧模型，确保有足够的磁盘空间
        if max_models > 0:
            delete_old_models(save_path, model_name, max_models - 1)

        # Get current timestamp (format: MMDD_HHMM)
        timestamp = datetime.now().strftime('%m%d_%H%M')

        # Save model state_dict and optimizer (optional)
        model_save_path = os.path.join(save_path, f"{model_name}_epoch{epoch}_{timestamp}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save the model checkpoint
        torch.save(checkpoint, model_save_path)
        print(
            f"Saved best model at epoch {epoch} with validation loss: {val_loss:.4f} (best: {best_loss:.4f}) at {timestamp}")

        # Optional: log the model saving information
        logger.info(f"Epoch {epoch}: Best loss : {best_loss:.4f}. ")

    return best_loss


def delete_old_models(save_path, model_name, max_models):
    """
    Delete older saved models, ensuring only up to `max_models` remain before saving the new one.

    Args:
        save_path (str): Path where models are saved.
        model_name (str): Base name of the model files.
        max_models (int): Maximum number of models to keep.
    """
    # List all saved model files that match the base name
    model_files = [f for f in os.listdir(save_path) if f.startswith(model_name) and f.endswith('.pth')]

    # Sort files by modification time (oldest first)
    model_files = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(save_path, x)))

    # If the number of files exceeds `max_models`, remove the oldest ones
    for file_to_remove in model_files[:len(model_files) - max_models]:
        os.remove(os.path.join(save_path, file_to_remove))
        print(f"Removed old model: {file_to_remove}")
        logging.info(f"Removed old model: {file_to_remove}")


# Dice Coefficient 计算类
class DiceCoeff(torch.autograd.Function):
    """Dice coefficient for individual examples."""

    @staticmethod
    def forward(ctx, input, target):
        eps = 1e-6
        ctx.save_for_backward(input, target)
        inter = torch.dot(input.view(-1), target.view(-1))
        union = input.sum() + target.sum() + eps
        return (2 * inter + eps) / union

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        grad_input = grad_target = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * (2 * target - 2 * input) / (input.sum() + target.sum() + 1e-6)
        if ctx.needs_input_grad[1]:
            grad_target = grad_output * (2 * input - 2 * target) / (input.sum() + target.sum() + 1e-6)
        return grad_input, grad_target


# Batch-wise Dice coefficient
def dice_coeff(input, target):
    """
    Compute Dice coefficient for a batch of samples.
    Args:
        input (torch.Tensor): Predicted masks (batch_size, 1, H, W).
        target (torch.Tensor): Ground truth masks (batch_size, 1, H, W).
    Returns:
        float: Average Dice coefficient for the batch.
    """
    if input.is_cuda:
        s = torch.FloatTensor(1).to(input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, (inp, tar) in enumerate(zip(input, target)):
        s += DiceCoeff.apply(inp, tar)

    return s / (i + 1)


# Intersection over Union (IoU) 计算函数
def cal_iou(outputs, labels):
    """
    Calculate IoU (Intersection over Union).
    Args:
        outputs (numpy.ndarray): Predicted masks (batch_size, H, W).
        labels (numpy.ndarray): Ground truth masks (batch_size, H, W).
    Returns:
        float: Mean IoU over the batch.
    """
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()


# 评估分割性能的主函数
def eval_seg(pred, true_mask_p, thresholds):
    """
    Evaluate segmentation performance using IoU and Dice coefficient.
    Args:
        pred (torch.Tensor): Predicted masks (batch_size, 1, H, W).
        true_mask_p (torch.Tensor): Ground truth masks (batch_size, 1, H, W).
        thresholds (tuple): List of thresholds to evaluate.
    Returns:
        tuple: Mean IoU and Dice coefficient across all thresholds.
    """
    eiou, edice = 0, 0
    for th in thresholds:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()

        # Convert to NumPy for IoU calculation
        disc_pred = vpred.cpu().numpy().astype('int32')
        disc_mask = gt_vmask_p.cpu().numpy().astype('int32')

        # Calculate IoU
        eiou += cal_iou(disc_pred, disc_mask)

        # Calculate Dice coefficient
        edice += dice_coeff(vpred, gt_vmask_p).item()

    return eiou / len(thresholds), edice / len(thresholds)


def extract_gt_points(label, output_path, min_area=1):
    """
    从二值图像中提取白色方块的中心点，并在图像上标记中心点保存到指定路径。

    Args:
        label (torch.Tensor or np.ndarray): 输入二值图像。
        output_path (str): 输出图像保存路径。
        min_area (int): 忽略的最小区域大小（像素个数）。

    Returns:
        list: 中心点的坐标列表 [(x1, y1), (x2, y2), ...]。
    """
    # 如果输入是 PyTorch 张量，转换为 NumPy 数组
    if isinstance(label, torch.Tensor):
        if label.is_cuda:
            label = label.cpu().numpy()  # 移动到 CPU 并转换为 NumPy
        else:
            label = label.numpy()

    # 移除多余维度（如果有）
    label = np.squeeze(label, axis=0)

    # 将输入图像二值化，确保像素值为 0 或 255
    binary_image = np.where(label > 0, 255, 0).astype(np.uint8)

    # 使用形态学操作（闭运算）消除小的断点和噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # 使用连通组件分析找到每个白色方块
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    # 提取中心点
    true_coordinates = []
    for i in range(1, num_labels):  # 跳过背景
        area = stats[i, cv2.CC_STAT_AREA]  # 获取区域面积
        if area >= min_area:  # 忽略小区域
            x, y = int(centroids[i][0]), int(centroids[i][1])
            true_coordinates.append((x, y))

    # 创建标记中心点的图像
    marked_image = np.zeros_like(binary_image, dtype=np.uint8)
    for x, y in true_coordinates:
        marked_image[y, x] = 255  # 在中心点标记一个白点

    # 保存生成的标记图像
    cv2.imwrite(output_path, marked_image)
    print(f"Marked image saved to: {output_path}")

    return true_coordinates


from scipy import ndimage
#
# def extract_coordinates_and_visualize_via_watershed(pred_density_np, save_path, filename, min_area=1, local_max_size=9):
#     """
#     使用增强分水岭方法从预测图中提取角点坐标并可视化。
#
#     返回图像为 (3, H, W) 格式，兼容你自己的 save_image 函数。
#     """
#     # Step 1: Tensor → NumPy
#     if isinstance(pred_density_np, torch.Tensor):
#         pred_density_np = pred_density_np.detach().cpu().squeeze().numpy()
#
#     assert pred_density_np.ndim == 2, "输入必须是 (H, W) 灰度图"
#
#     # Step 2: 归一化并转 uint8 图像
#     img = normalize_image_to_uint8(pred_density_np)
#
#
#     # Step 3: 二值图 & 彩图
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     # Step 4: 腐蚀 & 距离变换
#     kernel = np.ones((2, 2), np.uint8)
#     eroded = cv2.erode(binary, kernel, iterations=1)
#     dist_transform = cv2.distanceTransform(eroded.astype(np.uint8), cv2.DIST_L2, 5)
#
#     # Step 5: 局部最大值 marker
#     local_max = ndimage.maximum_filter(dist_transform, size=local_max_size) == dist_transform
#     markers, _ = ndimage.label(local_max)
#     markers = markers.astype(np.int32)
#
#     # Step 6: 分水岭
#     color_img_ws = color_img.copy()
#     cv2.watershed(color_img_ws, markers)
#
#     # Step 7: 提取中心点并画在图上
#     coordinates = []
#     for label in np.unique(markers):
#         if label <= 0:
#             continue
#         mask = np.uint8(markers == label)
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in cnts:
#             area = cv2.contourArea(cnt)
#             if area < min_area:
#                 continue
#             M = cv2.moments(cnt)
#             if M["m00"] > 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 if binary[cy, cx] == 0:
#                     continue
#                 coordinates.append((cx, cy))
#                 cv2.circle(color_img_ws, (cx, cy), 2, (0, 255, 0), -1)
#
#     # Step 8: 格式转换 HWC → CHW
#     chw_image = np.transpose(color_img_ws, (2, 0, 1))  # (H, W, 3) → (3, H, W)
#
#     # Step 9: 保存图像
#      # 你原项目里的函数
#     save_image(chw_image, os.path.join(save_path, filename))
#
#     return coordinates

import os
import cv2
import numpy as np
import torch
from scipy import ndimage


def extract_coordinates_and_visualize_via_watershed(pred_density_np, save_path, filename, min_area=1, local_max_size=9):
    """
    使用增强分水岭方法 + 方形拟合（补充点） 提取中心点并可视化，支持保存为 CHW 图像。
    """
    # Step 1: Tensor → NumPy
    if isinstance(pred_density_np, torch.Tensor):
        pred_density_np = pred_density_np.detach().cpu().squeeze().numpy()

    assert pred_density_np.ndim == 2, "输入必须是 (H, W) 灰度图"

    # Step 2: 归一化并转 uint8 图像
    img = normalize_image_to_uint8(pred_density_np)

    # Step 3: 二值图 & 彩图
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Step 4: 腐蚀 & 距离变换
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(eroded.astype(np.uint8), cv2.DIST_L2, 5)

    # Step 5: 局部最大值 marker
    local_max = ndimage.maximum_filter(dist_transform, size=local_max_size) == dist_transform
    markers, _ = ndimage.label(local_max)
    markers = markers.astype(np.int32)

    # Step 6: 分水岭处理
    color_img_ws = color_img.copy()
    cv2.watershed(color_img_ws, markers)

    # Step 7: 方法 A（分水岭提点）
    coordinates_a = []
    for label in np.unique(markers):
        if label <= 0:
            continue
        mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if binary[cy, cx] == 0:
                    continue
                coordinates_a.append((cx, cy))
                cv2.circle(color_img_ws, (cx, cy), 2, (0, 255, 0), -1)  # 方法 A 绿色点

    # Step 8: 方法 B（方形拟合补充点）
    contours_b, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    supplemental_points = []
    for cnt in contours_b:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        center = rect[0]
        cx_b, cy_b = int(center[0]), int(center[1])

        # 距离判断：和所有方法 A 点比较，寻找最近距离
        dists = [np.linalg.norm(np.array([cx_b, cy_b]) - np.array(pt)) for pt in coordinates_a]
        if len(dists) == 0:
            continue
        min_dist = min(dists)
        if 3 < min_dist < 15:  # 满足条件，补充此点
            supplemental_points.append((cx_b, cy_b))
            coordinates_a.append((cx_b, cy_b))
            cv2.circle(color_img_ws, (cx_b, cy_b), 2, (0, 0, 255), -1)  # 方法 B 红色补点

    # Step 9: 转为 CHW 并保存图像
    chw_image = np.transpose(color_img_ws, (2, 0, 1))  # (H, W, 3) → (3, H, W)
    save_image(chw_image, os.path.join(save_path, filename))

    return coordinates_a


def generate_pointmap_and_save(kpoint, coordinates, save_path, filename, rate=1):
    """
    生成点坐标地图并保存。

    Args:
        kpoint (numpy.ndarray): 局部极大值点的位置图，形状为 (H, W)。
        coordinates (list): 提取的点的坐标列表，每个点为 (x, y) 格式。
        save_path (str): 保存地图的路径。
        filename (str): 保存的文件名称。
        rate (int): 放缩比例，用于调整可视化大小。

    Returns:
        numpy.ndarray: 点坐标地图图像。
    """
    # 初始化地图背景
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255

    # 在地图上绘制点
    for x, y in coordinates:
        cv2.circle(point_map, (int(x * rate), int(y * rate)), 2, (0, 0, 255), -1)  # 红色点

    # 保存地图
    os.makedirs(save_path, exist_ok=True)
    # 保存点图
    save_file = os.path.join(save_path, filename.replace(".jpg", "_kpoint.jpg"))
    cv2.imwrite(save_file, point_map)
    # 保存源图
    # save_file = os.path.join(save_path, filename.replace(".png", "_denoised.jpg"))
    # cv2.imwrite(save_file, denoise_image)
    # print(f" saved to: {save_file}_kpoint")

    return coordinates


def calculate_f1_precision_recall(pred_coordinates, true_coordinates, match_threshold=5):
    """
    计算 F1 分数、Precision 和 Recall。

    Args:
        pred_coordinates (list of tuple): 预测的坐标点列表，每个点为 (x, y) 格式。
        true_coordinates (list of tuple): 真实的坐标点列表，每个点为 (x, y) 格式。
        match_threshold (int): 匹配距离阈值，如果预测点与真实点的欧氏距离小于等于该值，则认为匹配。

    Returns:
        tuple: F1 分数、Precision、Recall。
    """
    from scipy.spatial.distance import cdist

    # 将坐标列表转换为 numpy 数组
    pred_coordinates = np.array(pred_coordinates)
    true_coordinates = np.array(true_coordinates)

    # 如果没有预测点或真实点
    if len(pred_coordinates) == 0:
        precision = 0 if len(true_coordinates) > 0 else 1
        recall = 0
        f1 = 0
        return f1, precision, recall
    if len(true_coordinates) == 0:
        precision = 0
        recall = 0 if len(pred_coordinates) > 0 else 1
        f1 = 0
        return f1, precision, recall

    # 计算预测点和真实点之间的距离矩阵
    distances = cdist(pred_coordinates, true_coordinates, metric='euclidean')

    # 找出匹配点（距离小于等于阈值的）
    matched_pred = set()
    matched_true = set()
    for i, row in enumerate(distances):
        for j, dist in enumerate(row):
            if dist <= match_threshold and i not in matched_pred and j not in matched_true:
                matched_pred.add(i)
                matched_true.add(j)

    # 计算 Precision 和 Recall
    tp = len(matched_pred)  # True Positive
    fp = len(pred_coordinates) - tp  # False Positive
    fn = len(true_coordinates) - tp  # False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def predict_torch(
        predictor,
        point_coords=None,
        point_labels=None,
        boxes=None,
        multimask_output=True,
):
    '''
    This function is copied from segment anything predictor and modified for
    '''
    # we modify the definition of point_labels here to define pos point point label = 1 , neg point label = 0
    if point_coords is not None:
        assert len(point_coords) == len(point_labels)
        points = (point_coords, point_labels)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=None,
    )

    # Predict masks
    low_res_masks, iou_predictions, cls_scores = predictor.model.mask_decoder(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        dino_feats=predictor.dino_feats,
    )
    # B,C,H,W -> B,H,W,C for MLP to process
    return low_res_masks, iou_predictions, cls_scores


@torch.no_grad()
def cache_feature(image, sam, max_steps=100, feat_size=40, patch_size=14, debug=False):
    """
    Caches image embeddings for the SAM model from the training data.

    Args:
    - train_dataloader (DataLoader): DataLoader providing the training data.
    - sam (SamPredictor): The SAM model instance.
    - max_steps (int): Maximum number of steps to cache embeddings.
    - feat_size (int): Feature size (default 40).
    - patch_size (int): Patch size (default 14).
    - debug (bool): If true, enables debugging mode.

    Returns:
    - cache (list): List of cached features including image embeddings, DINO features, target boxes, image dimensions, and masks.
    """
    # Initialize dataloader iterator

    cache = []

    imgs = image

    # Select one image for training
    image = imgs[0]
    image_np = np.array(image)

    img_height, img_width = image_np.shape[1:]
    image_np = np.transpose(image_np, (1, 2, 0))

    # Set the image in the SAM model
    sam.set_image(image_np)

    # Get DINO features
    dino_features = sam.dino_feats

    # Cache the image embeddings, DINO features, target boxes, image dimensions, and masks
    cache.append([sam.get_image_embedding().cuda(), dino_features.cuda(), (img_height, img_width),
                  ])

    return cache


# 训练函数
def train(model, train_dataloader, optimizer, lossfunc, threshold, device, epoch, logger, save_path):
    model.train()
    train_loss, iou_list, dice_list = [], [], []

    progress_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader), mininterval=30)

    for batch_idx, data in enumerate(progress_bar):
        try:
            # 尝试解包四个变量
            image, label, points, text = data
        except Exception:
            image, label, points = data
            image = image.to(device)
            label = label.to(device)

        optimizer.zero_grad()
        try:
            denoised_img, pred = model(image, points, text)
        except Exception:
            # print("没有文本输入，没有文本输入，没有文本输入")
            denoised_img, pred = model(image, points)
        pred = pred.to(device=device)
        loss = lossfunc(pred, label)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}", "dice": f"{dice:.4f}"})
        # logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, IoU = {iou:.4f}, Dice = {dice:.4f}")
        log_save_path = os.path.join(save_path, "train_sample")
        if batch_idx < 3:
            save_visualization(
                denoised_img=denoised_img,
                pred=pred,
                label=label,
                save_path=log_save_path,
                filenames=train_dataloader.dataset.image_list,
                idx=batch_idx,
                epoch=epoch,
            )

    loss_mean = np.mean(train_loss)
    iou_mean = np.mean(iou_list)
    dice_mean = np.mean(dice_list)
    logger.info(
        f"| epoch {epoch:3d} | train loss {loss_mean:5.4f} | train iou {iou_mean:3.2f} | train dice {dice_mean:3.2f}"
    )
    return loss_mean, iou_mean, dice_mean


def evaluate(model, val_dataloader, lossfunc, threshold, device, save_path, epoch):
    """
    评估模型在验证集上的性能，并可视化结果。
    自动打印/校验中间变量维度、设备、类型，降低调试难度。
    """
    model.eval()
    val_loss = []
    iou_list = []
    dice_list = []

    log_save_path = os.path.join(save_path, "val_sample")
    os.makedirs(log_save_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            try:
                image, label, points, text = data
            except Exception:
                image, label, points = data
                text = None

            # === 设备迁移与检查 ===
            image = image.to(device)
            label = label.to(device)
            if isinstance(points, torch.Tensor):
                points = points.to(device)

            # print(f"\n[Batch {batch_idx}]")
            # print(f"  image.shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")
            # print(f"  label.shape: {label.shape}, dtype: {label.dtype}, device: {label.device}")
            #
            # # === 基本校验 ===
            assert image.dim() == 4, f"image 必须是 [B, C, H, W]，当前是 {image.shape}"
            assert label.dim() in [3, 4], f"label 必须是 [B, H, W] 或 [B, 1, H, W]，当前是 {label.shape}"

            try:
                denoised_img, pred = model(image, points, text)
            except Exception:
                denoised_img, pred = model(image, points)

            # print(f"  pred.shape: {pred.shape}, denoised_img.shape: {denoised_img.shape}")
            assert pred.dim() == 4, f"pred 应为 [B, C, H, W]，实际为 {pred.shape}"
            assert denoised_img.dim() == 4, f"denoised_img 应为 [B, C, H, W]，实际为 {denoised_img.shape}"

            # === label 补充 channel 维 ===
            if label.dim() == 3:
                label = label.unsqueeze(1)
                print(f"  label 补充 channel 维度: {label.shape}")

            # === batch 文件名 ===
            batch_size = pred.size(0)
            filenames = val_dataloader.dataset.image_list[
                        batch_idx * batch_size: (batch_idx + 1) * batch_size
                        ]
            assert len(filenames) == batch_size, "filenames 数量与 batch size 不一致"

            for b in range(batch_size):
                print(f"\n  [Sample {b}]")

                single_pred = pred[b]
                single_label = label[b]
                filename = filenames[b]
                base_name = os.path.splitext(os.path.basename(filename))[0]
                full_filename = f"epoch{epoch}_{base_name}.png"

                # 维度规整
                if single_pred.dim() == 3 and single_pred.size(0) != 1:
                    single_pred = single_pred[0]
                single_pred = single_pred.detach().cpu().squeeze()

                # print(f"    single_pred.shape: {single_pred.shape}, dtype: {single_pred.dtype}")
                assert single_pred.dim() == 2, f"single_pred 应为 [H, W]，实际为 {single_pred.shape}"

                # 保存 denoised / pred / label 图像
                save_visualization(
                    denoised_img=denoised_img,
                    pred=pred,
                    label=label,
                    save_path=log_save_path,
                    filenames=val_dataloader.dataset.image_list,
                    idx=batch_idx,
                    epoch=epoch,
                )
                # print(f"    可视化图已保存到 {log_save_path}")

                # 提取坐标并可视化点图
                extract_coordinates_and_visualize_via_watershed(
                    pred_density_np=single_pred,
                    save_path=log_save_path,
                    filename=full_filename,
                    min_area=3,
                    local_max_size=9,
                )
                # print(f"    点图保存完成：{full_filename}")

            # === 损失计算与检查 ===
            loss = lossfunc(pred, label)
            assert isinstance(loss, torch.Tensor), "loss 应为 Tensor 类型"
            val_loss.append(loss.item())
            # print(f"  当前 batch loss: {loss.item():.4f}")

            # === 计算 IoU 和 Dice ===
            iou, dice = eval_seg(pred, label, threshold)
            iou_list.append(iou)
            dice_list.append(dice)
            # print(f"  当前 batch IoU: {iou:.4f}, Dice: {dice:.4f}")

    # === 汇总 ===
    loss_mean = np.mean(val_loss)
    iou_mean = np.mean(iou_list)
    dice_mean = np.mean(dice_list)

    logger.info(
        f"| epoch {epoch:3d} | val loss {loss_mean:5.4f} | val iou {iou_mean:3.2f} | val dice {dice_mean:3.2f}"
    )

    return loss_mean, iou_mean, dice_mean


def test_evaluate(
        model, val_dataloader, lossfunc, threshold, device,
        save_path, data_name, epoch=0
):
    """
    评估模型在验证集上的性能，记录每个 batch 的 F1、Precision、Recall、MAE、MSE。
    同时保存 denoised、pred、label、kpoint 图。
    """
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    vis_path = os.path.join(save_path, f"test_sample/{data_name}")
    os.makedirs(vis_path, exist_ok=True)

    batch_results = []

    # ✅ 文件命名：带时间和 data_name
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = os.path.join(save_path, f"evaluation_{data_name}_{date_str}.txt")

    with open(result_file, "w") as f:
        f.write("Evaluation Results:\n" + "=" * 50 + "\n")

    total_mae = 0
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            try:
                image, label, text, gt_point = data
            except Exception:
                image, label, gt_point = data
                text = None

            filenames = val_dataloader.dataset.image_list[
                        batch_idx * image.size(0):(batch_idx + 1) * image.size(0)
                        ]

            image = image.to(device)
            label = label.to(device)

            try:
                denoised_img, pred = model(image, text)
            except Exception:
                denoised_img, pred = model(image)

            if label.dim() == 3:
                label = label.unsqueeze(1)

            batch_size = pred.size(0)
            batch_f1, batch_precision, batch_recall = 0, 0, 0
            batch_mae, batch_mse = 0, 0  # ✅ 初始化每个 batch 的误差统计

            for b in range(batch_size):
                filename = filenames[b]
                base_name = os.path.splitext(os.path.basename(filename))[0]
                full_prefix = f"epoch{epoch}_{base_name}"

                single_pred = pred[b]
                if single_pred.dim() == 3 and single_pred.size(0) != 1:
                    single_pred = single_pred[0]
                single_pred = single_pred.detach().cpu().squeeze()

                denoised_np = denoised_img[b].detach().cpu().numpy()
                pred_np = single_pred.numpy()
                label_np = label[b].detach().cpu().squeeze().numpy()

                save_image(denoised_np, os.path.join(vis_path, f"{full_prefix}_denoised.png"))
                save_image(pred_np, os.path.join(vis_path, f"{full_prefix}_pred.png"))
                save_image(label_np, os.path.join(vis_path, f"{full_prefix}_label.png"))

                pred_coordinates = extract_coordinates_and_visualize_via_watershed(
                    pred_density_np=single_pred,
                    save_path=vis_path,
                    filename=f"{full_prefix}_kpoint.png",
                    min_area=1,
                    local_max_size=9,
                )

                true_coordinates = gt_point[b]
                gt_np = np.zeros_like(label_np)
                for coord in true_coordinates:
                    x, y = coord
                    if 0 <= x < gt_np.shape[1] and 0 <= y < gt_np.shape[0]:
                        gt_np[y, x] = 1
                save_image(gt_np, os.path.join(vis_path, f"{full_prefix}_gt.png"))

                f1, precision, recall = calculate_f1_precision_recall(
                    pred_coordinates, true_coordinates, match_threshold=8
                )

                batch_f1 += f1
                batch_precision += precision
                batch_recall += recall

                pred_count = len(pred_coordinates)
                gt_count = len(true_coordinates)
                mae = abs(pred_count - gt_count)
                mse = (pred_count - gt_count) ** 2

                # ✅ 打印每张图的评估指标
                print(f"[{base_name}] F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}, MAE: {mae}, MSE: {mse}")

                # ✅ 统计全局误差
                total_mae += mae
                total_mse += mse
                total_samples += 1

                # ✅ 累加 batch 内误差
                batch_mae += mae
                batch_mse += mse

            # ✅ batch 内平均
            batch_f1 /= batch_size
            batch_precision /= batch_size
            batch_recall /= batch_size
            batch_mae /= batch_size
            batch_mse /= batch_size

            batch_results.append({
                "batch_idx": batch_idx,
                "filenames": filenames,
                "f1": batch_f1,
                "precision": batch_precision,
                "recall": batch_recall,
            })

            # ✅ 写入 batch 级结果
            with open(result_file, "a") as f:
                f.write(f"Batch Index: {batch_idx}\n")
                f.write(
                    f"F1: {batch_f1:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}, MAE: {batch_mae:.2f}, MSE: {batch_mse:.2f}\n")
                f.write(f"Files: {', '.join(filenames)}\n")
                f.write("=" * 50 + "\n")

    # ✅ 整体平均指标
    mean_precision = np.mean([r["precision"] for r in batch_results])
    mean_recall = np.mean([r["recall"] for r in batch_results])
    mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
    mean_mae = total_mae / total_samples
    mean_mse = total_mse / total_samples

    with open(result_file, "a") as f:
        f.write(f"mF1: {mean_f1:.4f}, mPrecision: {mean_precision:.4f}, mRecall: {mean_recall:.4f}\n")
        f.write(f"MAE: {mean_mae:.2f}, MSE: {mean_mse:.2f}\n")
        f.write("=" * 50 + "\n")

    print(
        f"\n✅ Evaluate Done! mF1: {mean_f1:.4f}, mP: {mean_precision:.4f}, mR: {mean_recall:.4f}, MAE: {mean_mae:.2f}, MSE: {mean_mse:.2f}")

    # ❗ 找出 F1 最低的 batch
    lowest_batches = sorted(batch_results, key=lambda x: x["f1"])[:3]
    for batch in lowest_batches:
        print(
            f"❗ Low F1 Batch {batch['batch_idx']} | F1: {batch['f1']:.4f}, P: {batch['precision']:.4f}, R: {batch['recall']:.4f}\n  Files: {batch['filenames']}")
        with open(result_file, "a") as f:
            f.write(
                f"Low F1 Batch {batch['batch_idx']} | F1: {batch['f1']:.4f}, P: {batch['precision']:.4f}, R: {batch['recall']:.4f}\n  Files: {batch['filenames']}\n")

    return mean_f1, mean_precision, mean_recall, mean_mae, mean_mse


def custom_collate_fn(batch):
    # 解包 batch 元素
    images, labels, gt_points = zip(*batch)

    # 将 images 和 labels 合并为 tensor
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    # gt_points 仍然是 list of lists
    return images, labels, list(gt_points)


# 主程序
if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('-data_name', type=str, required=True, help='Dataset name')
    parser.add_argument('-exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('-bs', type=int, default=4, help='Batch size')
    parser.add_argument('-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('-model_type', type=str, default="vit_h", help='SAM model type')
    parser.add_argument('-ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('-sam_ckpt', type=str, required=True, help='SAM checkpoint path')
    parser.add_argument('-save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('-mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--config_file', default='/home/zy/wjj/CrowdSAM-main/configs/crowdhuman.yaml')
    parser.add_argument('-image_encoder_configuration', type=int, nargs='+',
                        default=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                 3, 3, 3],
                        help='Image encoder configuration: 0: original SAM, 1: space adapter, 2: MLP adapter, 3: space + MLP adapter')
    parser.add_argument('-fine_tuning_configuration', type=int, nargs='+',
                        default=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0],
                        help="1: doesn't freeze the specific block, 0: freeze the block")
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    # 设置随机种子
    set_seed()
    # 设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args, model_type=args.model_type, sam_checkpoint=args.sam_ckpt).to(device)
    # 损失函数
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # lossfunc=RegionMAELoss(weight=1.0, bkg_weight=0.1, ratio=0.9)
    # lossfunc = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')

    # 测试阈值
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    # 日志路径设置
    log_path = os.path.join(args.save_path, "log", f"{datetime.now().strftime('%m%d_%H_%M')}.txt")
    logger = setup_logger(log_path)
    # 模型测试分支
    if args.mode == 'test':
        test_data = TestDataset(
            os.path.join(args.data_path, 'images/'),
            os.path.join(args.data_path, 'labels/'),
            # os.path.join(args.data_path, 'test/text/'),
            None,
            os.path.join(args.data_path),
            is_robustness=False  # 或者根据需求设置为 True
        )

        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=args.bs,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        # 检查是否提供了 ckpt 路径，并检查路径是否有效
        if not args.ckpt or not os.path.exists(args.ckpt):
            raise FileNotFoundError(
                f"Error: Checkpoint file not found or invalid path provided: {args.ckpt}. Please provide a valid checkpoint path.")

        # # 加载模型的 state_dict
        # state_dict = torch.load(args.ckpt, map_location=device)
        #
        # # 获取模型的层名
        # model_keys = set(model.state_dict().keys())

        # # 获取 state_dict 中的层名
        # state_dict_keys = set(state_dict.keys())

        # # 打印出不匹配的层（模型中有但是 state_dict 中没有的层）
        # missing_keys = model_keys - state_dict_keys
        # if missing_keys:
        #     print("Missing keys (in model but not in state_dict):")
        #     for key in missing_keys:
        #         print(key)

        # # 打印出不匹配的层（state_dict 中有但是模型中没有的层）
        # unexpected_keys = state_dict_keys - model_keys
        # if unexpected_keys:
        #     print("Unexpected keys (in state_dict but not in model):")
        #     for key in unexpected_keys:
        #         print(key)
        # 加载模型的 state_dict
        state_dict = torch.load(args.ckpt, map_location=device)

        # 只加载模型的 state_dict
        model_state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict

        # 将 state_dict 加载到模型中
        model.load_state_dict(model_state_dict, strict=True)

        # 测试模型
        f1, mean_precision, mean_recall, mean_mae, mean_mse = test_evaluate(model, test_dataloader, lossfunc, threshold,
                                                                            device, args.save_path, args.data_name)
    # 模型训练分支
    elif args.mode == 'train':
        # 加载训练数据集和验证数据集
        train_data = TrainDataset(
            os.path.join(args.data_path, 'train/images/'),
            os.path.join(args.data_path, 'train/labels/'),
            # os.path.join(args.data_path, 'train/groundtruth/'),
            # os.path.join(args.data_path, 'train/text/'),
            is_robustness=False  # 如果需要
        )
        val_data = TestDataset(
            os.path.join(args.data_path, 'valid/images/'),
            os.path.join(args.data_path, 'valid/labels/'),  # os.path.join(args.data_path, 'valid/groundtruth/'),
            # os.path.join(args.data_path, 'valid/text/'),
            is_robustness=False  # 如果需要
        )
        # 数据加载器
        train_dataloader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle=True)

        # print(f"Number of model parameters: {len(list(model.parameters()))}")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter name: {name}")
        #
        #         print("-" * 30)

        # 优化器
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # 初始化损失
        best_loss = float('inf')
        interval_cache = {
            "best_loss": float("inf"),  # 当前区间内最优 val_loss
            "best_epoch": None,  # 当前区间内最佳 epoch
        }
        # 训练
        for epoch in range(args.epochs):
            # 训练
            train_loss, train_iou, train_dice = train(model, train_dataloader, optimizer, lossfunc, threshold, device,
                                                      epoch, logger, args.save_path)
            # 验证
            torch.cuda.empty_cache()
            val_loss, val_iou, val_dice = evaluate(model, val_dataloader, lossfunc, threshold, device, args.save_path,
                                                   epoch)

            # 保存模型：最小val_loss
            # best_loss = update_and_save_model(
            #     model, "./checkpoints", val_loss, epoch,
            #     best_loss=best_loss,
            #     optimizer=optimizer,
            #     model_name="my_model",
            #     max_models=3,
            #     save_strategy="always_best"
            # )

            # 保存模型：每 10 个 epoch 保存一个阶段内最优
            update_and_save_model(
                model,
                os.path.join(args.save_path, "model"),
                val_loss,
                epoch,
                optimizer=optimizer,
                model_name="my_model",
                max_models=20,
                save_strategy="interval_best",
                interval=10,
                cache_dict=interval_cache
            )

            # export PYTHONPATH = / home / user / wjj / SAML / Prompt_sam_localization / segment_anything:$PYTHONPATH
