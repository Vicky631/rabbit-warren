import argparse
from os import write
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch
import logging
import torch.nn as nn
from monai.losses import DiceCELoss
from unet_model import UNet_model
import sys
sys.path.insert(0, "/home/zy/wjj/Prompt_sam_localization/segment_anything")
from segment_anything import sam_model_registry
from tqdm import tqdm
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from torchvision import transforms
import math
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from scipy.spatial.distance import cdist
import scipy.io
from datetime import datetime
from GDFN import GDFN

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

# SAM 模型
class Sam_model(nn.Module):
    def __init__(self, model_type, sam_checkpoint):
        super(Sam_model, self).__init__()
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

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


# 主模型：结合 UNet 和 SAM
class Model(nn.Module):
    def __init__(self, model_type, sam_checkpoint):
        super(Model, self).__init__()
        # self.unet = UNet_model(in_channels=3, out_channels=3)

        self.gdfn = GDFN(in_channels=64, hidden_channels=128, out_channels=64)
        self.sam = Sam_model(model_type=model_type, sam_checkpoint=sam_checkpoint)

        # 冻结非 mask_decoder 部分的参数
        for n, value in self.sam.named_parameters():
            if 'mask_decoder' in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, x, points,text=None):
        # denoised_img = self.unet(x)
        denoised_img = self.gdfn(x)
        # denoised_img = x
        img_add = x + denoised_img
        img_add = torch.clamp(img_add, 0, 255)
        try:
            masks = self.sam(img_add, points,text)
        except Exception:
            masks = self.sam(img_add, points)
        return denoised_img, masks


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
        pred_np = pred[i].detach().cpu().numpy()
        save_image(
            pred_np,
            os.path.join(save_path, f"epoch{epoch}_{base_name}_pred.png")
        )

        # 保存 label
        label_np = label[i].detach().cpu().numpy()
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
def save_best_model(model, save_path, best_loss, val_loss, epoch, optimizer=None, model_name="best_model", max_models=1):
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
        print(f"Saved best model at epoch {epoch} with validation loss: {val_loss:.4f} (best: {best_loss:.4f}) at {timestamp}")

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


def extract_coordinates_and_visualize(pred_density, save_path, filename, threshold=0.1, local_max_size=37):
    """
    从预测密度图中提取定位点的坐标，并将其可视化保存。

    Args:
        pred_density (torch.Tensor): 预测密度图，形状为 (H, W) 或 (1, H, W)。
        save_path (str): 保存可视化结果的路径。
        filename (str): 保存文件的名称。
        threshold (float): 密度图的阈值，用于过滤低值区域。
        local_max_size (int): 局部极大值的核大小，用于定位点。

    Returns:
        list: 点的坐标列表，每个点为 (x, y) 格式。
    """
    # 添加批次和通道维度以适应 max_pool2d 的输入格式
    if pred_density.dim() == 2:
        pred_density = pred_density.unsqueeze(0).unsqueeze(0)
    elif pred_density.dim() == 3:
        pred_density = pred_density.unsqueeze(0)

    # 获取密度图的最大值
    max_value = torch.max(pred_density).item()
    padding = (local_max_size-1) // 2
    # 使用最大池化找到局部极大值
    keep = F.max_pool2d(pred_density, (local_max_size, local_max_size), stride=1, padding=padding)
    keep = (keep == pred_density).float()  # 找到局部极大值的位置
    pred_density = keep * pred_density

    # 设置局部极大值的阈值
    pred_density[pred_density < threshold * max_value] = 0  # 小于阈值的点置为 0
    pred_density[pred_density > 0] = 1  # 局部极大值点置为 1

    # 提取局部极大值的坐标
    points = pred_density.squeeze().nonzero(as_tuple=True)  # 非零位置
    coordinates = list(zip(points[1].cpu().numpy(), points[0].cpu().numpy()))  # 转为 (x, y) 格式

    # 生成点坐标可视化地图
    generate_pointmap_and_save(
        kpoint=pred_density.squeeze(0).squeeze(0).cpu().numpy(),
        coordinates=coordinates,
        save_path=save_path,
        filename=filename,

    )

    return coordinates


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
    print(f" saved to: {save_file}")

    return coordinates



def read_mat_and_generate_gt_plot(file_path, save_path):
    # 读取 .mat 文件
    data = scipy.io.loadmat(file_path)

    # 提取坐标信息 (假设字段名为 'image_info')
    if 'image_info' in data:
        points = data['image_info'][0][0][0][0][0]  # 提取点坐标信息
    else:
        raise KeyError("The key 'image_info' was not found in the .mat file.")

    # 提取点的数量
    num_points = points.shape[0]

    # max_x = int(np.ceil(np.max(points[:, 0])))
    # max_y = int(np.ceil(np.max(points[:, 1])))
    rate = 1  # 比例因子，可根据需要调整
    # background_width = int(max_x * rate)
    # background_height = int(max_y * rate)    # 将坐标映射到 (256, 256) 的范围
    # # 初始化动态背景
    # image = np.ones((background_height, background_width, 3), dtype=np.uint8) * 255  # 白色背景

    points[:, 0] = (points[:, 0] / np.max(points[:, 0])) * 256
    points[:, 1] = (points[:, 1] / np.max(points[:, 1])) * 256  # 反转 Y 轴

    # 创建一个空白图像 (256x256)
    image = np.ones((256, 256, 3), dtype=np.uint8) * 255  # 白色背景


    # 在图像上绘制点
    for point in points:
        x, y = int(point[0] * rate), int(point[1] * rate)
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # 红色点

    # 保存图像到指定路径
    cv2.imwrite(save_path, image)
    print(f"Scatter plot saved to: {save_path}")
    print(f"Total number of points: {num_points}")

    return points



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

# 训练函数
def train(model, train_dataloader, optimizer, lossfunc, threshold, device, epoch, logger,save_path):
    model.train()
    train_loss, iou_list, dice_list = [], [], []

    progress_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))

    for batch_idx, data in enumerate(progress_bar):
        try:
            # 尝试解包四个变量
            image, label, points, text = data
        except Exception:
            # 如果解包失败，尝试解包三个变量
            # print("没有文本输入，没有文本输入，没有文本输入")
            image, label, points = data

        image, label = image.to(device=device), label.to(device=device)
        optimizer.zero_grad()
        try:
            denoised_img, pred = model(image, points,text)
        except Exception:
            print("没有文本输入，没有文本输入，没有文本输入")
            denoised_img, pred = model(image, points)

        loss = lossfunc(pred, label)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}", "dice": f"{dice:.4f}"})
        # logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, IoU = {iou:.4f}, Dice = {dice:.4f}")
        log_save_path = os.path.join(save_path, "sample")
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



def evaluate(model, val_dataloader, lossfunc, threshold, device, save_path="output"):
    """
    评估模型在验证集上的性能，并可视化结果。

    Args:
        model (torch.nn.Module): 被评估的模型。
        val_dataloader (DataLoader): 验证集数据加载器。
        lossfunc (Callable): 损失函数。
        threshold (tuple): 用于计算 IoU 和 Dice 的阈值列表。
        device (torch.device): 设备 (CPU 或 CUDA)。
        save_path (str): 保存可视化结果的路径。

    Returns:
        tuple: 验证集上的平均损失、IoU 和 Dice 系数。
    """
    model.eval()  # 进入评估模式

    val_loss = []
    iou_list = []
    dice_list = []

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):

            try:
                # 尝试解包四个变量
                image, label, points, text = data
            except Exception:
                # 如果解包失败，尝试解包三个变量
                # print("没有文本输入，没有文本输入，没有文本输入")
                image, label, points = data

            image, label = image.to(device=device), label.to(device=device)
            optimizer.zero_grad()
            try:
                denoised_img, pred = model(image, points, text)
            except Exception:
                # print("没有文本输入，没有文本输入，没有文本输入")
                denoised_img, pred = model(image, points)

            # 提取当前批次的文件名
            filenames = val_dataloader.dataset.image_list[
                batch_idx * image.size(0):(batch_idx + 1) * image.size(0)
            ]
            # 遍历批次中的每个样本
            batch_size = pred.size(0)
            for b in range(batch_size):
                single_pred = pred[b]  # 提取单个预测结果 (channel, h, w)
                single_denoised_img = denoised_img[b]
                filename = filenames[b]  # 获取当前样本的文件名

                # 确保文件名带扩展名（如无扩展名则添加 `.png`）
                if not filename.endswith(".png"):
                    filename = filename.split('.')[0] + ".png"

                # 保存原始密度图
                single_pred1 = pred[b].squeeze(0).cpu().numpy()  # 移除多余维度，得到 (H, W)
                single_pred1 = (single_pred1 * 255).clip(0, 255).astype(np.uint8)
                filename_pred = os.path.join(save_path, f"{filenames[b].split('.')[0]}_pred.png")
                os.makedirs(os.path.dirname(filename_pred), exist_ok=True)  # 确保目录存在
                cv2.imwrite(filename_pred, single_pred1)
                print(f"Prediction saved to: {filename_pred}")

                # 提取并可视化坐标点
                extract_coordinates_and_visualize(
                    pred_density=single_pred,
                    save_path=save_path,
                    filename=filename,
                    threshold=0.1,
                    local_max_size=5
                )
                print(f"Extracted Coordinates for {filename}")

            # 计算损失
            loss = lossfunc(pred, label)
            val_loss.append(loss.item())

            # 计算 IoU 和 Dice 系数
            iou, dice = eval_seg(pred, label, threshold)
            iou_list.append(iou)
            dice_list.append(dice)

        # 计算平均损失、IoU 和 Dice
        loss_mean = np.mean(val_loss)
        iou_mean = np.mean(iou_list)
        dice_mean = np.mean(dice_list)
    logger.info(
        f"| epoch {epoch:3d} | val loss {loss_mean:5.4f} | val iou {iou_mean:3.2f} | val dice {dice_mean:3.2f}"
    )
    return loss_mean, iou_mean, dice_mean

def test_evaluate(
    model, val_dataloader, lossfunc, threshold, device,
    save_path="output_no_unet"
):
    """
    评估模型在验证集上的性能，并记录每个 batch 的 F1、Precision、Recall。

    Args:
        model (torch.nn.Module): 被评估的模型。
        val_dataloader (DataLoader): 验证集数据加载器。
        lossfunc (Callable): 损失函数。
        threshold (tuple): 用于计算 IoU 和 Dice 的阈值列表。
        device (torch.device): 设备 (CPU 或 CUDA)。
        save_path (str): 保存结果的路径。

    Returns:
        tuple: 平均 F1、Precision、Recall 和最低的三个 batch 文件名。
    """
    model.eval()  # 进入评估模式
    os.makedirs(save_path, exist_ok=True)

    # 记录每个 batch 的结果
    batch_results = []
    result_file = os.path.join(save_path, "evaluation_results.txt")

    # 如果文件已经存在，先清空内容
    with open(result_file, "w") as f:
        f.write("Evaluation Results:\n")
        f.write("=" * 50 + "\n")
    with torch.no_grad():
        for batch_idx, (image, label, points, text) in enumerate(val_dataloader):
            # 提取当前 batch 的文件名
            filenames = val_dataloader.dataset.image_list[
                batch_idx * image.size(0):(batch_idx + 1) * image.size(0)
            ]

            gtfiles = val_dataloader.dataset.gt_list[batch_idx*image.size(0):(batch_idx+1)*image.size(0)]

            # 将数据转移到设备
            image = image.to(device)
            # label = label.to(device)

            # 获取预测结果
            denoised_img, pred = model(image, points, text)
            batch_size = pred.size(0)

            # 初始化 batch 的指标
            batch_f1, batch_precision, batch_recall = 0, 0, 0

            for b in range(batch_size):
                # 提取预测和真值
                single_pred = pred[b]
                filename = filenames[b]

                # 保存原始密度图
                single_pred1 = pred[b].squeeze(0).cpu().numpy()  # 移除多余维度，得到 (H, W)
                single_pred1 = (single_pred1 * 255).clip(0, 255).astype(np.uint8)
                filename_pred = os.path.join(save_path, f"{filenames[b].split('.')[0]}-pred.png")
                os.makedirs(os.path.dirname(filename_pred), exist_ok=True)  # 确保目录存在
                cv2.imwrite(filename_pred, single_pred1)


                # 提取预测点和真值点
                pred_coordinates = extract_coordinates_and_visualize(
                    pred_density=single_pred,
                    save_path=save_path,
                    filename=filename,
                    threshold=0.1,
                    local_max_size=9,
                )
                true_coordinates = read_mat_and_generate_gt_plot(
                    os.path.join(val_dataloader.dataset.gt_path, gtfiles[b]),
                    os.path.join(save_path, filename.replace(".jpg","_gt.jpg")),
                )

                # 计算 F1、Precision、Recall
                f1, precision, recall = calculate_f1_precision_recall(
                    pred_coordinates, true_coordinates, match_threshold=5
                )

                batch_f1 += f1
                batch_precision += precision
                batch_recall += recall

            # 计算 batch 的平均指标
            batch_f1 /= batch_size
            batch_precision /= batch_size
            batch_recall /= batch_size

            # 保存结果到 batch_results
            batch_results.append({
                "batch_idx": batch_idx,
                "filenames": filenames,
                "f1": batch_f1,
                "precision": batch_precision,
                "recall": batch_recall,
            })

            # 保存当前 batch 的结果到总文件中
            with open(result_file, "a") as f:
                f.write(f"Batch Index: {batch_idx}\n")
                f.write(f"F1: {batch_f1:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}\n")
                f.write(f"Files: {', '.join(filenames)}\n")
                f.write("=" * 50 + "\n")

            # print(f"Saved results for batch {batch_idx} to {result_file}")



    # 计算全局精确率和召回率
    mean_precision = np.sum([result["precision"] for result in batch_results]) / len(batch_results)
    mean_recall = np.sum([result["recall"] for result in batch_results]) / len(batch_results)

    # 计算全局 F1 分数
    mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

    with open(result_file, "a") as f:
        f.write(f"mF1: {mean_f1:.4f}, mPrecision: {mean_precision:.4f}, mRecall: {mean_recall:.4f}\n")
        f.write("=" * 50 + "\n")
    print("mf1:",mean_f1,"mpre:", mean_precision,"mrecall:",mean_recall)
    # 找出 F1 最低的三个 batch
    sorted_batches = sorted(batch_results, key=lambda x: x["f1"])
    lowest_batches = sorted_batches[:3]
    for batch in lowest_batches:
        print(f"Batch {batch['batch_idx']}, name: {batch['filenames']}, F1: {batch['f1']:.4f}, precision: {batch['precision']}, recall: {batch['recall']}\n")
        with open(result_file, "a") as f:
            f.write(f"Batch {batch['batch_idx']}, name: {batch['filenames']}, F1: {batch['f1']:.4f}, precision: {batch['precision']}, recall: {batch['recall']}\n")

    return mean_f1, mean_precision, mean_recall

# 主程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('-data_name', type=str, required=True, help='Dataset name')
    parser.add_argument('-exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('-bs', type=int, default=2, help='Batch size')
    parser.add_argument('-epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('-model_type', type=str, default="vit_h", help='SAM model type')
    parser.add_argument('-ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('-sam_ckpt', type=str, required=True, help='SAM checkpoint path')
    parser.add_argument('-save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('-mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_type=args.model_type, sam_checkpoint=args.sam_ckpt).to(device)
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    log_path = os.path.join(args.save_path, "log", f"{datetime.now().strftime('%m%d_%H_%M')}.txt")
    logger = setup_logger(log_path)

    if args.mode == 'test':
        test_data = TestDataset(
            os.path.join(args.data_path, 'test/images/'),
            os.path.join(args.data_path, 'test/labels/'),
            os.path.join(args.data_path, 'test/text/'),os.path.join(args.data_path, 'test/ground_truth/'),
            is_robustness=False  # 或者根据需求设置为 True
        )

        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=args.bs,
            shuffle=False
        )


        # 检查是否提供了 ckpt 路径，并检查路径是否有效
        if not args.ckpt or not os.path.exists(args.ckpt):
            raise FileNotFoundError(
                f"Error: Checkpoint file not found or invalid path provided: {args.ckpt}. Please provide a valid checkpoint path.")

        # 加载模型的状态字典
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        # 测试模型
        f1 = test_evaluate(model, test_dataloader, lossfunc, threshold, device)


    elif args.mode == 'train':
        train_data = TrainDataset(
            os.path.join(args.data_path, 'train/images/'),
            os.path.join(args.data_path, 'train/labels/'),
            # os.path.join(args.data_path, 'train/groundtruth/'),
            # os.path.join(args.data_path, 'train/text/'),
            is_robustness=False  # 如果需要
        )

        val_data = TestDataset(
            os.path.join(args.data_path, 'valid/images/'),
            os.path.join(args.data_path, 'valid/labels/'), #os.path.join(args.data_path, 'valid/groundtruth/'),
            # os.path.join(args.data_path, 'valid/text/'),
            is_robustness=False  # 如果需要
        )

        train_dataloader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



        best_loss = float('inf')
        for epoch in range(args.epochs):
            train_loss, train_iou, train_dice = train(model, train_dataloader, optimizer, lossfunc, threshold, device, epoch, logger,args.save_path)
            torch.cuda.empty_cache()
            val_loss, val_iou, val_dice = evaluate(model, val_dataloader, lossfunc, threshold, device)

            best_loss = save_best_model(
                model=model,
                save_path=os.path.join(args.save_path,"model"),
                best_loss=best_loss,
                val_loss=val_loss,
                epoch=epoch,
                optimizer=optimizer,
                model_name="no_text",
                # max_models=3  # 保存最多 3 个模型
            )