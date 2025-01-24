import argparse
from dataset import TrainDataset,TestDataset
from torch.utils.data import DataLoader
import torch
import logging
import torch.nn as nn
from monai.losses import DiceCELoss
from unet_model import UNet_model
from segment_anything import sam_model_registry
from torch.autograd import Function
import torchmetrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import numpy as np
# from matplotlib.pylab import mpl
# mpl.use('Qt5Agg')
from torchvision.utils import save_image
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
class Sam_model(nn.Module):
    def __init__(self,model_type,sam_checkpoint):
        super(Sam_model, self).__init__()

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    def forward(self, x, points):

        image = self.sam.image_encoder(x)
        se, de = self.sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        pred, _ = self.sam.mask_decoder(
            image_embeddings=image,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
        )

        return pred





class Model(nn.Module):
    def __init__(self,model_type,sam_checkpoint):
        super(Model,self).__init__()

        self.unet = UNet_model(in_channels=3,out_channels=3)     # 这个可能就是head的本体了
        self.sam = Sam_model(model_type=model_type,sam_checkpoint=sam_checkpoint)

        for n, value in self.sam.named_parameters():
            if 'mask_decoder' in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, x, points):

        denoised_img = self.unet(x)
        img_add = x + denoised_img
        img_add = torch.clamp(img_add,0,255)
        masks = self.sam(img_add,points)
        return denoised_img, masks


def save_visualization(denoised_img, pred, label, save_path, epoch, idx):
    """
    保存 denoised_img、pred 和 label 的可视化结果。

    Args:
        denoised_img (torch.Tensor): 去噪后的图片 (1, 1, H, W) 或 (1, C, H, W)。
        pred (torch.Tensor): 预测的分割结果 (1, H, W)。
        label (torch.Tensor): Ground truth (1, H, W)。
        save_path (str): 保存路径。
        epoch (int): 当前 epoch。
        idx (int): 样本索引。
    """
    os.makedirs(save_path, exist_ok=True)

    # 去噪后的图片
    denoised_img_np = denoised_img.detach().squeeze().cpu().numpy()  # 去掉多余维度
    if denoised_img_np.ndim == 3 and denoised_img_np.shape[0] == 3:  # 检测是否是 (C, H, W)
        denoised_img_np = denoised_img_np.transpose(1, 2, 0)  # 从 (C, H, W) -> (H, W, C)

    # 保存去噪图片
    plt.imshow(denoised_img_np, cmap='gray' if denoised_img_np.ndim == 2 else None)
    plt.title(f"Epoch {epoch}, Sample {idx}: Denoised Image")
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"epoch{epoch}_sample{idx}_denoised.png"))
    plt.close()

    # 保存预测结果
    pred_np = pred.detach().squeeze().cpu().numpy()  # 转为 NumPy 格式
    plt.imshow(pred_np, cmap='jet')
    plt.title(f"Epoch {epoch}, Sample {idx}: Prediction")
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"epoch{epoch}_sample{idx}_pred.png"))
    plt.close()

    # 保存 Ground Truth
    label_np = label.detach().squeeze().cpu().numpy()  # 转为 NumPy 格式
    plt.imshow(label_np, cmap='jet')
    plt.title(f"Epoch {epoch}, Sample {idx}: Ground Truth")
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"epoch{epoch}_sample{idx}_label.png"))
    plt.close()


#  可视化和定位
def extract_and_visualize_coordinates(pred_density, save_path, filename, threshold=0.1, local_max_size=37):
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
    if pred_density.dim() == 2:  # 如果输入是 (H, W)，添加批次和通道维度
        pred_density = pred_density.unsqueeze(0).unsqueeze(0)
    elif pred_density.dim() == 3:  # 如果输入是 (1, H, W)，添加通道维度
        pred_density = pred_density.unsqueeze(0)

    # 获取密度图的最大值
    max_value = torch.max(pred_density).item()

    # 使用最大池化找到局部极大值
    local_max = F.max_pool2d(pred_density, kernel_size=local_max_size, stride=1, padding=local_max_size // 2)
    local_max = (local_max == pred_density).float()  # 找到局部极大值位置

    # 阈值化，过滤低值区域
    pred_density = local_max * pred_density  # 仅保留局部极大值位置的值
    pred_density[pred_density < threshold * max_value] = 0  # 小于阈值的点设为 0
    pred_density[pred_density > 0] = 1  # 局部极大值位置的像素值设为 1

    # 提取点的坐标
    points = pred_density.squeeze().nonzero(as_tuple=True)  # 获取非零位置
    coordinates = list(zip(points[1].cpu().numpy(), points[0].cpu().numpy()))  # 转为 (x, y) 格式

    # 可视化并保存
    visualize_and_save(pred_density.squeeze().cpu().numpy(), coordinates, save_path, filename)

    return coordinates

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


# 保存最好的 K 个模型
def save_top_k_models(model, score, save_path, model_name, top_k=3):
    """保存最好的 K 个模型"""
    if not hasattr(save_top_k_models, "saved_models"):
        save_top_k_models.saved_models = []  # 存储模型及其分数的列表

    # 插入当前模型及分数
    save_top_k_models.saved_models.append((score, model))
    save_top_k_models.saved_models = sorted(save_top_k_models.saved_models, key=lambda x: x[0])

    # 如果超过 K 个模型，移除分数最差的
    if len(save_top_k_models.saved_models) > top_k:
        save_top_k_models.saved_models.pop(0)

    # 保存前 K 个模型
    for i, (score, model) in enumerate(save_top_k_models.saved_models):
        torch.save(model.state_dict(), f"{save_path}/{model_name}_top{i+1}.pt")


def train(model, train_dataloader):
    model.train()
    train_loss = []
    iou_list = []
    dice_list = []

    # 使用 tqdm 包装 train_dataloader
    progress_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))

    for batch_idx, (image, label, points) in enumerate(progress_bar):
        image = image.to(device=device)
        label = label.to(device=device)
        optimizer.zero_grad()

        denoised_img, pred = model(image, points)

        loss = lossfunc(pred, label) * 100
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

        # 更新进度条显示的信息
        progress_bar.set_postfix({
            "loss": f"{loss.item():.2f}",
            "iou": f"{iou:.2f}",
            "dice": f"{dice:.2f}"
        })

        # 随机保存三组样本
        if batch_idx < 3:  # 只保存前三个样本的可视化结果
            save_visualization(
                denoised_img=denoised_img,
                pred=pred,
                label=label,
                save_path="log",  # 保存文件夹路径
                epoch=epoch,
                idx=batch_idx
            )

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)

    logger.info(
        f"| epoch {epoch:3d} | "f"train loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}"
    )



def evaluate(model, val_dataloader):
    model.eval()

    val_loss = []
    iou_list = []
    dice_list = []

    with torch.no_grad():
        for i, (image, label, points) in enumerate(val_dataloader):
            # 获取当前图像文件名
            filename = val_dataloader.dataset.image_list[i]  # 从 dataset 的 image_list 中获取文件名

            image = image.to(device=device)
            label = label.to(device=device)

            # 传递到模型，获取预测结果
            denoised_img, pred = model(image, points)

            # 可视化和保存
            save_path = "/home/zy/WJJ/SAML/Prompt_sam_localization/output"
            coordinates = extract_and_visualize_coordinates(pred, save_path, filename, threshold=0.1, local_max_size=37)
            print(f"Extracted Coordinates for {filename}: {coordinates}")

            # 计算损失、IoU 和 Dice
            loss = lossfunc(pred, label)
            val_loss.append(loss.item())
            iou, dice = eval_seg(pred, label, threshold)
            iou_list.append(iou)
            dice_list.append(dice)

        # 计算平均值
        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)

    return loss_mean, iou_mean, dice_mean


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t
def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
def eval_seg(pred, true_mask_p, threshold):

    eiou, edice = 0, 0
    for th in threshold:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

        disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += cal_iou(disc_pred, disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

    return eiou / len(threshold), edice / len(threshold)


def cal_iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()
if __name__ == '__main__':
    # 初始化
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, required=True, default='./dataset/10028/30', help='The path of cryo-PPP data')
    parser.add_argument('-data_name', type=str, required=True, default='10028',help='the name of your dataset')
    parser.add_argument('-exp_name', type=str, required=True, default='exp1',help='the name of your experiment')
    parser.add_argument('-bs', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=100, help='the number of training sessions')
    parser.add_argument('-lr', type=int, default=0.00005, help='learning rate')
    parser.add_argument('-model_type', type=str, default="vit_h", help='')
    parser.add_argument('-ckpt', default=None,type=str, help='the checkpoint you want to test')
    parser.add_argument('-sam_ckpt', default='/home/zy/WJJ/SAML/Prompt_sam_localization/checkpoint/sam_vit_h_4b8939.pth', type=str, help='sam checkpoint path')
    parser.add_argument('-save_path', type=str, required=True, default='./model_checkpoint/saml',help='the path to save your training result')
    parser.add_argument('-mode', type=str, choices=['train', 'test'], required=True,help="Specify 'train' for training or 'test' for testing.")
    args = parser.parse_args()
    # 加载数据集
    # 如果是测试模式，仅加载测试集
    if args.mode == 'test':
        test_image = f'{args.data_path}/test/images/'
        test_label = f'{args.data_path}/test/labels/'
        test_data = TestDataset(test_image, test_label, is_robustness=False)
        test_dataloader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False)
        print("testing mode activated.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(model_type=args.model_type,sam_checkpoint=args.sam_ckpt)
        model = model.to(device)
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

        state_dict = torch.load(f'{args.ckpt}', map_location='cuda')
        model.load_state_dict(state_dict, strict=False)

        loss_list = []
        result_list = []
        iou_list = []
        dice_list = []
        model.eval()
        with torch.no_grad():
            for data, label, points, image_name in test_dataloader:
                data = data.to(device=device)
                label = label.to(device=device)

                denoised_img, pred = model(data, points)

                loss = lossfunc(pred, label)
                loss_list.append(loss.item())

                result = eval_seg(pred, label, threshold)
                result_list.append(result)
                iou_list.append((result[0]))
                dice_list.append(result[1])

            print(f'Total score:{np.mean(loss_list)}, IOU:{np.mean(iou_list)}, DICE:{np.mean(dice_list)}')

    if args.mode == 'train':
        train_image = f'{args.data_path}/train/images/'
        train_label = f'{args.data_path}/train/labels/'
        train_data = TrainDataset(train_image, train_label, is_robustness=False)
        train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)

        val_image = f'{args.data_path}/valid/images/'
        val_label = f'{args.data_path}/valid/labels/'
        val_data = TestDataset(val_image, val_label, is_robustness=False)
        val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True)

        print("training mode activated.")
        best_loss = 100
        best_iou = 1
        best_dice = 1
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger = logging.getLogger('SAML')

        model = Model(model_type=args.model_type, sam_checkpoint=args.sam_ckpt)
        model = model.to(device)

        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            train(model, train_dataloader)
            val_loss, iou, dice = evaluate(model, val_dataloader)

            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {val_loss:5.4f} | "f"iou {iou:3.2f}  | "f"dice {dice:3.2f}"
            )
            logger.info("-" * 89)

            # 保存最好的模型
            save_dir = args.save_path
            os.makedirs(save_dir, exist_ok=True)
            save_top_k_models(model, -val_loss, save_dir, f'head_prompt_{args.data_name}_{args.exp_name}', top_k=3)

        logger.info(f'Head prompt {args.data_name} {args.exp_name} training is complete ! ')

#
# # 计算 Precision
# precision = precision_metric(y_pred, y_true)
# print("Precision:", precision)
#
# # 计算 Recall
# recall = recall_metric(y_pred, y_true)
# print("Recall:", recall)
#
# # 计算 F1-score
# f1 = f1_metric(y_pred, y_true)
# print("F1-score:", f1)    修正以上代码不合理的地方，但是不要遗失内容