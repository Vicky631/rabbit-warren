from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm  # 用于显示进度条

# 检查是否使用 Google Colab
using_colab = True
if using_colab:
    import torch
    import torchvision

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())


# 显示分割结果
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # 随机颜色
        img[m] = color_mask
    ax.imshow(img)


# 加载并显示图片
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# 获取SAM模型
def get_sam_model(model_type="vit_h", checkpoint="images/sam_vit_h_4b8939.pth", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return sam


# 生成掩码并显示
def generate_and_show_masks(image, sam, threshold_params=None):
    # 初始化分割生成器
    if threshold_params is None:
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        mask_generator = SamAutomaticMaskGenerator(**threshold_params)

    # 生成掩码
    masks = mask_generator.generate(image)

    # 输出掩码数量和结构
    print(f"Number of masks generated: {len(masks)}")
    if len(masks) > 0:
        print(f"Mask keys: {masks[0].keys()}")

    # 显示掩码
    return masks


# 保存掩码图像
def save_masks(image, masks, output_path, image_name):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    output_file = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}_mask.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


# 主程序
if __name__ == "__main__":
    # 设置路径和设备
    sam_checkpoint = "images/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    # 获取SAM模型
    sam = get_sam_model(model_type=model_type, checkpoint=sam_checkpoint, device=device)

    # 输入和输出文件夹路径
    input_folder = "images/part_A_final/train_data/images"  # 图片文件夹
    output_folder = "output_masks"  # 掩码输出文件夹

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的所有图片文件
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 设置严格的阈值配置，用于第二组掩码
    threshold_params_strict = {
        "model": sam,
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100  # 需要OpenCV进行后处理
    }

    # 使用tqdm显示进度
    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        image_name = os.path.basename(image_path)
        print(f"Processing {image_name}...")

        # 加载图像
        image = get_image(image_path)

        # 生成并显示第一个掩码（使用默认参数）
        masks1 = generate_and_show_masks(image, sam)
        save_masks(image, masks1, output_folder, image_name)

        # 生成并显示第二个掩码（使用严格的阈值参数）
        masks2 = generate_and_show_masks(image, sam, threshold_params_strict)
        save_masks(image, masks2, output_folder, image_name)

    print("Mask generation completed for all images.")

