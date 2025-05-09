import os
import numpy as np
from PIL import Image


class ImageLoader:
    def __init__(self, image_dir = '/home/zy/wjj/Prompt_sam_localization/dataset/UCF-QNRF/QNRF/images'):
        """
        初始化图像加载器类。
        :param image_dir: 图像文件所在的目录路径
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_files.sort()  # 按照文件名排序，确保顺序一致
        self.current_index = 0  # 当前图像索引

    def load_next_image(self):
        """
        返回按顺序加载的下一张图片。
        :return: 返回下一张图像的ndarray格式。如果没有更多图片，则返回None
        """
        if self.current_index < len(self.image_files):
            # 获取当前图像路径
            image_file = self.image_files[self.current_index]
            image_path = os.path.join(self.image_dir, image_file)

            # 打开图片
            img = Image.open(image_path)

            # 转换为RGB格式
            img = img.convert('RGB')

            # 将图片转换为ndarray
            img_array = np.array(img)

            # 更新当前索引
            self.current_index += 1

            return img_array
        else:
            return None  # 如果已经没有更多图片了，返回None

    def reset(self):
        """
        重置索引，以便重新从头开始加载图像。
        :return: None
        """
        self.current_index = 0


import torch
import matplotlib.pyplot as plt
import numpy as np


class TensorVisualizer:
    def __init__(self, tensor):
        # 确保输入是一个Tensor，并且具有形状 (1, 1, H, W)
        if isinstance(tensor, torch.Tensor) and tensor.ndimension() == 4 and tensor.shape[0] == 1 and tensor.shape[
            1] == 1:
            self.tensor = tensor
        else:
            raise ValueError("Tensor must have shape (1, 1, H, W)")

    def normalize_tensor(self):
        # 归一化到 [0, 1] 范围
        min_val = self.tensor.min()
        max_val = self.tensor.max()
        normalized_tensor = (self.tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def save_image(self, save_path):
        # 归一化Tensor
        normalized_tensor = self.normalize_tensor()

        # 将归一化后的Tensor转换为NumPy数组，并去掉 batch 和 channel 维度
        image = normalized_tensor.squeeze().cpu().detach().numpy()

        # 创建可视化
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Image saved at {save_path}")