import csv
import os
import cv2
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import random

from utils.get_gt_point import PointExtractor


class TrainDataset:
    def __init__(self, image_path, label_path, text_path=None, is_robustness=False):
        """
        初始化数据集。

        Args:
            image_path (str): 图像数据路径。
            label_path (str): 标签数据路径。
            text_path (str, optional): 文本数据路径。如果为 None，则不加载文本。默认为 None。
            is_robustness (bool): 是否采用鲁棒性测试模式。
        """
        self.image_path = image_path
        self.label_path = label_path
        self.text_path = text_path
        self.is_robustness = is_robustness

        # 获取文件列表

        if "train" not in image_path:
            root_path=os.path.dirname(image_path.rstrip('/'))
            file_path = os.path.join(root_path, "train.txt")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.image_list = [line.strip() + '.jpg' for line in lines]
                self.label_list = [line.strip() + '.png' for line in lines]
                self.text_list = sorted(os.listdir(self.text_path)) if self.text_path else None
        else:
            self.image_list = sorted(os.listdir(self.image_path))
            self.label_list = sorted(os.listdir(self.label_path))
            self.text_list = sorted(os.listdir(self.text_path)) if self.text_path else None

        if self.is_robustness:
            self.image_list, self.label_list, self.text_list = self.get_images_labels_and_texts_path_for_loop()

    def __getitem__(self, item):
        """
        获取单个数据项。

        Args:
            item (int): 索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[str], Tuple[torch.Tensor, torch.Tensor]]:
                图像张量、标签张量、可选的文本嵌入、点提示 (点坐标, 点标签)。
        """
        # 加载图像
        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        resize_transform = transforms.Resize((1024, 1024))  # 指定目标尺寸
        image = resize_transform(image)
        # image = image.resize((1024,1024), Image.ANTIALIAS)
        image = transforms.ToTensor()(image)

        # 加载标签
        label_name = self.label_list[item]
        label = Image.open(os.path.join(self.label_path, label_name)).convert('L')

        # label = label.resize((256, 256), Image.LANCZOS)
        resize_transform = transforms.Resize((256, 256))  # 指定目标尺寸
        label = resize_transform(label)
        label_array = np.array(label)
        # 将所有非零像素值设置为 1
        label_array[label_array > 50] = 255
        # 使用 transforms.ToTensor() 将 numpy 数组转换为 Tensor，并将其标准化为 [0, 1]
        label_tensor = transforms.ToTensor()(Image.fromarray(label_array))
        label = label_tensor.long()


        # label = transforms.ToTensor()(label).long()





        # 加载文本（如果存在）

        if self.text_list:
            text_name = self.text_list[item]
            with open(os.path.join(self.text_path, text_name), 'r') as f:
                text = f.read().strip()
        else:
            text=None

        # 我要写个helloworld函数
        
        # 生成点提示
        points_scale = np.array(image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image, device='cuda')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda')
        points = (in_points, in_labels)

        # 返回内容
        if text is not None:
            return image, label, points, text
        else:
            return image, label, points

    def __len__(self):
        """
        返回数据集长度。

        Returns:
            int: 数据集的样本数。
        """
        return len(self.image_list)

    def get_images_labels_and_texts_path_for_loop(self):
        """
        获取鲁棒性测试模式下的图像、标签和文本路径列表。

        Returns:
            Tuple[List[str], List[str], Optional[List[str]]]: 图像路径、标签路径、可选的文本路径列表。
        """
        self.label_list_robust = sorted([img for img in random.sample(self.label_list, 5)])
        self.image_list_robust = sorted([self.image_list[self.label_list.index(image)] for image in self.label_list_robust])
        self.text_list_robust = (
            sorted([self.text_list[self.label_list.index(image)] for image in self.label_list_robust])
            if self.text_list else None
        )

        print(f'Train list: {self.label_list_robust}')
        return self.image_list_robust, self.label_list_robust, self.text_list_robust


class ValDataset:
    def __init__(self, image_path, label_path,  text_path=None, gt_path=None, is_robustness=False):
        """
        初始化测试数据集。

        Args:
            image_path (str): 图像数据路径。
            label_path (str): 标签数据路径。
            text_path (str, optional): 文本数据路径。如果为 None，则不加载文本。默认为 None。
            is_robustness (bool): 是否采用鲁棒性测试模式。
        """
        self.image_path = image_path
        self.label_path = label_path
        self.text_path = text_path
        self.gt_path = gt_path
        self.is_robustness = is_robustness

        # 获取文件列表

        if "vaild" not in image_path:
            root_path=os.path.dirname(image_path.rstrip('/'))
            file_path = os.path.join(root_path, "train.txt")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.image_list = [line.strip() + '.jpg' for line in lines]
                self.label_list = [line.strip() + '.png' for line in lines]

        else:
            self.image_list = sorted(os.listdir(self.image_path))
            self.label_list = sorted(os.listdir(self.label_path))

        # delete_ids = {
        #     3143, 3150, 3155, 3181, 3187, 3188, 3246, 3259, 3272, 3280,
        #     3309, 3319, 3335, 3346, 3348, 3380, 3387, 3389, 3390, 3414,
        #     3422, 3423, 3432, 3446, 3447, 3482, 3487, 3503, 3509, 3528,
        #     3557, 3568, 3577, 3602, 3606
        # }

        if True:
            extractor = PointExtractor()
            self.gt_points_dict = extractor.extract_all_points_from_txt(root_path+"/val_gt_loc.txt")
        else:
            self.gt_points_dict = None

        self.text_list = sorted(os.listdir(self.text_path)) if self.text_path else None

        if self.is_robustness:
            self.image_list, self.label_list, self.text_list = self.get_images_labels_and_texts_path_for_loop()

        # if "NWPU" in gt_path:

    def __getitem__(self, item):
        """
        获取单个数据项。

        Args:
            item (int): 索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[str], Tuple[torch.Tensor, torch.Tensor], List[List[int]]]:
                图像张量、标签张量、可选的文本嵌入、点提示 (点坐标, 点标签)、对应的gt点坐标。
        """
        # 加载图像
        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        resize_transform = transforms.Resize((1024, 1024))  # 指定目标尺寸
        image = resize_transform(image)
        image = transforms.ToTensor()(image)

        # 加载标签
        label_name = self.label_list[item]
        label = Image.open(os.path.join(self.label_path, label_name)).convert('L')
        original_width, original_height = label.size
        resize_transform = transforms.Resize((256, 256))  # 指定目标尺寸
        label = resize_transform(label)
        label_array = np.array(label)
        # 将所有非零像素值设置为 1
        label_array[label_array > 50] = 255
        # 使用 transforms.ToTensor() 将 numpy 数组转换为 Tensor，并将其标准化为 [0, 1]
        label_tensor = transforms.ToTensor()(Image.fromarray(label_array))
        label = label_tensor.long()

        # 加载文本（如果存在）
        text = None
        if self.text_list:
            text_name = self.text_list[item]
            with open(os.path.join(self.text_path, text_name), 'r') as f:
                text = f.read().strip()

        # 生成点提示
        points_scale = np.array(image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image, device='cuda')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda')
        points = (in_points, in_labels)

        # 获取对应的gt_points
        if self.gt_points_dict:
            image_id = os.path.splitext(image_name)[0]
            if image_id in self.gt_points_dict:
                total, points_list = self.gt_points_dict[image_id]
                if total==0:
                    gt_points = []
                else:
                    points_array = np.array(points_list)
                    points_array[:, 0] = (points_array[:, 0] * 256 / original_width).astype(int)
                    points_array[:, 1] = (points_array[:, 1] * 256 / original_height).astype(int)
                    gt_points = points_array.tolist()
            else:
                gt_points = []
        else:
            gt_points = []

        # # 把点画在 label 上
        # label_np = label.squeeze().cpu().numpy()
        # label_with_points = cv2.cvtColor((label_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # for x, y in gt_points:
        #     if 0 <= x < label_with_points.shape[1] and 0 <= y < label_with_points.shape[0]:
        #         cv2.circle(label_with_points, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        #
        # # 可视化
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(label_np, cmap='gray')
        # plt.title('Original Label')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(label_with_points, cv2.COLOR_BGR2RGB))
        # plt.title('Label with GT Points')
        # plt.show()

        # 返回内容
        if text is not None:
            return image, label,  text, gt_points
        else:
            return image, label, gt_points


    def __len__(self):
        """
        返回数据集长度。

        Returns:
            int: 数据集的样本数。
        """
        return len(self.image_list)



    def get_images_labels_and_texts_path_for_loop(self):
        """
        获取鲁棒性测试模式下的图像、标签和文本路径列表。

        Returns:
            Tuple[List[str], List[str], Optional[List[str]]]: 图像路径、标签路径、可选的文本路径列表。
        """
        self.label_list_robust = sorted([img for img in random.sample(self.label_list, 5)])
        self.image_list_robust = sorted([self.image_list[self.label_list.index(image)] for image in self.label_list_robust])
        self.text_list_robust = (
            sorted([self.text_list[self.label_list.index(image)] for image in self.label_list_robust])
            if self.text_list else None
        )

        print(f'Train list: {self.label_list_robust}')
        return self.image_list_robust, self.label_list_robust, self.text_list_robust

    # 这里假设 build_all_layer_point_grids 函数已经定义
    # 若未定义，需要补充该函数的实现
    def build_all_layer_point_grids(n_per_side, n_layers, scale_per_layer):
        # 这里简单返回一个占位数组，实际需要根据具体逻辑实现
        return [np.random.rand(10, 2)]

class TestDataset:
    def __init__(self, image_path, label_path,  text_path=None, gt_path=None, is_robustness=False):
        """
        初始化测试数据集。

        Args:
            image_path (str): 图像数据路径。
            label_path (str): 标签数据路径。
            text_path (str, optional): 文本数据路径。如果为 None，则不加载文本。默认为 None。
            is_robustness (bool): 是否采用鲁棒性测试模式。
        """
        self.image_path = image_path
        self.label_path = label_path
        self.text_path = text_path
        self.gt_path = gt_path
        self.is_robustness = is_robustness

        # 获取文件列表
        if "SHHB" in gt_path or "QNRF" in gt_path or "NWPU" in gt_path:
            file_path = os.path.join(gt_path, "test.txt")  # 假设要读取的是test.txt，可按需修改
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.image_list = [line.strip() + '.jpg' for line in lines]
                self.label_list = [line.strip() + '.png' for line in lines]
        # delete_ids = {
        #     3143, 3150, 3155, 3181, 3187, 3188, 3246, 3259, 3272, 3280,
        #     3309, 3319, 3335, 3346, 3348, 3380, 3387, 3389, 3390, 3414,
        #     3422, 3423, 3432, 3446, 3447, 3482, 3487, 3503, 3509, 3528,
        #     3557, 3568, 3577, 3602, 3606
        # }
        if "NWPU" in gt_path:
            file_path = os.path.join(gt_path, "test.txt")  # 假设要读取的是test.txt，可按需修改
            with open(file_path, 'r') as f:
                lines = f.readlines()
                self.image_list = [line.strip().split()[0] + '.jpg' for line in lines]
                self.label_list = [line.strip().split()[0] + '.png' for line in lines]
                # for line in lines:
                #     parts = line.strip().split()
                #     # 确保至少有两列
                #     if len(parts) >= 2:
                #         try:
                #             # 检查第二列是否为0
                #             if int(parts[1]) == 0:
                #                 image_names.append(int(parts[0]))  # 添加第一列到列表中
                #         except ValueError:
                #             continue  # 忽略无法转换为整数的行

        # else:
        #     self.image_list = sorted(os.listdir(self.image_path))
        #     self.label_list = sorted(os.listdir(self.label_path))



        if self.gt_path:
            extractor = PointExtractor()
            self.gt_points_dict = extractor.extract_all_points_from_txt(self.gt_path+"/test_gt_loc.txt")
        else:
            self.gt_points_dict = None
        self.text_list = sorted(os.listdir(self.text_path)) if self.text_path else None

        if self.is_robustness:
            self.image_list, self.label_list, self.text_list = self.get_images_labels_and_texts_path_for_loop()

        # if "NWPU" in gt_path:

    def __getitem__(self, item):
        """
        获取单个数据项。

        Args:
            item (int): 索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[str], Tuple[torch.Tensor, torch.Tensor], List[List[int]]]:
                图像张量、标签张量、可选的文本嵌入、点提示 (点坐标, 点标签)、对应的gt点坐标。
        """
        # 加载图像
        image_name = self.image_list[item]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        resize_transform = transforms.Resize((1024, 1024))  # 指定目标尺寸
        image = resize_transform(image)
        image = transforms.ToTensor()(image)

        # 加载标签
        label_name = self.label_list[item]
        label = Image.open(os.path.join(self.label_path, label_name)).convert('L')
        original_width, original_height = label.size
        resize_transform = transforms.Resize((256, 256))  # 指定目标尺寸
        label = resize_transform(label)
        label_array = np.array(label)
        # 将所有非零像素值设置为 1
        label_array[label_array > 50] = 255
        # 使用 transforms.ToTensor() 将 numpy 数组转换为 Tensor，并将其标准化为 [0, 1]
        label_tensor = transforms.ToTensor()(Image.fromarray(label_array))
        label = label_tensor.long()

        # 加载文本（如果存在）
        text = None
        if self.text_list:
            text_name = self.text_list[item]
            with open(os.path.join(self.text_path, text_name), 'r') as f:
                text = f.read().strip()

        # 生成点提示
        points_scale = np.array(image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image, device='cuda')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda')
        points = (in_points, in_labels)

        # 获取对应的gt_points
        if self.gt_points_dict:
            image_id = os.path.splitext(image_name)[0]
            if image_id in self.gt_points_dict:
                total, points_list = self.gt_points_dict[image_id]
                if total==0:
                    gt_points = []
                else:
                    points_array = np.array(points_list)
                    points_array[:, 0] = (points_array[:, 0] * 256 / original_width).astype(int)
                    points_array[:, 1] = (points_array[:, 1] * 256 / original_height).astype(int)
                    gt_points = points_array.tolist()
            else:
                gt_points = []
        else:
            gt_points = []

        # # 把点画在 label 上
        # label_np = label.squeeze().cpu().numpy()
        # label_with_points = cv2.cvtColor((label_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # for x, y in gt_points:
        #     if 0 <= x < label_with_points.shape[1] and 0 <= y < label_with_points.shape[0]:
        #         cv2.circle(label_with_points, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        #
        # # 可视化
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(label_np, cmap='gray')
        # plt.title('Original Label')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(label_with_points, cv2.COLOR_BGR2RGB))
        # plt.title('Label with GT Points')
        # plt.show()

        # 返回内容
        if text is not None:
            return image, label,  text, gt_points
        else:
            return image, label, gt_points


    def __len__(self):
        """
        返回数据集长度。

        Returns:
            int: 数据集的样本数。
        """
        return len(self.image_list)



    def get_images_labels_and_texts_path_for_loop(self):
        """
        获取鲁棒性测试模式下的图像、标签和文本路径列表。

        Returns:
            Tuple[List[str], List[str], Optional[List[str]]]: 图像路径、标签路径、可选的文本路径列表。
        """
        self.label_list_robust = sorted([img for img in random.sample(self.label_list, 5)])
        self.image_list_robust = sorted([self.image_list[self.label_list.index(image)] for image in self.label_list_robust])
        self.text_list_robust = (
            sorted([self.text_list[self.label_list.index(image)] for image in self.label_list_robust])
            if self.text_list else None
        )

        print(f'Train list: {self.label_list_robust}')
        return self.image_list_robust, self.label_list_robust, self.text_list_robust

    # 这里假设 build_all_layer_point_grids 函数已经定义
    # 若未定义，需要补充该函数的实现
    def build_all_layer_point_grids(n_per_side, n_layers, scale_per_layer):
        # 这里简单返回一个占位数组，实际需要根据具体逻辑实现
        return [np.random.rand(10, 2)]


def natural_sort_key(s):
    import re
    # 提取文件名中的数字部分用于排序
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')




def get_imagse_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    print(mode, "data length:", len(label_list), len(image_list))

    return label_list, image_list

class CryopppDataset(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='random_click',
                 plane=False, iteration = -1, train_sample = []):

        self.train_list = train_sample
        self.valid_list = []
        self.args = args

        if iteration != -1:
            label_list, name_list = self.get_images_and_labels_path_for_loop(data_path, mode)
        else:
            label_list, name_list = get_images_and_labels_path(data_path, mode)

        self.original_size = (256, 256)
        self.target_length = 1024
        self.name_list = name_list
        self.label_list = label_list
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt  # or bboxes
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):

        inout = 1
        point_label = 1
        boxes = []
        box_old = []
        pt = np.array([0, 0])
        bboxes = []

        """Get the images"""
        name = self.name_list[index]
        # img_path = os.path.join(self.data_path, self.mode, "images", name)
        img_path = name

        mask_name = self.label_list[index]
        msk_path = mask_name

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'box':
            img_name = img_path.split('/')[-1]
            with open(os.path.join(self.data_path,"bbox.csv"),mode="r") as box_file:
                reader = csv.reader(box_file)
                for index, row in enumerate(reader):
                    if index != 0 and self.mode == row[0] and img_name == row[1]:
                        boxes = np.array([int(row[2]),int(row[3]),int(row[4]),int(row[5])])

            if boxes.any():
                boxes = boxes[None, :]
                boxes = self.apply_boxes(boxes, self.original_size)
                # box_torch = torch.as_tensor(boxes, dtype=torch.float, device="cuda")
                # boxes = box_torch[None, :]
                pass

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)

            torch.set_rng_state(state)

            if self.prompt == 'points_grids':
                point_grids = build_all_layer_point_grids(
                    n_per_side=32,
                    n_layers=0,
                    scale_per_layer=1,
                )
                points_scale = np.array(img.shape[1:])[None, ::-1]
                points_for_image = point_grids[0] * points_scale  # (1024 * 2)
                in_points = torch.as_tensor(points_for_image)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
                # points = (in_points, in_labels)
                pt = points_for_image
                point_label = np.array(in_labels)

            if self.transform_msk:
                mask = self.transform_msk(mask)

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'box': boxes,
            # 'box_old':box_old,
            'image_meta_dict': image_meta_dict,
            'ground_truth_bboxes': bboxes
        }

    def get_images_and_labels_path_for_loop(self, data_path, mode):

        if mode == 'train' or mode == "valid":
            label_list = sorted([os.path.join(data_path, "training_set", "labels", label_file) for label_file in
                                 os.listdir(os.path.join(data_path, "training_set", "labels"))])
            image_list = sorted([os.path.join(data_path, "training_set", "images", image_file) for image_file in
                                 os.listdir(os.path.join(data_path, "training_set", "images"))])

            if mode == 'train':
                label_train_list = sorted([img for img in random.sample(label_list, 5)])
                image_train_list = sorted([image_list[label_list.index(image)] for image in label_train_list])

                print(mode, "data length:", len(label_train_list), len(image_train_list))

                self.train_list = image_train_list

                print("train_dataset:")
                for i in range(len(self.train_list)):
                    print(self.train_list[i].split("/")[-1])

                return label_train_list, image_train_list

            elif mode == "valid":
                image_train_list = sorted([img for img in random.sample(image_list, 1) if img not in self.train_list])
                label_train_list = sorted([label_list[image_list.index(image)] for image in image_train_list])

                self.valid_list = image_train_list

                print("\nvalid_dataset:")
                for i in range(len(self.valid_list)):
                    print(self.valid_list[i].split("/")[-1])

                print(mode, "data length:", len(label_train_list), len(image_train_list))

                return label_train_list, image_train_list

        elif mode == "test":

            label_list = sorted([os.path.join(data_path, "testing_set", "labels", label_file) for label_file in
                                 os.listdir(os.path.join(data_path, "testing_set", "labels"))])
            image_list = sorted([os.path.join(data_path, "testing_set", "images", image_file) for image_file in
                                 os.listdir(os.path.join(data_path, "testing_set", "images"))])

            print(mode, "data length:", len(label_list), len(image_list))

            return label_list, image_list

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def get_images_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    print(mode, "data length:", len(label_list), len(image_list))

    return label_list, image_list
