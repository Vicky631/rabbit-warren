import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


class PointExtractor:
    def __init__(self):
        pass

    def extract_all_points_from_txt(self, txt_path):
        """
        从txt文件中批量提取所有图像的点坐标，封装到字典中
        :param txt_path: txt文件路径
        :return: 包含图像信息的字典，格式为{img_name: [total_count, points_list]}
        """
        result_dict = {}
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            img_name = parts[0]
            total_count = int(parts[1])
            coords = []
            for i in range(2, len(parts), 5):
                coords.extend(list(map(int, parts[i:i + 2])))
            if len(coords) % 2 != 0:
                raise ValueError(f"[{img_name}] 的坐标数据不是 (x, y) 对，数量: {len(coords)}")
            points = np.array(coords).reshape(-1, 2)
            result_dict[img_name] = [total_count, points.tolist()]
        return result_dict

    def resize_image_and_points(self, label_img_path, points):
        """
        将label图像和点坐标resize到256*256
        :param label_img_path: label图像路径
        :param points: 点坐标列表
        :return: 调整后的图像和点坐标
        """
        label_img = cv2.imread(label_img_path)
        if label_img is None:
            raise FileNotFoundError(f"❌ 无法读取图像: {label_img_path}")
        h, w = label_img.shape[:2]
        label_img_resized = cv2.resize(label_img, (256, 256))
        points = np.array(points)
        points[:, 0] = (points[:, 0] * 256 / w).astype(int)
        points[:, 1] = (points[:, 1] * 256 / h).astype(int)
        return label_img_resized, points.tolist()

    def draw_points_on_label(self, label_img_path, points):
        """
        在label图像上绘制红色点坐标
        :param label_img_path: label图像路径
        :param points: 点坐标列表
        :return: 绘制点后的图像
        """
        label_img = cv2.imread(label_img_path)
        if label_img is None:
            raise FileNotFoundError(f"❌ 无法读取图像: {label_img_path}")

        # 绘制红点
        for x, y in points:
            cv2.circle(label_img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

        return label_img

