import os.path

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from transformers import CLIPTokenizer, CLIPModel
from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, slice(0, 1)]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        # src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        # 步骤1：对特征图进行聚/分类
        masks = upscaled_embedding
        # labels, category_embeddings = cluster_image(masks.detach(), 3)
        # 假设输入特征图的形状为 (2, 32, 256, 256)
        classifier = PixelClassifier(32).to(masks.device)
          # 预测像素分类
        try:
            import torch.nn.functional as F
            labels,category_embeddings = classifier.predict(masks)
          # 显示每个像素的类别标签（0, 1, 或 2）
        except Exception as e:
            print(e)
        # 步骤2：根据聚类标签生成带颜色的输出图像
        text = "people"
        # 加载 CLIP 模型
        clip_model = CLIPModel.from_pretrained("clip/clip-vit-large-patch14")
        clip_model.eval()  # 设置为评估模式

        # 加载 Tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("clip/clip-vit-large-patch14")

        # 生成文本嵌入
        text_embeddings = get_text_embeddings(text, clip_model, tokenizer)
        projection = torch.nn.Linear(768, 32)  # 创建一个线性层将768维映射到32维
        text_embeddings_mapped = projection(text_embeddings)
        # import torch.nn.functional as F
        #
        # similarities = [cosine_similarity(text_embeddings_mapped, category_embedding) for category_embedding in
        #                 category_embeddings]
        similarities = [cosine_similarity(text_embeddings_mapped, category_embedding) for category_embedding in
                        category_embeddings]

        # 找到与文本最相似的类别
        best_category_index = torch.argmax(torch.tensor(similarities))
        filtered_labels = (labels == best_category_index).long()

        # colored_image = generate_colored_image(filtered_labels.detach(), 3)
        colored_image = generate_colored_image(predicted_classes.clone(), 3)

        binary_image = convert_to_grayscale(colored_image)

        temp_dir = "/home/zy/wjj/Prompt_sam_localization/"
        masks_save_path = os.path.join(temp_dir, "masksfenlei")
        # visualize_and_save_with_cv2(binary_image, masks_save_path)
        maskss = binary_image.unsqueeze(1)
        return maskss, iou_pred
# 定义一个全卷积网络用于像素分类
class PixelClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super(PixelClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)  # 输出通道数为类别数

    class PixelClassifier(nn.Module):
        def __init__(self, in_channels, num_classes=3, feature_dim=128):
            super(PixelClassifier, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)  # 输出通道数为类别数
            self.num_classes = num_classes
            self.feature_dim = feature_dim  # 假设每个类别的中心向量的维度

            # 假设类别的特征向量是随机初始化的
            self.centroids = nn.Parameter(torch.randn(num_classes, self.feature_dim))  # 类别的中心向量，随机初始化

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)  # shape: (B, num_classes, H, W)
            return x

        def predict(self, x):
            # 获取每个像素的类别预测
            with torch.no_grad():
                logits = self.forward(x)  # shape: (B, num_classes, H, W)
                predicted_classes = torch.argmax(logits, dim=1)  # shape: (B, H, W)

                # 计算每个像素类别的平均特征向量
                # 根据预测的类别索引从 centroid 中获取对应的类别平均向量
                batch_size, _, height, width = logits.shape
                centroids_expanded = self.centroids[predicted_classes]  # (B, H, W, feature_dim)

                # 通过扩展维度，确保返回的 centroids 是 (B, C, H, W)
                centroids_expanded = centroids_expanded.permute(0, 3, 1, 2)  # shape: (B, feature_dim, H, W)

            return predicted_classes, centroids_expanded




def cosine_similarity(a, b):
            return F.cosine_similarity(a, b)

def get_text_embeddings(text,clip_model,tokenizer):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(clip_model.device)

    # 获取文本嵌入
    with torch.no_grad():
        clip_embeddings = clip_model.get_text_features(**inputs)
        clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)  # 归一化

    return clip_embeddings





import cv2
import os
import torch
import numpy as np

def visualize_and_save_with_cv2(mask: torch.Tensor, save_path: str):
    try:
     # Move the tensor to CPU and convert it to NumPy
     mask = mask.detach().cpu().numpy()  # Remove batch dimension and move to CPU

     # Get batch size from the mask
     batch_size = mask.shape[0]

     for i in range(batch_size):
         single_mask = mask[i]  # Extract each mask
         # Remove unnecessary dimensions if needed
         single_mask = single_mask.squeeze()

         # Normalize the mask
         min_value = np.min(single_mask)
         max_value = np.max(single_mask)

         # Directly normalize and clip
         normalized_mask = (single_mask - min_value) / (max_value - min_value)
         normalized_mask = np.clip(normalized_mask, 0, 1)  # Ensuring it's between [0, 1]

         # If the mask has 1 channel, save as a grayscale image
         # 如果是灰度图像（单通道），直接处理
         if single_mask.ndim == 2:  # 灰度图像
             color_map = (normalized_mask * 255).astype(np.uint8)
             color_map = np.expand_dims(color_map, axis=-1)  # (H, W) -> (H, W, 1)
         else:  # 多通道图像（如RGB）
             color_map = (normalized_mask * 255).astype(np.uint8)
             color_map = np.transpose(color_map, (1, 2, 0))  # (C, H, W) -> (H, W, C)
             color_map = color_map[..., ::-1]  # RGB -> BGR

         # Ensure save path exists
         if not os.path.exists(save_path):
             os.makedirs(save_path)

         # Save the image using OpenCV
         file_path = os.path.join(save_path, f"mask_{i}.png")
         if not cv2.imwrite(file_path, color_map):
             raise ValueError(f"Failed to save image at {file_path}")

    except Exception as e:
     # Print the error message and stop execution
     print(f"Error occurred: {e}")
     return


import numpy as np
import torch
from sklearn.cluster import KMeans
def cluster_image(feature_map, n_clusters=3):
    """
    对输入的特征图进行聚类，将每个像素分配到一个类别，并返回每个类别的嵌入
    :param feature_map: 输入特征图，形状为 (B, C, H, W)
    :param n_clusters: 聚类类别数，默认是3
    :return: 聚类标签图 (B, H, W) 和类别嵌入 (n_clusters, C)
    """
    B, C, H, W = feature_map.shape

    # 将特征图展平为 (B * H * W, C)，每行是一个像素点的特征（C维）
    feature_map_flat = feature_map.view(B, C, -1).permute(0, 2, 1).contiguous()
    feature_map_flat = feature_map_flat.view(-1, C)  # 现在形状为 (B * H * W, C)

    # 对特征图进行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_map_flat.cpu().numpy())  # 聚类标签
    cluster_centers = kmeans.cluster_centers_  # 聚类中心

    # labels 是一个长向量，形状为 (B * H * W,)
    labels = torch.tensor(labels).long().view(B, H, W)

    # 将类别嵌入转换为 Torch 张量
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)

    return labels, cluster_centers


def generate_colored_image(labels, n_classes=3):
    """
    根据聚类标签生成带颜色的图像
    :param labels: 聚类标签，形状为 (B, H, W)
    :param n_classes: 聚类类别数，默认是3
    :return: 带颜色的输出图像，形状为 (B, 3, H, W)
    """
    try:
        B, H, W = labels.shape

        # 定义n_classes种颜色（这里以3类为例）
        # 定义5种颜色（这里以5类为例）
        color_map = np.array([[1, 0, 0],  # 类1: 红色
                              [0, 1, 0],  # 类2: 绿色
                              [0, 0, 1],  # 类3: 蓝色
                              [1, 1, 0],  # 类4: 黄色
                              [1, 0, 1],  # 类5: 品红
                              [0, 1, 1],  # 类6: 青色
                              [1, 0.5, 0],  # 类7: 橙色
                              [0.5, 0, 0.5]])  # 类8: 紫色
        # 创建一个空的RGB图像，形状为 (B, 3, H, W)
        colored_image = np.zeros((B, 3, H, W), dtype=np.uint8)

        for b in range(B):
            for c in range(n_classes):
                # 找到当前类别的像素点
                mask = (labels[b] == c)  # mask 形状为 (H, W)

                # 使用np.where进行颜色填充
                colored_image[b, 0, :, :] = np.where(mask, color_map[c][0] * 255, colored_image[b, 0, :, :])  # 红色通道
                colored_image[b, 1, :, :] = np.where(mask, color_map[c][1] * 255, colored_image[b, 1, :, :])  # 绿色通道
                colored_image[b, 2, :, :] = np.where(mask, color_map[c][2] * 255, colored_image[b, 2, :, :])  # 蓝色通道

        # 确保返回的是 PyTorch 张量
        return torch.tensor(colored_image, dtype=torch.uint8).contiguous().detach()

    except Exception as e:
        # 捕获并输出错误信息
        print(f"Error occurred: {str(e)}")
        raise  # 可选: 重新抛出异常，以便进一步调试


def convert_to_grayscale(colored_image):
    """
    将 RGB 图像转换为灰度图
    :param colored_image: 输入的彩色图像，形状为 (B, 3, H, W)
    :return: 灰度图，形状为 (B, 1, H, W)
    """
    # 权重用于将 RGB 转换为灰度
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=colored_image.device)

    # 检查形状是否符合预期
    assert colored_image.ndim == 4 and colored_image.shape[1] == 3, \
        f"colored_image 应该是形状为 (B, 3, H, W)，但得到 {colored_image.shape}"
    assert weights.ndim == 1 and weights.shape[0] == 3, \
        f"weights 应该是长度为 3 的 1D 张量，但得到 {weights.shape}"

    # 确保 weights 和 colored_image 的设备一致
    weights = weights.to(colored_image.device)

    # 将 colored_image 转为 Float 类型
    colored_image = colored_image.float()

    # 调整 colored_image 的形状到 (B, H, W, 3)，然后计算灰度
    gray_image = torch.tensordot(colored_image.permute(0, 2, 3, 1), weights, dims=([-1], [0]))
    print("Gray Image Shape:", gray_image.shape)
    return gray_image

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
