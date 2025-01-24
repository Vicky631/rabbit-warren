
import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
from .common import LayerNorm2d

from transformers import CLIPTokenizer, CLIPModel




class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        text_embedding_dim: int = 768,  # 新增文本嵌入维度
        clip_model_path: str = "./clip/clip-vit-large-patch14"
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 现有的嵌入器
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # 初始化 CLIP 文本编码器
        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = nn.Linear(text_embedding_dim, embed_dim)  # 映射到 embed_dim 大小


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding


    def _embed_text(self, texts: list) -> torch.Tensor:
        """
        使用 CLIP 模型嵌入文本。
        """
        tokenizer = CLIPTokenizer.from_pretrained("./clip/clip-vit-large-patch14")
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.clip_model.device)

        with torch.no_grad():
            clip_embeddings = self.clip_model.get_text_features(**inputs)
            clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)  # 归一化
            return self.text_encoder(clip_embeddings)  # 映射到 embed_dim

    def _get_batch_size(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
            text: Optional[torch.Tensor] = None  # 添加 text 参数
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.

        Arguments:
            points (Optional[Tuple[torch.Tensor, torch.Tensor]]): 点的坐标和标签。
            boxes (Optional[torch.Tensor]): 边界框。
            masks (Optional[torch.Tensor]): 掩码。
            text (Optional[torch.Tensor]): 文本嵌入。

        Returns:
            int: Batch size 推断结果。
        """
        if points is not None:
            return points[0].shape[0]  # 根据点坐标的批次大小确定
        elif boxes is not None:
            return boxes.shape[0]  # 根据边界框的批次大小确定
        elif masks is not None:
            return masks.shape[0]  # 根据掩码的批次大小确定
        elif text is not None:
            return text.shape[0]  # 根据文本嵌入的批次大小确定
        else:
            return 1  # 默认值

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
            text: Optional[torch.Tensor] = None,  # 新增文本输入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates and labels to embed.
          boxes (torch.Tensor or none): boxes to embed.
          masks (torch.Tensor or none): masks to embed.
          text (torch.Tensor or none): text embeddings to encode.

        Returns:
          torch.Tensor: sparse embeddings for the points, boxes, and text.
          torch.Tensor: dense embeddings for the masks.
        """
        bs = self._get_batch_size(points, boxes, masks)

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # 新增：处理文本嵌入
        if text is not None:
            text_embeddings = self._embed_text(text)  # 将文本嵌入映射到 embed_dim
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings.unsqueeze(1)], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
