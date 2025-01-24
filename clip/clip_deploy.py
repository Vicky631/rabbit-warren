import torch
from transformers import CLIPTokenizer, CLIPModel


class CLIPTextEmbedder:
    """
    CLIP 文本嵌入生成器类。

    使用指定的 CLIP 模型生成文本嵌入。

    Attributes:
        clip_model (CLIPModel): CLIP 模型实例。
        tokenizer (CLIPTokenizer): CLIP 模型的 Tokenizer 实例。
        device (torch.device): 模型运行设备（CPU 或 CUDA）。
    """

    def __init__(self, clip_model_path: str, device: torch.device = None):
        """
        初始化 CLIP 文本嵌入生成器。

        Args:
            clip_model_path (str): CLIP 模型的路径。
            device (torch.device, optional): 模型运行设备（默认为 None，自动检测 GPU）。
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # 加载 CLIP 模型和 Tokenizer
            self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading CLIP model or tokenizer: {e}")

    def generate_embeddings(self, texts: list) -> torch.Tensor:
        """
        生成文本嵌入。

        Args:
            texts (list[str]): 输入文本列表，例如 ["a photo of a cat", "a dog running in a park"]。

        Returns:
            torch.Tensor: 文本嵌入，形状为 (batch_size, embedding_dim)。
        """
        try:
            # 编码文本
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",  # 返回 PyTorch 张量
                padding=True,  # 对输入进行填充，确保批次长度一致
                truncation=True  # 截断超长文本，避免超出模型最大序列长度
            ).to(self.device)  # 将输入移动到指定设备 (CPU 或 GPU)
        except Exception as e:
            raise RuntimeError(f"Error during tokenization: {e}")

        try:
            # 生成文本嵌入
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)  # 使用 CLIP 模型生成文本嵌入
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化嵌入
        except Exception as e:
            raise RuntimeError(f"Error generating text embeddings: {e}")

        return text_features


# 示例用法
if __name__ == "__main__":
    clip_model_path = "/home/zy/WJJ/SAML/Prompt_sam_localization/clip/clip-vit-large-patch14"
    texts = ["a photo of a cat", "a dog running in a park", "a beautiful sunset over the ocean"]

    # 创建 CLIP 文本嵌入生成器实例
    try:
        clip_embedder = CLIPTextEmbedder(clip_model_path)
    except Exception as e:
        print(f"Error initializing CLIPTextEmbedder: {e}")
        exit()

    # 生成文本嵌入
    try:
        embeddings = clip_embedder.generate_embeddings(texts)
        print(f"Text embeddings shape: {embeddings.shape}")
        print(f"Text embeddings: {embeddings}")
    except Exception as e:
        print(f"An error occurred: {e}")
