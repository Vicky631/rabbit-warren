import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 路径到本地的 CLIP 模型
clip_path = "/home/zy/WJJ/SAML/Prompt_sam_localization/clip/clip-vit-large-patch14"

# 初始化 CLIP Tokenizer
try:
    tokenizer = CLIPTokenizer.from_pretrained(
        clip_path,
        local_files_only=True,
        torch_dtype=torch.float16
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# 输入文本提示
prompt = ['a dog wearing a hat']

# 对提示进行编码
try:
    tok = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
except Exception as e:
    print(f"Error tokenizing prompt: {e}")
    exit()

# 打印编码后的结果
print(f"Input IDs shape: {tok.input_ids.shape}")
print(tok)

# 打印前7个 token 及其对应的词汇
for token in list(tok.input_ids[0, :7]):
    print(f"{token.item()}: {tokenizer.convert_ids_to_tokens(int(token))}")

# 初始化 CLIP 文本编码器
try:
    text_encoder = CLIPTextModel.from_pretrained(
        clip_path,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to('cuda')
except Exception as e:
    print(f"Error loading text encoder: {e}")
    exit()

# 生成文本嵌入
try:
    with torch.no_grad():
        emb = text_encoder(tok.input_ids.to("cuda"))[0].half()
except Exception as e:
    print(f"Error during embedding generation: {e}")
    exit()

# 打印嵌入的形状及内容
print(f"Shape of embedding: {emb.shape}")
print(emb)
