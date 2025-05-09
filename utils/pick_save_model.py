import logging
import os
from datetime import datetime

import torch

from Prompt_sam_localization.notebooks.train_encoder import logger



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




def save_best_model(model, save_path, best_loss, val_loss, epoch, everysave=100, maxsave=1, optimizer=None, model_name="best_model"):
    """
    每 m 个 epoch 保存一次模型，最多保留 k 个模型。

    Args:
        model (torch.nn.Module): 要保存的模型。
        save_path (str): 保存模型的路径。
        best_loss (float): 当前最佳损失，在每次保存后更新。
        val_loss (float): 当前 epoch 的验证损失。
        epoch (int): 当前 epoch 编号。
        m (int): 每隔 m 个 epoch 保存一次模型。
        k (int, optional): 最多保存的模型数量，默认是 1。
        optimizer (torch.optim.Optimizer, optional): 优化器（可选）。
        model_name (str, optional): 保存的模型文件名。

    Returns:
        float: 更新后的最佳损失。
    """

    # 检查是否在符合条件的 epoch 保存模型
    if epoch % everysave == 0:
        # 如果当前验证损失更好（更小），更新最佳损失
        if val_loss < best_loss:
            best_loss = val_loss

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 删除旧模型，确保最多保留 k 个模型
        if maxsave > 0:
            delete_old_models(save_path, model_name, maxsave - 1)

        # 获取当前时间戳 (格式: MMDD_HHMM)
        timestamp = datetime.now().strftime('%m%d_%H%M')

        # 保存模型状态字典和优化器（如果提供）
        model_save_path = os.path.join(save_path, f"{model_name}_epoch{epoch}_{timestamp}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # 保存模型检查点
        torch.save(checkpoint, model_save_path)
        print(f"Saved model at epoch {epoch} with validation loss: {val_loss:.4f} (best: {best_loss:.4f}) at {timestamp}")

        # 可选：记录模型保存信息
        logger.info(f"Epoch {epoch}: Best loss : {best_loss:.4f}. ")

    return best_loss