# Prompt SAM Localization

本项目主要用于基于 SAM 的目标定位/计数实验，包含数据读取、SAM adapter/prompt 训练、验证测试、可视化以及若干历史实验脚本。项目代码主要放在 `notebooks/` 目录下。

## 项目路径

服务器上的项目根目录按以下路径使用：

```bash
/home/zy/wjj/Prompt_sam_localization/
```

如果脚本中出现本地调试路径，迁移到服务器时统一替换为上述路径。

## 目录说明

```text
/home/zy/wjj/Prompt_sam_localization/
├── notebooks/
│   ├── 0404_adapter.py          # SAM adapter 主要训练/测试脚本
│   ├── 0409_NWPU.py             # NWPU 数据相关训练/测试脚本
│   ├── dataset.py               # TrainDataset / ValDataset / TestDataset / CryopppDataset
│   ├── train_encoder.py         # encoder 训练入口
│   ├── train_prefix.py          # token/prefix prompt 训练入口
│   ├── function.py              # SAM adapter 训练、验证辅助函数
│   ├── function_token.py        # token prompt 训练、验证辅助函数
│   ├── RegionMAELoss.py         # 区域 MAE 损失
│   ├── GDFN.py                  # 网络模块
│   ├── unet_model*.py           # UNet baseline 相关模型
│   └── SAM_conf/
│       ├── SAM_cfg.py           # 训练参数配置
│       ├── SAM_utils.py         # 日志、checkpoint 等工具函数
│       └── global_settings.py   # 全局路径与训练设置
├── segment_anything/            # SAM 及 adapter 相关代码
├── dinov2/                      # DINOv2 相关依赖/代码
└── utils/                       # 通用工具函数
```

## 主要算法位置

- SAM adapter 主模型与训练流程：`notebooks/0404_adapter.py`
- NWPU 数据训练与测试流程：`notebooks/0409_NWPU.py`
- 数据集加载与点提示生成：`notebooks/dataset.py`
- encoder 训练：`notebooks/train_encoder.py`
- prefix/token prompt 训练：`notebooks/train_prefix.py`
- 训练与验证公共函数：`notebooks/function.py`、`notebooks/function_token.py`
- 参数配置：`notebooks/SAM_conf/SAM_cfg.py`

## 环境依赖

建议使用 Python 3.8+ 和 CUDA 环境。主要依赖包括：

```text
torch
torchvision
numpy
opencv-python
scipy
matplotlib
tqdm
```

还需要准备 SAM checkpoint，例如 `sam_vit_h_4b8939.pth`，并保证 `segment_anything` 可以被 Python 正确导入。

## 数据格式

常用数据目录形式如下：

```text
data_root/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

部分脚本也会直接读取：

```text
data_root/
├── images/
├── labels/
└── groundtruth/
```

具体以运行脚本中 `TrainDataset`、`ValDataset`、`TestDataset` 的传参为准。

## 运行示例

进入项目根目录：

```bash
cd /home/zy/wjj/Prompt_sam_localization/
export PYTHONPATH=/home/zy/wjj/Prompt_sam_localization/:/home/zy/wjj/Prompt_sam_localization/segment_anything:$PYTHONPATH
```

训练 SAM adapter：

```bash
python notebooks/0404_adapter.py \
  -mode train \
  -data_path /path/to/dataset \
  -data_name dataset_name \
  -exp_name exp_adapter \
  -sam_ckpt /path/to/sam_vit_h_4b8939.pth \
  -save_path /path/to/save \
  -bs 1 \
  -epochs 100 \
  -lr 0.00005
```

测试模型：

```bash
python notebooks/0404_adapter.py \
  -mode test \
  -data_path /path/to/dataset \
  -data_name dataset_name \
  -exp_name exp_adapter \
  -sam_ckpt /path/to/sam_vit_h_4b8939.pth \
  -ckpt /path/to/best_model.pth \
  -save_path /path/to/save
```

NWPU 相关实验可使用：

```bash
python notebooks/0409_NWPU.py \
  -mode train \
  -data_path /path/to/NWPU \
  -data_name NWPU \
  -exp_name exp_nwpu \
  -sam_ckpt /path/to/sam_vit_h_4b8939.pth \
  -save_path /path/to/save
```

## 输出说明

训练和测试结果通常保存在 `-save_path` 指定目录下，主要包括：

- `model/`：模型权重
- `log/`：训练日志
- `train_sample/`、`val_sample/`、`test_sample/`：可视化结果
- `evaluation_*.txt`：测试指标结果

常用指标包括 loss、IoU、Dice、Precision、Recall、F1、MAE、MSE 等。

## 注意事项

- 运行前确认 `-data_path`、`-sam_ckpt`、`-ckpt` 和 `-save_path` 路径存在。
- 脚本中部分历史实验路径需要按服务器环境替换为 `/home/zy/wjj/Prompt_sam_localization/`。
- 若导入 `segment_anything` 失败，先检查 `PYTHONPATH` 是否包含项目根目录和 `segment_anything` 目录。
- 不同脚本的数据目录组织略有差异，报错时优先检查 `notebooks/dataset.py` 中对应 Dataset 的读取逻辑。
