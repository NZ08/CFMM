# CFMM: 多模态推荐系统

## 项目简介

CFMM (Cross-Fusion Activated Multi-Modal) 是一个多模态推荐系统，通过交叉融合机制整合图像和文本特征，提升推荐效果。

### 核心特性

- **交叉注意力机制**: 实现图像和文本特征的深度融合
- **多模态特征处理**: 支持图像和文本的联合建模

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy, SciPy, tqdm

### 安装依赖

```bash
pip install torch torchvision torchaudio numpy scipy tqdm sentence-transformers
```

### 数据准备

```bash
cd data
python build_data.py
```

### 训练模型

```bash
cd codes
python main.py --dataset baby --cf_model Att_lightgcn --lr 0.0005 --embed_size 64
```

### 主要参数

- `--dataset`: 数据集选择 (baby/sports/clothing)
- `--cf_model`: 模型类型 (Att_lightgcn)
- `--lr`: 学习率 (默认: 0.0005)
- `--embed_size`: 嵌入维度 (默认: 64)
- `--batch_size`: 批次大小 (默认: 1024)

## 实验效果

### 数据集

| 数据集 | 用户数 | 物品数 | 交互数 |
|--------|--------|--------|--------|
| Baby   | 19,445 | 7,050  | 139,110|
| Sports | 35,598 | 18,357 | 296,337|
| Clothing| 39,387| 23,033 | 278,677|


相比基线方法平均提升 3% 以上。

## 项目结构

```
CFMM/
├── codes/
│   ├── main.py          # 主训练脚本
│   ├── Models.py        # 模型定义
│   ├── modules.py       # 核心模块
│   └── utility/         # 工具函数
└── data/                # 数据处理
```