# COCO-CN 数据集构建指南

## 📋 概述

COCO-CN 是 COCO 数据集的中文版本，包含人工翻译和人工编写的中文标注。

## 📁 文件结构

### 原始数据（在 `/mnt/disk/lxh/gill_data`）

```
/mnt/disk/lxh/gill_data/
├── train2014/                    # 训练集图片
│   ├── COCO_train2014_000000296735.jpg
│   └── ...
├── val2014/                      # 验证集图片
│   ├── COCO_val2014_000000043734.jpg
│   └── ...
└── annotations_trainval2014/
    └── annotations/
        ├── captions_train2014.json
        ├── captions_val2014.json
        └── ...
```

### COCO-CN 中文标注（在 `data/coco-cn`）

```
data/coco-cn/
├── coco-cn_train.txt             # 训练集图片ID列表
├── coco-cn_val.txt               # 验证集图片ID列表
├── coco-cn_test.txt              # 测试集图片ID列表
├── imageid.manually-translated-caption.txt    # 人工翻译的caption
├── imageid.human-written-caption.txt          # 人工编写的caption
└── ...
```

## 🚀 构建数据集

### 🏆 推荐方案：使用 human-written（默认）

**针对 GILL-Next-CN（中文文本生成布局 + 图像生成）项目，强烈推荐使用 `human-written`：**

```bash
python scripts/build_coco_cn_dataset.py \
    --coco-cn-dir data/coco-cn \
    --coco-images-dir /mnt/disk/lxh/gill_data \
    --coco-annotations-dir /mnt/disk/lxh/gill_data/annotations_trainval2014/annotations \
    --output-dir /mnt/disk/lxh/gill_data/coco-cn \
    --caption-type human-written \
    --include-spatial
```

**推荐理由：**
1. **数据量优势**：22,218 条 vs 5,000 条（4.4 倍），提供更强的梯度信号
2. **语言自然度**：中文母语者直接撰写，更符合真实用户 Prompt
3. **泛化能力**：多样化的中文词汇映射到固定类别，增强模型鲁棒性
4. **默认选项**：脚本默认使用 `human-written`

**注意**：`--include-spatial` 是默认选项，可以省略。

### 参数说明

- `--coco-cn-dir`: COCO-CN 标注文件目录（默认: `data/coco-cn`）
- `--coco-images-dir`: COCO 图片目录（包含 `train2014` 和 `val2014` 子目录）
- `--coco-annotations-dir`: COCO 标注文件目录（包含 `captions_*.json`）
- `--output-dir`: 输出 JSONL 文件目录
- `--caption-type`: 使用的 caption 类型（默认: `human-written`）
  - `human-written`: 人工编写的 caption（**推荐，22K 数据**）
  - `manually-translated`: 人工翻译的 caption（5K 数据，与英文对齐更好）
- `--splits`: 要构建的数据集划分（默认: `train val test`）

### 💡 最佳实践：混合使用两种 caption 类型

**同一图片 + 不同描述 = 天然的数据增强！**

#### 步骤 1：生成 human-written 版本（主力）

```bash
python scripts/build_coco_cn_dataset.py \
    --coco-cn-dir data/coco-cn \
    --coco-images-dir /mnt/disk/lxh/gill_data \
    --coco-annotations-dir /mnt/disk/lxh/gill_data/annotations_trainval2014/annotations \
    --output-dir /mnt/disk/lxh/gill_data/coco-cn-human \
    --caption-type human-written
```

#### 步骤 2：生成 manually-translated 版本（辅助）

```bash
python scripts/build_coco_cn_dataset.py \
    --coco-cn-dir data/coco-cn \
    --coco-images-dir /mnt/disk/lxh/gill_data \
    --coco-annotations-dir /mnt/disk/lxh/gill_data/annotations_trainval2014/annotations \
    --output-dir /mnt/disk/lxh/gill_data/coco-cn-translated \
    --caption-type manually-translated
```

#### 步骤 3：合并数据

使用合并脚本自动合并两种类型的数据：

```bash
python scripts/merge_coco_cn_captions.py \
    --translated-file /mnt/disk/lxh/gill_data/coco-cn-translated/coco-cn_train.jsonl \
    --human-file /mnt/disk/lxh/gill_data/coco-cn-human/coco-cn_train.jsonl \
    --output-file /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train_merged.jsonl
```

**或者使用命令行简单合并：**

```bash
cat /mnt/disk/lxh/gill_data/coco-cn-translated/coco-cn_train.jsonl \
    /mnt/disk/lxh/gill_data/coco-cn-human/coco-cn_train.jsonl \
    > /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train_merged.jsonl
```

**注意**：不需要去重！同一图片的不同描述是两条不同的训练样本，能显著提升模型鲁棒性。

### 只构建训练集

```bash
python scripts/build_coco_cn_dataset.py \
    --coco-cn-dir data/coco-cn \
    --coco-images-dir /mnt/disk/lxh/gill_data \
    --coco-annotations-dir /mnt/disk/lxh/gill_data/annotations_trainval2014/annotations \
    --output-dir /mnt/disk/lxh/gill_data/coco-cn \
    --caption-type manually-translated \
    --splits train
```

## 📝 输出格式

### 包含空间标注的格式（默认）

```json
{
    "image_path": "/mnt/disk/lxh/gill_data/train2014/COCO_train2014_000000296735.jpg",
    "caption": "机舱内有蓝色和黑色相间的条纹座椅。",
    "image_id": "COCO_train2014_000000296735",
    "coco_image_id": 296735,
    "width": 640,
    "height": 480,
    "objects": [
        {
            "name": "椅子",
            "bbox": [100, 200, 300, 400],
            "category_id": 62
        },
        {
            "name": "椅子",
            "bbox": [150, 250, 350, 450],
            "category_id": 62
        }
    ]
}
```

**字段说明：**
- `image_path`: 图片的完整路径
- `caption`: 中文标注（来自 COCO-CN）
- `image_id`: COCO 图片ID（格式：`COCO_train2014_000000296735`）
- `coco_image_id`: COCO 原始图片ID（数字）
- `width`, `height`: 图片尺寸（像素）
- `objects`: 对象列表（来自 COCO instances 标注）
  - `name`: 对象类别名称（中文，如 "人", "汽车", "椅子" 等），与 COCO-CN 的中文 caption 保持一致，也与标注脚本输出格式一致
  - `bbox`: 边界框 `[x1, y1, x2, y2]`，范围 0-1000
  - `category_id`: COCO 类别ID（1-80），可选字段，不影响训练

### 仅文本格式（使用 `--no-spatial`）

```json
{
    "image_path": "/mnt/disk/lxh/gill_data/train2014/COCO_train2014_000000296735.jpg",
    "caption": "机舱内有蓝色和黑色相间的条纹座椅。",
    "image_id": "COCO_train2014_000000296735",
    "coco_image_id": 296735,
    "width": 640,
    "height": 480
}
```

## 🎯 关于空间标注

### COCO-CN 项目本身
- **COCO-CN 项目只提供中文 caption**，没有空间标注信息
- 文件包括：`imageid.manually-translated-caption.txt`、`imageid.human-written-caption.txt` 等

### 原始 COCO 数据集
- **原始 COCO 数据集包含完整的空间标注**：
  - `instances_train2014.json` / `instances_val2014.json`
  - 包含边界框（bbox）、分割掩码（segmentation）、80 个对象类别
  - 格式：`[x, y, width, height]`（像素坐标）

### 本脚本的处理
- **默认会合并 COCO 的空间标注**：
  - 通过匹配 `image_id` 将 COCO instances 标注与 COCO-CN caption 合并
  - 边界框自动转换为 0-1000 范围（与标注脚本兼容）
  - 格式：`[x1, y1, x2, y2]`（0-1000 范围）

### 使用选项
- **包含空间标注**（默认）：
```bash
python scripts/build_coco_cn_dataset.py ... --include-spatial
```

- **不包含空间标注**（仅文本）：
```bash
python scripts/build_coco_cn_dataset.py ... --no-spatial
```

## 🔍 验证数据集

构建完成后，可以验证数据集：

```bash
# 检查文件大小和记录数
wc -l /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train.jsonl
wc -l /mnt/disk/lxh/gill_data/coco-cn/coco-cn_val.jsonl

# 查看第一条记录
head -1 /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train.jsonl | python3 -m json.tool
```

## 📊 数据集统计

构建完成后，脚本会显示统计信息：

- 成功构建的记录数
- 缺失图片的数量
- 缺失标注的数量

## 🎯 后续使用

### 1. 如果已包含空间标注（使用 `--include-spatial`）

构建的 JSONL 文件**已经包含边界框信息**，可以直接用于训练：

```bash
python scripts/train_spatial_adapter.py \
    --train_jsonl /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train.jsonl \
    ...
```

**注意**：COCO 的空间标注已经包含在数据中，通常不需要再次标注。

### 2. 如果需要使用 LLM 重新标注或增强

如果希望使用 LLM 对 caption 进行空间布局规划或增强，可以运行标注脚本：

```bash
python scripts/annotate_async_vllm.py \
    --input /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train.jsonl \
    --image-root /mnt/disk/lxh/gill_data \
    --output /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train_labeled.jsonl \
    ...
```

**注意**：如果输入文件已包含 `objects` 字段，标注脚本会保留这些信息，并可能添加 LLM 生成的空间布局规划。

## 📊 Caption 类型对比

| 特性 | manually-translated | human-written |
|------|---------------------|---------------|
| **生成方式** | 英文标注 → 中文翻译 | 直接看图 → 中文描述 |
| **数据量** | 5,000 条 | 22,218 条（**4.4 倍**） |
| **语言自然度** | 可能带翻译腔 | **更自然的中文表达** |
| **与英文对齐** | 与 COCO 英文标注一一对应 | 独立撰写，视角可能不同 |
| **推荐场景** | 需要与英文版本对比 | **中文文生图（推荐）** |

### 选择建议

- **中文文生图项目（GILL-Next-CN）**：推荐 `human-written`
  - 数据量更多（22K vs 5K）
  - 更自然的中文表达
  - 增强模型对真实用户 Prompt 的理解

- **需要与英文对齐**：使用 `manually-translated`
  - 与原始 COCO 数据集对齐更好
  - 适合对比研究

- **追求极致效果**：混合使用两种类型
  - 同一图片 + 不同描述 = 天然数据增强
  - 获得两种标注的优势

## ⚠️ 注意事项

1. **图片路径**：确保 `--coco-images-dir` 包含 `train2014` 和 `val2014` 子目录
2. **标注文件**：确保 COCO-CN 标注文件在 `--coco-cn-dir` 目录下
3. **Caption 类型**：
   - `human-written`（默认）：22K 数据，更自然的中文，**推荐用于中文文生图**
   - `manually-translated`：5K 数据，与英文对齐更好
4. **数据量**：COCO-CN 数据集相对较小（约 2 万张图片），适合作为补充数据
5. **混合使用**：可以分别构建两种类型的数据，然后合并使用，获得数据增强效果

## 📚 参考

- COCO 数据集：https://cocodataset.org/
- COCO-CN 项目：https://github.com/li-xirong/coco-cn

