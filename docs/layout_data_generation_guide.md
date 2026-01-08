# Layout Planner 训练数据生成指南

## 📊 数据来源

当前训练数据 `data/layout_planner_train.jsonl` 是从标注数据中提取的：

- **原始标注数据**: `/mnt/disk/lxh/gill_data/wukong_labeled.jsonl`
- **当前训练数据**: 60,000 条（全部为 Layout 数据）

## 🔄 更新训练数据

当你的标注进度更新后，使用以下命令重新生成训练数据：

### 方法 1：生成纯 Layout 训练数据（推荐）

```bash
python scripts/generate_layout_training_data.py \
    --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \
    --output data/layout_planner_train.jsonl
```

**特点**：
- 提取所有有对象的标注数据
- 自动过滤标注错误的数据
- 适合训练纯 Layout Planner

### 方法 2：限制对象数量

如果你想避免过于复杂的场景（对象太多），可以限制对象数量：

```bash
python scripts/generate_layout_training_data.py \
    --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \
    --output data/layout_planner_train.jsonl \
    --min-objects 1 \
    --max-objects 10
```

**特点**：
- 只保留对象数在 1-10 之间的数据
- 避免过于复杂的场景（可能导致训练不稳定）

### 方法 3：生成混合训练数据（Layout + 通用）

如果你想训练一个既能做 Layout 又能做通用图像理解的模型：

```bash
python scripts/prepare_mixed_training_data.py \
    --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \
    --unlabeled /mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl \
    --output data/mixed_train_20pct.jsonl \
    --layout-ratio 0.2 \
    --total-size 100000
```

**特点**：
- 混合 20% Layout 数据 + 80% 通用数据
- 总数据量 100,000 条
- 适合训练多任务模型

## 📈 数据统计

### 当前训练数据（旧）

```
文件: data/layout_planner_train.jsonl
总数据量: 60,000 条
  - Layout 数据: 60,000 条 (100%)
  - 通用数据: 0 条 (0%)
```

### 最新标注数据（新）

运行以下命令查看最新统计：

```bash
python3 -c "
import json

labeled_file = '/mnt/disk/lxh/gill_data/wukong_labeled.jsonl'
total = 0
with_objects = 0

with open(labeled_file, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
            if len(item.get('objects', [])) > 0:
                if not item.get('annotations_error') and not item.get('error_type'):
                    with_objects += 1
        except:
            pass

print(f'总数据量: {total:,} 条')
print(f'可用 Layout 数据: {with_objects:,} 条')
print(f'增量: {with_objects - 60000:,} 条')
"
```

## 🚀 重新训练流程

### 1. 生成新的训练数据

```bash
# 备份旧数据
mv data/layout_planner_train.jsonl data/layout_planner_train_old.jsonl

# 生成新数据
python scripts/generate_layout_training_data.py \
    --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \
    --output data/layout_planner_train.jsonl
```

### 2. 检查数据质量

```bash
# 查看前几条数据
head -3 data/layout_planner_train.jsonl | python -m json.tool

# 统计数据量
wc -l data/layout_planner_train.jsonl
```

### 3. 开始训练

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_layout_planner.py \
    --layout-json data/layout_planner_train.jsonl \
    --val-json data/coco-cn/coco-cn_val.jsonl \
    --base-model ./model/qwen2.5-7B-Instruct \
    --output-dir ./checkpoints/layout_planner \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --lr 1e-4 \
    --use-format-metric \
    --format-eval-samples 50
```

## 💡 最佳实践

### 数据质量优先

1. **过滤标注错误**：默认会过滤 `annotations_error` 和 `error_type` 字段
2. **限制对象数量**：建议 `--max-objects 10`，避免过于复杂的场景
3. **定期更新**：每次标注进度更新后，重新生成训练数据

### 训练策略

1. **首次训练**：使用所有可用数据，观察效果
2. **增量训练**：如果新增数据较少（< 10%），可以考虑从上次的 checkpoint 继续训练
3. **格式优先**：使用 `--use-format-metric` 确保格式正确性

### 数据增量

如果你的标注数据从 60,000 增加到 80,000：

- **增量 < 20%**：可以考虑增量训练（从上次 checkpoint 继续）
- **增量 > 20%**：建议重新训练（从头开始）

## 📝 数据格式

训练数据格式（JSONL）：

```json
{
  "image_path": "/path/to/image.jpg",
  "caption": "图像描述文本",
  "objects": [
    {"name": "对象名", "bbox": [x1, y1, x2, y2]},
    {"name": "对象名", "bbox": [x1, y1, x2, y2]}
  ],
  "width": 1000,
  "height": 667,
  "has_layout": true
}
```

## 🔧 脚本参数说明

### generate_layout_training_data.py

- `--labeled`: 已标注数据文件路径（必需）
- `--output`: 输出训练数据文件路径（必需）
- `--min-objects`: 最少对象数（默认 1）
- `--max-objects`: 最多对象数（默认无限制）
- `--no-filter-errors`: 不过滤标注错误的数据

### prepare_mixed_training_data.py

- `--labeled`: 已标注数据文件路径（必需）
- `--unlabeled`: 未标注数据文件路径（必需）
- `--output`: 输出混合数据文件路径（必需）
- `--layout-ratio`: Layout 数据占比（默认 0.2）
- `--total-size`: 总数据量（默认使用所有 Layout 数据）
- `--seed`: 随机种子（默认 42）

## ❓ 常见问题

### Q: 如何知道我的标注数据增加了多少？

```bash
# 统计最新标注数据
python3 -c "
import json
with open('/mnt/disk/lxh/gill_data/wukong_labeled.jsonl', 'r') as f:
    count = sum(1 for line in f if line.strip() and len(json.loads(line).get('objects', [])) > 0)
print(f'可用 Layout 数据: {count:,} 条')
"

# 统计当前训练数据
wc -l data/layout_planner_train.jsonl
```

### Q: 是否需要重新训练？

- **增量 < 10%**：可以继续使用当前模型
- **增量 10-20%**：建议重新训练，效果会有提升
- **增量 > 20%**：强烈建议重新训练

### Q: 如何验证新数据的质量？

```bash
# 查看前几条数据
head -5 data/layout_planner_train.jsonl | python -m json.tool

# 统计对象数量分布
python3 -c "
import json
from collections import Counter
with open('data/layout_planner_train.jsonl', 'r') as f:
    counts = [len(json.loads(line).get('objects', [])) for line in f if line.strip()]
dist = Counter(counts)
for k in sorted(dist.keys())[:10]:
    print(f'{k} 个对象: {dist[k]:,} 条')
"
```
