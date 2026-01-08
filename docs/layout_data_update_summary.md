# Layout Planner 训练数据更新总结

## 🎉 好消息！你的标注数据大幅增长

### 数据对比

| 项目 | 旧数据 | 新数据 | 增长 |
|------|--------|--------|------|
| **总数据量** | 60,000 条 | 244,977 条 | +308% |
| **可用 Layout 数据** | 60,000 条 | 198,551 条 | +231% |
| **增量** | - | +138,551 条 | **3.3倍** |

### 数据质量

- ✅ 有对象: 198,551 条 (81.0%)
- ⚪ 无对象: 40,730 条 (16.6%)
- ❌ 标注错误: 5,696 条 (2.3%)

### 对象数量分布

```
1 个对象:  54,810 条 (27.6%)
2 个对象:  47,529 条 (23.9%)
3 个对象:  37,471 条 (18.9%)
4 个对象:  26,711 条 (13.5%)
5 个对象:  14,798 条 (7.5%)
6+ 个对象: 17,232 条 (8.7%)
```

## 🚀 快速更新流程

### 一键更新（推荐）

```bash
bash scripts/update_layout_data.sh
```

这个脚本会自动：
1. 备份旧数据
2. 统计最新标注数据
3. 生成新的训练数据
4. 验证数据质量
5. 给出训练建议

### 手动更新

```bash
# 1. 备份旧数据
cp data/layout_planner_train.jsonl data/layout_planner_train_old.jsonl

# 2. 生成新数据
python scripts/generate_layout_training_data.py \
    --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \
    --output data/layout_planner_train.jsonl

# 3. 验证
wc -l data/layout_planner_train.jsonl
```

## 💡 训练建议

由于数据增长了 **3.3 倍**，强烈建议重新训练！

### 推荐训练配置

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

### 预期效果

- **训练时间**: 约 3-4 小时（取决于 GPU）
- **数据量**: 198,551 条（比之前多 3.3 倍）
- **预期提升**: 
  - 格式正确率: 预计从 82% 提升到 90%+
  - 泛化能力: 显著提升（数据更丰富）
  - 复杂场景: 更好处理多对象场景

## 📊 数据质量分析

### 优势

1. **数据量充足**: 198,551 条，远超一般微调需求
2. **分布合理**: 1-5 个对象占 91.3%，符合实际场景
3. **错误率低**: 仅 2.3% 标注错误，质量很高

### 建议

1. **限制对象数量**: 可以考虑 `--max-objects 10`，避免过于复杂的场景
2. **分批训练**: 如果显存不足，可以先用 10 万条数据训练，再用全量数据
3. **监控格式正确率**: 使用 `--use-format-metric` 确保格式正确性

## 🔄 增量训练 vs 重新训练

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **重新训练** | 充分利用新数据，效果最好 | 耗时较长 | 数据增长 > 20%（推荐） |
| **增量训练** | 节省时间，保留旧知识 | 可能欠拟合新数据 | 数据增长 < 10% |

**你的情况**: 数据增长 231%，**强烈建议重新训练**！

## 📝 相关文档

- [训练数据生成指南](layout_data_generation_guide.md)
- [训练指南：过拟合 vs 格式正确性](layout_planner_training_guide.md)

## ❓ 常见问题

### Q: 为什么不使用全部 244,977 条数据？

A: 因为其中包含：
- 40,730 条无对象数据（不适合 Layout Planner）
- 5,696 条标注错误数据（会影响训练质量）

只使用 198,551 条有对象且无错误的数据，确保训练质量。

### Q: 是否需要调整训练参数？

A: 数据量增加后，建议：
- 保持 `--epochs 3`（数据多了，不需要太多 epoch）
- 可以适当增加 `--batch-size`（如果显存允许）
- 保持 `--lr 1e-4`（这是 LoRA 的标准学习率）

### Q: 如何验证新模型的效果？

A: 训练完成后，使用验证脚本：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/verify_layout.py \
    --base-model ./model/qwen2.5-7B-Instruct \
    --adapter-path ./checkpoints/layout_planner/final \
    --device cuda:0
```

观察格式正确率和坐标准确性。
