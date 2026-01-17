# 🚀 Zero Init 修复后的快速启动指南

## ✅ 修复已完成

**修复内容**：在 `gill/spatial_adapter_fixed.py` 中添加了 Zero Initialization

**验证状态**：✅ 所有测试通过（运行 `scripts/test_zero_init.py` 验证）

---

## 🎯 立即开始重新训练

### 方案 1：完整训练（推荐，双卡）

```bash
cd /home/lxh/Project/gill-main

# 使用 GPU 0 和 2
bash scripts/train_spatial_adapter_zero_init.sh
```

**预期**：
- Gate 从 0 缓慢上升（0 → 0.01 → 0.05 → 0.1 → ...）
- Loss 持续下降
- 生成图像始终正常（不花屏）
- 控制力逐渐增强

---

### 方案 2：快速验证（单卡，1 epoch）

如果你想先快速验证修复是否有效：

```bash
cd /home/lxh/Project/gill-main

# 只用 GPU 0，训练 1 epoch
CUDA_VISIBLE_DEVICES=0 python scripts/train_spatial_adapter.py \
  --mixed-data ./data/coco2014_cn_train_clean.jsonl \
  --kolors-model ./model/Kolors \
  --output-dir ./checkpoints/spatial_adapter_zero_init_test \
  --image-dir ./data/coco2014/train2014 \
  --batch-size 2 \
  --epochs 1 \
  --lr 1e-4 \
  --phrase-dropout 0.1 \
  --save-epoch \
  --log-gate-stats \
  --log-tensorboard
```

**观察指标**：
1. Gate 是否从 0 开始？
2. Gate 是否在缓慢上升（而非下降）？
3. 生成的图像是否正常（不花屏）？

如果以上都是 ✅，说明修复成功！

---

## 📊 训练监控

### 1. 实时查看 Gate 变化

```bash
# 在另一个终端运行
tail -f checkpoints/spatial_adapter_zero_init_v1/train.log | grep "Gate"
```

**期望看到**：
```
Step 100: Gate stats - mean: 0.001, std: 0.0005, min: 0.0, max: 0.002
Step 200: Gate stats - mean: 0.005, std: 0.002, min: 0.001, max: 0.008
Step 300: Gate stats - mean: 0.012, std: 0.005, min: 0.003, max: 0.020
...
```

### 2. TensorBoard 可视化

```bash
tensorboard --logdir ./outputs/tensorboard --port 6006
```

**关键曲线**：
- `train/gate_mean`：应该**缓慢上升**
- `train/loss`：应该**持续下降**
- `train/lr`：学习率曲线

---

## 🧪 训练完成后的验证

### 1. Oracle 评估（使用 GT Box）

```bash
# 使用训练好的 checkpoint
bash scripts/test_gate_effect_single_gpu.sh
```

修改脚本中的 checkpoint 路径：
```bash
ADAPTER_CHECKPOINT="./checkpoints/spatial_adapter_zero_init_v1/epoch_10.pt"
```

### 2. 对比不同 Gate 值的效果

```bash
# 测试 Gate=0.5
python scripts/test_spatial_adapter.py \
  --adapter-checkpoint ./checkpoints/spatial_adapter_zero_init_v1/epoch_10.pt \
  --gate-scale 0.5 \
  --output-dir ./outputs/test_gate_0.5

# 测试 Gate=1.0
python scripts/test_spatial_adapter.py \
  --adapter-checkpoint ./checkpoints/spatial_adapter_zero_init_v1/epoch_10.pt \
  --gate-scale 1.0 \
  --output-dir ./outputs/test_gate_1.0

# 测试 Gate=2.0
python scripts/test_spatial_adapter.py \
  --adapter-checkpoint ./checkpoints/spatial_adapter_zero_init_v1/epoch_10.pt \
  --gate-scale 2.0 \
  --output-dir ./outputs/test_gate_2.0
```

**期望结果**：
- Gate=0.5：控制力中等，图像质量好
- Gate=1.0：控制力强，图像质量好
- Gate=2.0：控制力很强，图像质量仍然好（不花屏！）

---

## 🔍 故障排查

### 问题 1：Gate 仍然在下降

**可能原因**：
- Phrase Embedding 质量差
- 数据清洗过于激进

**解决方案**：
```bash
# 降低 Phrase Dropout
--phrase-dropout 0.05

# 放宽数据清洗阈值
--min-area 5e-5
--min-side 0.005
```

### 问题 2：Loss 不下降

**可能原因**：
- 学习率太小

**解决方案**：
```bash
# 增大学习率
--lr 5e-4
```

### 问题 3：生成图像仍然花屏

**可能原因**：
- Zero Init 没有生效（检查代码）

**验证方案**：
```bash
# 重新运行验证脚本
/home/lxh/.conda/envs/gill/bin/python scripts/test_zero_init.py
```

如果验证失败，说明代码修改有问题，需要重新检查。

---

## 📈 成功标准

### 训练过程

| 指标 | 初始 | 中期 | 后期 | 状态 |
|------|------|------|------|------|
| Gate Mean | 0.00 | 0.05-0.1 | 0.3-0.8 | ✅ 上升 |
| Loss | 0.15 | 0.12 | 0.10 | ✅ 下降 |
| 图像质量 | 正常 | 正常 | 正常 | ✅ 稳定 |

### Oracle 评估

| 指标 | 目标 | 说明 |
|------|------|------|
| IoU > 0.5 | > 70% | 位置准确性 |
| CLIP Score | > 0.25 | 语义一致性 |
| FID | < 30 | 图像质量 |

---

## 🎓 理论预期

### 训练曲线对比

**修复前（❌ 错误）**：
```
Gate: 3.0 → 2.5 → 2.0 → 1.5 → 1.0 → 0.5 → 0.29 (塌缩)
Loss: 0.15 → 0.14 → 0.13 → 0.12 (下降但 Adapter 被屏蔽)
```

**修复后（✅ 正确）**：
```
Gate: 0.0 → 0.01 → 0.05 → 0.1 → 0.3 → 0.5 → 0.8 (自然增长)
Loss: 0.15 → 0.13 → 0.11 → 0.09 (下降且 Adapter 有效)
```

---

## 🚨 重要提醒

### 1. 不要使用旧的 checkpoint

旧的 checkpoint 是用**未修复的代码**训练的，包含随机初始化的 `to_out` 权重。

**必须重新训练！**

### 2. 不要设置 `--gate-init-value`

现在 Gate 已经在代码中初始化为 0，不需要通过命令行参数设置。

### 3. 不要使用 `--freeze-gate`

让 Gate 自由学习，它会自然增长到合适的值。

---

## 📞 需要帮助？

如果遇到问题，请提供：

1. **训练日志**：`checkpoints/spatial_adapter_zero_init_v1/train.log`
2. **Gate 统计**：最近 100 步的 Gate mean/std/min/max
3. **Loss 曲线**：TensorBoard 截图
4. **生成样例**：保存几张生成的图像

---

## 🎉 预祝成功！

这次修复是**关键性的突破**！

如果训练成功，你将拥有：
- ✅ 首个支持中文的 Grounded T2I 模型
- ✅ 可发表的实验结果
- ✅ 完整的技术方案

**加油！这就是 SOTA 和废铁之间的那层窗户纸！**

---

**创建日期**: 2026-01-21  
**修复状态**: ✅ 完成  
**下一步**: 重新训练验证

