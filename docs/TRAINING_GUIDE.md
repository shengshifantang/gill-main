# Spatial Adapter 训练指南

## 🎯 快速开始

### 单卡训练（推荐）

```bash
cd /home/lxh/Project/gill-main
bash scripts/train_single_gpu.sh
```

### 双卡训练

```bash
# 修改 train_single_gpu.sh 中的 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,2
# 然后使用 torchrun
```

---

## 📊 训练监控

### 实时查看 Gate 变化

```bash
tail -f checkpoints/spatial_adapter_zero_init_single_gpu/train.log | grep "Gate"
```

### 实时查看 Loss

```bash
tail -f checkpoints/spatial_adapter_zero_init_single_gpu/train.log | grep "Loss"
```

### TensorBoard 可视化

```bash
tensorboard --logdir ./outputs/tensorboard --port 6006
```

---

## ✅ 成功标准

| 指标 | 初始值 | 目标值 | 说明 |
|------|--------|--------|------|
| Gate Mean | 0.0 | 0.3-0.5 | 应该缓慢上升 |
| Loss | 0.15 | 0.10 | 应该持续下降 |
| Gate Std | 0.0 | 0.05-0.1 | 不同层的差异 |

---

## 🔧 核心文件

| 文件 | 说明 |
|------|------|
| `gill/spatial_adapter_fixed.py` | 核心实现（包含 Zero Init + LayerNorm） |
| `scripts/train_spatial_adapter.py` | 训练脚本（完善监控） |
| `scripts/train_single_gpu.sh` | 单卡训练启动脚本 |

---

## 📈 预期训练曲线

### Gate 变化

```
Step 0:     Gate Mean = 0.0000  (初始状态)
Step 1000:  Gate Mean = 0.0200  (开始学习)
Step 5000:  Gate Mean = 0.1500  (突变收敛)
Step 10000: Gate Mean = 0.3500  (稳定增长)
```

### Loss 变化

```
Step 0:     Loss = 0.15
Step 5000:  Loss = 0.12
Step 10000: Loss = 0.10
```

---

## ⚠️ 故障排查

### Gate 长期不增长（< 0.01）

**原因**：Phrase Embedding 质量差或数据清洗过于激进

**解决方案**：
```bash
# 降低 Phrase Dropout
--phrase-dropout 0.05

# 放宽数据清洗
--min-area 5e-5
--min-side 0.005
```

### 显存不足（OOM）

**解决方案**：
```bash
# 降低 Batch Size
--batch-size 1
```

---

## 📝 关键修复

1. ✅ **Zero Initialization**：`to_out` 层权重初始化为 0
2. ✅ **LayerNorm**：`text_proj` 添加归一化层
3. ✅ **Gate 监控**：每 100 步记录 mean/std/min/max
4. ✅ **数值稳定性**：dtype 对齐、梯度裁剪

---

## 📚 参考文档

- `docs/zero_init_fix_report.md` - 详细修复报告
- `docs/quick_start_after_fix.md` - 快速启动指南

---

**最后更新**: 2026-01-21

