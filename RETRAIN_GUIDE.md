# 🚀 重新训练指南

## ✅ 已完成的修复

1. **诊断问题**：Gate 和 to_out 参数全部为 0（训练失败）
2. **修复代码**：`gill/spatial_adapter_fixed.py` 中 Gate 初始化从 0.0 改为 0.1
3. **清理文件**：删除了临时调试文件和失败的 checkpoint

## 🎯 开始训练

在终端中运行：

```bash
cd /home/lxh/Project/gill-main
conda activate qwen_py310

# 确保旧的 checkpoint 已删除
rm -rf checkpoints/spatial_adapter_fp16_fixed/

# 开始训练
python scripts/train_spatial_adapter.py \
  --mixed-data "./data/coco2014_cn_train_clean.jsonl" \
  --kolors-model "./model/Kolors" \
  --output-dir "./checkpoints/spatial_adapter_fp16_fixed" \
  --image-dir "./data/coco2014/train2014" \
  --batch-size 2 \
  --epochs 10 \
  --lr 1e-4 \
  --phrase-dropout 0.1 \
  --scale-min 0.5 \
  --scale-max 1.0 \
  --save-every 500
```

## 📊 验证训练

训练 500 步后，检查参数是否更新：

```bash
python scripts/quick_check.py ./checkpoints/spatial_adapter_fp16_fixed/checkpoint-500.pt
```

期望输出：
```
✅ Gate 已更新，训练正常
```

## 📁 保留的文件

- `scripts/validate_training_fixed.py` - 修复版验证脚本
- `scripts/quick_check.py` - 快速检查工具
- `gill/spatial_adapter_fixed.py.bak` - 原文件备份

## 🗑️ 已删除的文件

- 临时调试脚本
- 失败的 checkpoint
- 失败的验证输出
