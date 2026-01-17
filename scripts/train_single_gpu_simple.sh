#!/bin/bash
# 简化版单卡训练脚本（只使用支持的参数）

set -e

# 配置
MIXED_DATA="./data/coco2014_cn_train_clean.jsonl"
KOLORS_MODEL="./model/Kolors"
OUTPUT_DIR="./checkpoints/spatial_adapter_zero_init_single_gpu"
IMAGE_DIR="./data/coco2014/train2014"

# 训练参数
BATCH_SIZE=2
EPOCHS=10
LR=1e-4
PHRASE_DROPOUT=0.1
SCALE_MIN=0.5
SCALE_MAX=1.0
SAVE_EVERY=2000

echo "=========================================="
echo "单卡训练 Spatial Adapter（Zero Init + FP16 修复版）"
echo "=========================================="
echo "GPU: 单张卡（CUDA_VISIBLE_DEVICES=0）"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""
echo "关键修复："
echo "  ✅ Zero Initialization 已修复"
echo "  ✅ Adapter/Phrase Embeddings 统一使用 FP16"
echo "  ✅ Gate 从 0 开始自然学习"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 单卡训练
export CUDA_VISIBLE_DEVICES=0

echo "开始训练..."
echo ""

python scripts/train_spatial_adapter.py \
  --mixed-data "$MIXED_DATA" \
  --kolors-model "$KOLORS_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --image-dir "$IMAGE_DIR" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --phrase-dropout $PHRASE_DROPOUT \
  --scale-min $SCALE_MIN \
  --scale-max $SCALE_MAX \
  --save-every $SAVE_EVERY

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "模型保存在: $OUTPUT_DIR"
echo ""
