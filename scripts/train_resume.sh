#!/bin/bash
# 恢复训练脚本（从 checkpoint 继续）

set -e

# ==================== 配置 ====================
MIXED_DATA="./data/coco2014_cn_train_clean.jsonl"
KOLORS_MODEL="./model/Kolors"
OUTPUT_DIR="./checkpoints/spatial_adapter_fp16_fixed"
IMAGE_DIR="./data/coco2014/train2014"

# 训练参数（与初始训练保持一致）
BATCH_SIZE=2
EPOCHS=1
LR=1e-4
PHRASE_DROPOUT=0.1
SCALE_MIN=0.5
SCALE_MAX=1.0
SAVE_EVERY=2000

# 恢复点（自动查找最新的 checkpoint）
RESUME_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/checkpoint-*.pt 2>/dev/null | head -1)

# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# ==================== 检查 ====================
if [ -z "$RESUME_CHECKPOINT" ]; then
    echo "❌ 错误：未找到 checkpoint"
    echo "请检查目录: $OUTPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "恢复训练 Spatial Adapter"
echo "=========================================="
echo "恢复点: $RESUME_CHECKPOINT"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "开始训练..."
echo ""

# 启动训练（添加 --resume 参数）
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
  --save-every $SAVE_EVERY \
  --resume "$RESUME_CHECKPOINT" \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

