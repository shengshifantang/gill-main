#!/bin/bash
# 单卡训练 Spatial Adapter（Zero Init 修复版）
# GPU: 单张 24GB 显卡

set -e

# 配置
MIXED_DATA="./data/coco2014_cn_train_clean.jsonl"
KOLORS_MODEL="./model/Kolors"
OUTPUT_DIR="./checkpoints/spatial_adapter_zero_init_single_gpu"
IMAGE_DIR="./data/coco2014/train2014"

# 训练参数（单卡优化）
BATCH_SIZE=2          # 单卡降低 batch size
EPOCHS=10
LR=1e-4
PHRASE_DROPOUT=0.1
SCALE_MIN=0.5
SCALE_MAX=1.0

# 显存优化
VAE_ENCODE_DEVICE="cuda"
VAE_ENCODE_FP16="--vae-encode-fp16"
AMP_DTYPE="fp16"

# 日志
LOG_GATE_STATS="--log-gate-stats"
LOG_GATE_INTERVAL=100
LOG_LOSS_INTERVAL=50
LOG_TENSORBOARD="--log-tensorboard"
TB_LOGDIR="./outputs/tensorboard"

# 保存策略
SAVE_EVERY=2000
SAVE_EPOCH="--save-epoch"
SAVE_STATE="--save-state"

# 数据过滤
MIN_AREA=1e-4
MIN_SIDE=0.01

echo "=========================================="
echo "单卡训练 Spatial Adapter（Zero Init 修复版）"
echo "=========================================="
echo "GPU: 单张卡（CUDA_VISIBLE_DEVICES=0）"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""
echo "关键修复："
echo "  ✅ Zero Initialization 已修复"
echo "  ✅ Gate 从 0 开始自然学习"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TB_LOGDIR"

# 单卡训练（只用 GPU 0）
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
  --save-every $SAVE_EVERY \
  $SAVE_EPOCH \
  $SAVE_STATE \
  --min-area $MIN_AREA \
  --min-side $MIN_SIDE \
  --vae-encode-device $VAE_ENCODE_DEVICE \
  $VAE_ENCODE_FP16 \
  --amp-dtype $AMP_DTYPE \
  $LOG_GATE_STATS \
  --log-gate-interval $LOG_GATE_INTERVAL \
  --log-loss-interval $LOG_LOSS_INTERVAL \
  $LOG_TENSORBOARD \
  --tb-logdir "$TB_LOGDIR"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "模型保存在: $OUTPUT_DIR"
echo ""
echo "查看 Gate 变化："
echo "  grep 'Gate' $OUTPUT_DIR/train.log"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir $TB_LOGDIR --port 6006"
echo ""

