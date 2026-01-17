#!/bin/bash
# 完整训练 Spatial Adapter（Zero Init + FP16 修复版）
# 预计训练时间：约 18-20 小时（单卡 RTX 4090）

set -e

# ==================== 配置 ====================
MIXED_DATA="./data/coco2014_cn_train_clean.jsonl"
KOLORS_MODEL="./model/Kolors"
OUTPUT_DIR="./checkpoints/spatial_adapter_fp16_fixed"
IMAGE_DIR="./data/coco2014/train2014"

# 训练参数
BATCH_SIZE=2              # 单卡 batch size
EPOCHS=1                  # 1 个 epoch ≈ 41k 步，足够观察 Gate 行为
LR=1e-4                   # 学习率
PHRASE_DROPOUT=0.1        # Phrase embedding dropout（增强鲁棒性）
SCALE_MIN=0.5             # Adapter 注入强度范围
SCALE_MAX=1.0
SAVE_EVERY=2000           # 每 2000 步保存一次

# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# ==================== 训练信息 ====================
echo "=========================================="
echo "完整训练 Spatial Adapter"
echo "=========================================="
echo "配置："
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - 学习率: $LR"
echo "  - 保存间隔: 每 $SAVE_EVERY 步"
echo ""
echo "关键修复："
echo "  ✅ Zero Initialization（Gate 从 0 开始）"
echo "  ✅ Adapter/Phrase Embeddings 统一 FP16"
echo "  ✅ VAE 保持 FP32（数值稳定性）"
echo ""
echo "预计时间："
echo "  - 总步数: ~41,000 步"
echo "  - 训练时间: ~18-20 小时"
echo "  - Checkpoint: 每 2000 步保存"
echo ""
echo "监控指标："
echo "  1. Loss 下降趋势（目标: 0.02-0.03）"
echo "  2. Gate 统计（期望: 0 → 0.3-0.5）"
echo "  3. 生成质量（每个 checkpoint 测试）"
echo ""
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录训练配置
cat > "$OUTPUT_DIR/train_config.txt" << EOF
训练配置
========================================
时间: $(date)
数据: $MIXED_DATA
模型: $KOLORS_MODEL
输出: $OUTPUT_DIR

参数:
  - Batch Size: $BATCH_SIZE
  - Epochs: $EPOCHS
  - Learning Rate: $LR
  - Phrase Dropout: $PHRASE_DROPOUT
  - Scale Range: [$SCALE_MIN, $SCALE_MAX]
  - Save Every: $SAVE_EVERY steps

修复:
  - Zero Initialization: ✅
  - FP16 Adapter: ✅
  - FP16 Phrase Embeddings: ✅
  - FP32 VAE: ✅
========================================
EOF

echo "开始训练..."
echo ""

# 启动训练
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
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "下一步："
echo "  1. 检查训练日志: cat $OUTPUT_DIR/train.log"
echo "  2. 测试 checkpoint: bash scripts/test_checkpoint.sh"
echo "  3. 对比不同 Gate 值的效果"
echo ""

