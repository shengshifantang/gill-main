#!/bin/bash
# 在 GPU 2 上训练脚本示例

# 设置 GPU 设备
GPU_ID=2

echo "============================================================"
echo "🚀 在 GPU $GPU_ID 上训练"
echo "============================================================"
echo ""

# 检查 GPU 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: nvidia-smi 未找到"
    exit 1
fi

# 显示 GPU 使用情况
echo "📊 当前 GPU 使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "GPU %s: %s - 显存: %s/%s - 利用率: %s\n", $1, $2, $3, $4, $5}'
echo ""

# 检查 GPU 2 是否可用
if ! nvidia-smi -i $GPU_ID &> /dev/null; then
    echo "❌ 错误: GPU $GPU_ID 不可用"
    exit 1
fi

echo "✅ GPU $GPU_ID 可用"
echo ""

# 示例1: Layout Planner 训练
echo "============================================================"
echo "示例1: Layout Planner 训练"
echo "============================================================"
echo ""
echo "python scripts/train_layout_planner.py \\"
echo "    --train-data data/splits/train.jsonl \\"
echo "    --val-data data/splits/val.jsonl \\"
echo "    --device cuda:$GPU_ID \\"
echo "    --output-dir checkpoints/layout_planner_gpu$GPU_ID \\"
echo "    --batch-size 8 \\"
echo "    --epochs 5"
echo ""

# 示例2: Spatial Adapter 训练
echo "============================================================"
echo "示例2: Spatial Adapter 训练"
echo "============================================================"
echo ""
echo "python scripts/train_spatial_adapter.py \\"
echo "    --mixed-data data/mixed_train_20pct.jsonl \\"
echo "    --kolors-model model/Kolors \\"
echo "    --device cuda:$GPU_ID \\"
echo "    --output-dir checkpoints/spatial_adapter_gpu$GPU_ID \\"
echo "    --batch-size 2 \\"
echo "    --epochs 5"
echo ""

echo "============================================================"
echo "💡 提示"
echo "============================================================"
echo "1. 卡0和卡1正在标注，不会与卡2冲突"
echo "2. 单卡训练时，建议减小 batch-size 以避免 OOM"
echo "3. 可以使用 'nvidia-smi -l 1' 监控 GPU 使用情况"
echo "============================================================"
