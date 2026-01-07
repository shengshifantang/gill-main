#!/bin/bash
# 完整的训练执行脚本

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🚀 开始训练流程"
echo "============================================================"
echo ""

# 配置
GPU_ID=2
MIXED_DATA="data/mixed_train_20pct.jsonl"
KOLORS_MODEL="model/Kolors"  # Kolors 模型在项目 model 目录下
OUTPUT_DIR_LAYOUT="checkpoints/layout_planner"
OUTPUT_DIR_ADAPTER="checkpoints/spatial_adapter_20pct"

# Step 0: 检查图像文件
echo "Step 0: 检查图像文件..."
python3 << 'EOF'
import json
import os

mixed_data = "data/mixed_train_20pct.jsonl"
sample_count = 0
existing = 0

with open(mixed_data, 'r') as f:
    for line in f:
        if sample_count >= 1000:
            break
        item = json.loads(line)
        path = item.get('image_path', '')
        if path:
            sample_count += 1
            if os.path.exists(path):
                existing += 1

rate = existing / sample_count * 100 if sample_count > 0 else 0
print(f"  检查了 {sample_count} 个样本")
print(f"  图像存在率: {rate:.1f}%")

if rate < 90:
    print("  ❌ 图像存在率 < 90%，需要先下载图像")
    exit(1)
else:
    print("  ✅ 图像文件充足，可以开始训练")
EOF

if [ $? -ne 0 ]; then
    echo "❌ 图像文件检查失败，请先下载图像"
    exit 1
fi

echo ""

# Step 1: 准备 Layout Planner 训练数据
echo "Step 1: 准备 Layout Planner 训练数据..."
python3 << 'EOF'
import json
import os

input_file = "data/mixed_train_20pct.jsonl"
output_file = "data/layout_planner_train.jsonl"

layout_data = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            if item.get('has_layout', False) and item.get('objects'):
                layout_data.append(item)

os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
with open(output_file, 'w') as f:
    for item in layout_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"  ✅ 提取了 {len(layout_data)} 条 Layout 数据")
print(f"  ✅ 保存到: {output_file}")
EOF

echo ""

# Step 2: 检查 GPU
echo "Step 2: 检查 GPU $GPU_ID..."
if ! nvidia-smi -i $GPU_ID &> /dev/null; then
    echo "  ❌ GPU $GPU_ID 不可用"
    exit 1
fi
echo "  ✅ GPU $GPU_ID 可用"
echo ""

# Step 3: 训练 Layout Planner
echo "============================================================"
echo "Step 3: 训练 Layout Planner（GPU $GPU_ID）"
echo "============================================================"
echo ""

python scripts/train_layout_planner.py \
    --train-data data/layout_planner_train.jsonl \
    --val-data data/layout_planner_train.jsonl \
    --device cuda:$GPU_ID \
    --output-dir $OUTPUT_DIR_LAYOUT \
    --batch-size 8 \
    --epochs 5 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "❌ Layout Planner 训练失败"
    exit 1
fi

echo ""
echo "✅ Layout Planner 训练完成"
echo ""

# Step 4: 训练 Spatial Adapter
echo "============================================================"
echo "Step 4: 训练 Spatial Adapter（GPU $GPU_ID）"
echo "============================================================"
echo ""

# 检查 Kolors 模型路径
if [ ! -d "$KOLORS_MODEL" ]; then
    echo "⚠️  警告: Kolors 模型路径不存在: $KOLORS_MODEL"
    echo "   请修改脚本中的 KOLORS_MODEL 变量"
    read -p "   是否继续？(y/n): " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

python scripts/train_spatial_adapter.py \
    --mixed-data $MIXED_DATA \
    --kolors-model $KOLORS_MODEL \
    --device cuda:$GPU_ID \
    --output-dir $OUTPUT_DIR_ADAPTER \
    --batch-size 2 \
    --epochs 5 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "❌ Spatial Adapter 训练失败"
    exit 1
fi

echo ""
echo "✅ Spatial Adapter 训练完成"
echo ""

# Step 5: 验证结果
echo "============================================================"
echo "Step 5: 验证训练结果"
echo "============================================================"
echo ""

echo "检查模型文件:"
if [ -d "$OUTPUT_DIR_LAYOUT" ]; then
    echo "  ✅ Layout Planner: $OUTPUT_DIR_LAYOUT"
    ls -lh $OUTPUT_DIR_LAYOUT/*.pth 2>/dev/null | head -3 || echo "     (未找到 .pth 文件)"
else
    echo "  ❌ Layout Planner 目录不存在"
fi

if [ -d "$OUTPUT_DIR_ADAPTER" ]; then
    echo "  ✅ Spatial Adapter: $OUTPUT_DIR_ADAPTER"
    ls -lh $OUTPUT_DIR_ADAPTER/*.pt 2>/dev/null | head -3 || echo "     (未找到 .pt 文件)"
else
    echo "  ❌ Spatial Adapter 目录不存在"
fi

echo ""
echo "============================================================"
echo "🎉 训练流程完成！"
echo "============================================================"
echo ""
echo "下一步:"
echo "  1. 检查模型文件是否生成"
echo "  2. 进行端到端测试（可选）"
echo "  3. 开始 Baseline 对比实验（Phase 2）"
echo ""
