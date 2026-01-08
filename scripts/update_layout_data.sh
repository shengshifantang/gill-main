#!/bin/bash
# 快速更新 Layout Planner 训练数据

set -e  # 遇到错误立即退出

echo "=========================================="
echo "📦 更新 Layout Planner 训练数据"
echo "=========================================="
echo ""

# 配置
LABELED_DATA="/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
OUTPUT_DATA="data/layout_planner_train.jsonl"
BACKUP_DATA="data/layout_planner_train_backup_$(date +%Y%m%d_%H%M%S).jsonl"

# 1. 检查标注数据是否存在
if [ ! -f "$LABELED_DATA" ]; then
    echo "❌ 错误: 标注数据文件不存在: $LABELED_DATA"
    exit 1
fi

# 2. 备份旧数据
if [ -f "$OUTPUT_DATA" ]; then
    echo "📦 备份旧数据..."
    cp "$OUTPUT_DATA" "$BACKUP_DATA"
    echo "✅ 已备份到: $BACKUP_DATA"
    echo ""
fi

# 3. 统计最新标注数据
echo "📊 统计最新标注数据..."
python3 -c "
import json

labeled_file = '$LABELED_DATA'
total = 0
with_objects = 0
no_objects = 0
errors = 0

with open(labeled_file, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
            objects = item.get('objects', [])
            num_obj = len(objects)
            
            if item.get('annotations_error') or item.get('error_type'):
                errors += 1
            elif num_obj > 0:
                with_objects += 1
            else:
                no_objects += 1
        except:
            pass

print(f'总数据量: {total:,} 条')
print(f'  - 有对象: {with_objects:,} 条 ({with_objects/total*100:.1f}%)')
print(f'  - 无对象: {no_objects:,} 条 ({no_objects/total*100:.1f}%)')
print(f'  - 标注错误: {errors:,} 条 ({errors/total*100:.1f}%)')
print(f'')
print(f'预计生成训练数据: {with_objects:,} 条')
"
echo ""

# 4. 生成新的训练数据
echo "🔄 生成新的训练数据..."
python scripts/generate_layout_training_data.py \
    --labeled "$LABELED_DATA" \
    --output "$OUTPUT_DATA" \
    --min-objects 1

echo ""

# 5. 验证新数据
echo "✅ 验证新数据..."
NEW_COUNT=$(wc -l < "$OUTPUT_DATA")
echo "新训练数据: $NEW_COUNT 条"

if [ -f "$BACKUP_DATA" ]; then
    OLD_COUNT=$(wc -l < "$BACKUP_DATA")
    INCREASE=$((NEW_COUNT - OLD_COUNT))
    INCREASE_PCT=$(python3 -c "print(f'{$INCREASE/$OLD_COUNT*100:.1f}')")
    echo "旧训练数据: $OLD_COUNT 条"
    echo "增量: $INCREASE 条 (+${INCREASE_PCT}%)"
fi

echo ""
echo "=========================================="
echo "💡 下一步建议"
echo "=========================================="

if [ -f "$BACKUP_DATA" ]; then
    OLD_COUNT=$(wc -l < "$BACKUP_DATA")
    INCREASE=$((NEW_COUNT - OLD_COUNT))
    INCREASE_PCT=$(python3 -c "print($INCREASE/$OLD_COUNT*100)")
    
    if (( $(echo "$INCREASE_PCT < 10" | bc -l) )); then
        echo "增量 < 10%，可以继续使用当前模型"
    elif (( $(echo "$INCREASE_PCT < 20" | bc -l) )); then
        echo "增量 10-20%，建议重新训练以获得更好效果"
    else
        echo "增量 > 20%，强烈建议重新训练！"
    fi
fi

echo ""
echo "重新训练命令:"
echo "CUDA_VISIBLE_DEVICES=2 python scripts/train_layout_planner.py \\"
echo "    --layout-json $OUTPUT_DATA \\"
echo "    --val-json data/coco-cn/coco-cn_val.jsonl \\"
echo "    --base-model ./model/qwen2.5-7B-Instruct \\"
echo "    --output-dir ./checkpoints/layout_planner \\"
echo "    --epochs 3 \\"
echo "    --batch-size 2 \\"
echo "    --gradient-accumulation-steps 4 \\"
echo "    --lr 1e-4 \\"
echo "    --use-format-metric"
echo ""
echo "=========================================="
