#!/bin/bash
# 快速查看标注进度

INPUT_FILE="/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
OUTPUT_FILE="/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
ERROR_FILE="/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"

echo "============================================================"
echo "📊 标注进度统计"
echo "============================================================"
echo ""

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 输入文件不存在: $INPUT_FILE"
    exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "⚠️  输出文件不存在，标注可能还未开始"
    exit 0
fi

# 统计行数
COMPLETED=$(wc -l < "$OUTPUT_FILE")
TOTAL=$(wc -l < "$INPUT_FILE")

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ 输入文件为空"
    exit 1
fi

# 计算进度
PERCENTAGE=$(python3 -c "print(f'{($COMPLETED/$TOTAL*100):.2f}')" 2>/dev/null || echo "0.00")
REMAINING=$((TOTAL - COMPLETED))

# 统计错误数（排除已成功的数据）
ERROR_COUNT=0
REAL_ERROR_COUNT=0
if [ -f "$ERROR_FILE" ]; then
    ERROR_COUNT=$(wc -l < "$ERROR_FILE")
    
    # 使用 Python 检查实际需要重试的错误数（排除已成功的数据）
    if [ -f "$OUTPUT_FILE" ] && [ "$ERROR_COUNT" -gt 0 ]; then
        REAL_ERROR_COUNT=$(python3 << 'PYEOF'
import json
import os

error_file = "/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
output_file = "/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
image_root = "/mnt/disk/lxh/gill_data/images"

# 读取已成功的数据
successful_paths = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                image_path = data.get('image_path', '')
                if image_path:
                    if not os.path.isabs(image_path):
                        full_path = os.path.join(image_root, image_path)
                    else:
                        full_path = image_path
                    successful_paths.add(os.path.normpath(full_path))
            except:
                pass

# 统计实际需要重试的错误数
real_error_count = 0
if os.path.exists(error_file):
    with open(error_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                error_entry = json.loads(line.strip())
                image_path = error_entry.get('image_path', '')
                if image_path:
                    if not os.path.isabs(image_path):
                        full_path = os.path.join(image_root, image_path)
                    else:
                        full_path = image_path
                    normalized_path = os.path.normpath(full_path)
                    if normalized_path not in successful_paths:
                        real_error_count += 1
            except:
                pass

print(real_error_count)
PYEOF
        )
    else
        REAL_ERROR_COUNT=$ERROR_COUNT
    fi
fi

# 文件大小和修改时间
FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
LAST_MODIFIED=$(stat -c %y "$OUTPUT_FILE" | cut -d'.' -f1)

echo "📈 进度信息："
echo "   已完成: $COMPLETED 条"
echo "   总计:   $TOTAL 条"
echo "   进度:   ${PERCENTAGE}%"
echo "   剩余:   $REMAINING 条"
echo ""

if [ "$ERROR_COUNT" -gt 0 ]; then
    if [ "$REAL_ERROR_COUNT" -lt "$ERROR_COUNT" ]; then
        echo "⚠️  错误记录: $ERROR_COUNT 条（实际需重试: $REAL_ERROR_COUNT 条，已清理 $((ERROR_COUNT - REAL_ERROR_COUNT)) 条已成功的数据）"
    else
    echo "⚠️  错误记录: $ERROR_COUNT 条"
    fi
    echo ""
fi

echo "📁 文件信息："
echo "   文件大小: $FILE_SIZE"
echo "   最后更新: $LAST_MODIFIED"
echo ""

# 检查进程是否在运行
if pgrep -f "annotate_async_vllm" > /dev/null; then
    echo "✅ 标注进程正在运行"
else
    echo "⚠️  标注进程未运行"
fi

echo ""
echo "============================================================"
echo "💡 查看实时进度：tmux attach -t 0 或 tmux a"
echo "============================================================"

