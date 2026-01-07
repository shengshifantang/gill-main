#!/bin/bash
# 重跑标注失败的数据
# 从错误日志中提取失败的数据，重新运行标注任务

# 配置参数（与 run_annotation_tmux.sh 保持一致）
INPUT_FILE="/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
IMAGE_ROOT="/mnt/disk/lxh/gill_data/images"
OUTPUT_FILE="/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
ERROR_FILE="/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
RETRY_INPUT_FILE="/mnt/disk/lxh/gill_data/wukong_retry_failed.jsonl"
MODEL_NAME="/mnt/disk/lxh/models/Qwen2.5-VL-32B-Instruct-AWQ"
API_BASE="http://localhost:8000/v1"
MAX_CONCURRENCY=32
SESSION_NAME="annotation_retry"
CONDA_ENV="gill_chinese"

echo "============================================================"
echo "🔄 重跑标注失败的数据"
echo "============================================================"
echo ""

# 检查错误日志文件是否存在
if [ ! -f "$ERROR_FILE" ]; then
    echo "❌ 错误日志文件不存在: $ERROR_FILE"
    echo ""
    echo "请先运行标注任务，失败的数据会被记录到错误日志中。"
    exit 1
fi

# 检查原始输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 原始输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 统计错误数量
ERROR_COUNT=$(wc -l < "$ERROR_FILE" 2>/dev/null || echo 0)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ 没有失败的数据需要重跑"
    exit 0
fi

echo "📊 发现 $ERROR_COUNT 条失败记录"
echo ""

# 检查 / 启动 vLLM 服务
echo "🔍 检查 vLLM 服务状态..."
if curl -s "$API_BASE/models" > /dev/null 2>&1; then
    echo "✅ vLLM 服务已在运行"
    SKIP_VLLM=true
else
    echo "❌ vLLM 服务未运行"
    SKIP_VLLM=false

    echo ""
    read -p "是否现在启动 vLLM 服务？(y/n，默认 y): " start_choice
    start_choice=${start_choice:-y}

    if [[ ! $start_choice =~ ^[Yy]$ ]]; then
        echo "❌ vLLM 服务未运行，无法继续重跑失败数据"
        exit 1
    fi
fi

echo ""

if [ "$SKIP_VLLM" = false ]; then
    echo "============================================================"
    echo "步骤 1: 启动 vLLM 服务端"
    echo "============================================================"
    echo ""

    # 检查是否已有 vllm tmux 会话
    if tmux has-session -t vllm 2>/dev/null; then
        echo "⚠️  发现已存在的 vllm 会话"
        read -p "是否杀死旧会话并重新启动？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            tmux kill-session -t vllm
            sleep 1
        else
            echo "使用现有会话"
            SKIP_VLLM=true
        fi
    fi

    if [ "$SKIP_VLLM" = false ]; then
        echo "📦 在 tmux 会话 'vllm' 中启动服务端..."
        tmux new-session -d -s vllm 'bash scripts/start_vllm_server.sh'

        echo ""
        echo "⏳ 等待服务启动（这可能需要 1-2 分钟）..."
        echo "   正在检查服务状态..."

        # 等待服务启动（最多等待 3 分钟）
        MAX_WAIT=180
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "$API_BASE/models" > /dev/null 2>&1; then
                echo ""
                echo "✅ vLLM 服务已就绪！"
                break
            fi
            sleep 5
            WAITED=$((WAITED + 5))
            if [ $((WAITED % 30)) -eq 0 ]; then
                echo "   等待中... (已等待 ${WAITED} 秒)"
            fi
        done

        if [ $WAITED -ge $MAX_WAIT ]; then
            echo ""
            echo "⚠️  警告: 服务启动超时，将继续尝试重跑标注任务"
            echo "   如果失败，请手动检查 vLLM 服务:"
            echo "     tmux attach -t vllm"
        fi
    fi
else
    echo "✅ 跳过 vLLM 服务启动（服务已在运行）"
fi

echo ""

# 再次检查服务（确保服务可用）
if ! curl -s "$API_BASE/models" > /dev/null 2>&1; then
    echo "❌ vLLM 服务未响应，无法启动重试标注任务"
    echo ""
    echo "请检查服务状态:"
    echo "  tmux attach -t vllm"
    echo "  或"
    echo "  curl $API_BASE/models"
    echo ""
    exit 1
fi

# 使用 Python 脚本提取失败的数据
echo "📖 从错误日志中提取失败的数据..."
python3 << 'PYTHON_SCRIPT'
import json
import os
import sys

# 配置
input_file = "/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
output_file = "/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
error_file = "/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
retry_input_file = "/mnt/disk/lxh/gill_data/wukong_retry_failed.jsonl"
image_root = "/mnt/disk/lxh/gill_data/images"

# 0. 读取输出文件，获取已成功处理的路径（用于清理错误日志）
print(f"📖 读取输出文件，检查已成功处理的数据: {output_file}")
successful_paths = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                image_path = data.get('image_path', '')
                if image_path:
                    # 标准化路径
                    if not os.path.isabs(image_path):
                        full_path = os.path.join(image_root, image_path)
                    else:
                        full_path = image_path
                    successful_paths.add(os.path.normpath(full_path))
            except json.JSONDecodeError:
                continue
print(f"✅ 已成功处理: {len(successful_paths)} 条")
print("")

# 1. 读取错误日志，提取失败的 image_path，并过滤已成功的
print(f"📖 读取错误日志: {error_file}")
failed_paths = set()
failed_entries = []  # 保存失败记录，用于清理错误日志
original_error_count = 0
with open(error_file, 'r', encoding='utf-8') as f:
    for line in f:
        original_error_count += 1
        try:
            error_entry = json.loads(line.strip())
            image_path = error_entry.get('image_path', '')
            if image_path:
                # 标准化路径（处理绝对路径和相对路径）
                if not os.path.isabs(image_path):
                    full_path = os.path.join(image_root, image_path)
                else:
                    full_path = image_path
                normalized_path = os.path.normpath(full_path)
                
                # 只保留未成功的数据
                if normalized_path not in successful_paths:
                    failed_paths.add(normalized_path)
                    failed_entries.append(error_entry)
        except json.JSONDecodeError:
            continue

print(f"✅ 提取到 {len(failed_paths)} 个仍需重试的图片路径")
if len(failed_entries) < original_error_count:
    cleaned_count = original_error_count - len(failed_entries)
    print(f"   （已过滤 {cleaned_count} 条已成功的数据）")
print("")

# 2. 从原始输入文件中提取对应的数据项
print(f"📖 从原始输入文件中匹配数据: {input_file}")
retry_items = []
matched_count = 0
not_found_paths = set(failed_paths)  # 保存原始失败路径集合，用于统计未找到的

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            image_path = item.get('image_path', '')
            if not image_path:
                continue
            
            # 标准化路径
            if not os.path.isabs(image_path):
                full_path = os.path.join(image_root, image_path)
            else:
                full_path = image_path
            normalized_path = os.path.normpath(full_path)
            
            # 检查是否在失败列表中
            if normalized_path in failed_paths:
                retry_items.append(item)
                matched_count += 1
                failed_paths.remove(normalized_path)  # 移除已匹配的
                not_found_paths.discard(normalized_path)  # 从未找到集合中移除
        except json.JSONDecodeError:
            continue

not_found_count = len(not_found_paths)

print(f"✅ 匹配到 {matched_count} 条数据")
if not_found_count > 0:
    print(f"⚠️  有 {not_found_count} 个失败的路径在原始输入文件中未找到")
    print("   这可能是路径格式不一致导致的")
print("")

# 3. 写入重试输入文件
if len(retry_items) == 0:
    print("❌ 没有找到需要重试的数据")
    sys.exit(1)

print(f"💾 写入重试输入文件: {retry_input_file}")
with open(retry_input_file, 'w', encoding='utf-8') as f:
    for item in retry_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已写入 {len(retry_items)} 条数据到重试输入文件")
print("")

# 4. 清理错误日志（备份后清理已成功的数据）
if len(failed_entries) < original_error_count:
    print(f"💾 清理错误日志中已成功的数据...")
    backup_file = error_file + ".backup"
    print(f"   备份原错误日志到: {backup_file}")
    
    # 备份原文件
    import shutil
    if os.path.exists(error_file):
        shutil.copy2(error_file, backup_file)
    
    # 写入清理后的错误日志
    with open(error_file, 'w', encoding='utf-8') as f:
        for entry in failed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    cleaned_count = original_error_count - len(failed_entries)
    print(f"✅ 已清理 {cleaned_count} 条已成功的数据，错误日志已更新")

# 输出统计信息
print("")
print("=" * 60)
print("📊 统计信息")
print("=" * 60)
print(f"原始错误日志: {original_error_count} 条")
print(f"仍需重试: {matched_count} 条")
print(f"成功匹配到输入文件: {matched_count} 条")
if not_found_count > 0:
    print(f"未在输入文件中找到: {not_found_count} 条")
print(f"重试文件: {retry_input_file}")
print("=" * 60)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "❌ 提取失败数据时出错"
    exit 1
fi

# 检查重试输入文件是否创建成功
if [ ! -f "$RETRY_INPUT_FILE" ]; then
    echo "❌ 重试输入文件创建失败"
    exit 1
fi

RETRY_COUNT=$(wc -l < "$RETRY_INPUT_FILE" 2>/dev/null || echo 0)
if [ "$RETRY_COUNT" -eq 0 ]; then
    echo "❌ 重试输入文件为空，没有需要重试的数据"
    exit 1
fi

echo ""
echo "============================================================"
echo "🚀 启动重试标注任务"
echo "============================================================"
echo ""

# 检查是否已有同名会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  发现已存在的 tmux 会话: $SESSION_NAME"
    echo ""
    echo "选项："
    echo "  1. 附加到现有会话（查看进度）"
    echo "  2. 杀死现有会话并创建新会话"
    echo "  3. 取消"
    echo ""
    read -p "请选择 (1/2/3): " choice
    
    case $choice in
        1)
            echo "📺 附加到现有会话..."
            tmux attach -t "$SESSION_NAME"
            exit 0
            ;;
        2)
            echo "🛑 停止现有会话..."
            tmux kill-session -t "$SESSION_NAME"
            sleep 1
            ;;
        3)
            echo "取消操作"
            exit 0
            ;;
        *)
            echo "无效选择，退出"
            exit 1
            ;;
    esac
fi

# 停止可能正在运行的旧进程
echo "🔍 检查并停止旧进程..."
pkill -f "annotate_async_vllm.*retry" 2>/dev/null
sleep 2

# 创建新的 tmux 会话
echo "📦 创建 tmux 会话: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -x 120 -y 30

# 准备运行命令
if command -v conda &> /dev/null; then
    RUN_CMD="conda activate $CONDA_ENV && "
else
    RUN_CMD=""
fi

# 构建单行命令（使用重试输入文件，输出到原文件）
RUN_CMD+="cd /home/lxh/Project/gill-main && "
RUN_CMD+="python scripts/annotate_async_vllm.py "
RUN_CMD+="--input $RETRY_INPUT_FILE "
RUN_CMD+="--image-root $IMAGE_ROOT "
RUN_CMD+="--output $OUTPUT_FILE "
RUN_CMD+="--model-name $MODEL_NAME "
RUN_CMD+="--api-base $API_BASE "
RUN_CMD+="--max-concurrency $MAX_CONCURRENCY"

# 在 tmux 会话中运行命令
echo "🚀 执行命令: $RUN_CMD"
tmux send-keys -t "$SESSION_NAME" "$RUN_CMD" C-m

echo ""
echo "✅ 重试标注任务已在 tmux 会话中启动"
echo ""
echo "============================================================"
echo "📋 常用命令"
echo "============================================================"
echo ""
echo "  查看进度:"
echo "    tmux attach -t $SESSION_NAME"
echo "    或: tmux a -t $SESSION_NAME"
echo ""
echo "  分离会话（不停止任务）:"
echo "    按 Ctrl+B，然后按 D"
echo ""
echo "  停止任务:"
echo "    在 tmux 会话中按 Ctrl+C"
echo ""
echo "  查看进度统计:"
echo "    bash scripts/check_progress.sh"
echo ""
echo "============================================================"
echo ""
echo "💡 提示："
echo "  - 重试的数据会追加到原输出文件: $OUTPUT_FILE"
echo "  - 标注脚本会自动跳过已成功处理的数据（断点续传）"
echo "  - 如果重试后仍有失败，可以再次运行此脚本"
echo ""
echo "============================================================"
