#!/bin/bash
# 在 tmux 会话中运行标注任务

# 配置参数
SESSION_NAME="annotation"
INPUT_FILE="/mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl"
IMAGE_ROOT="/mnt/disk/lxh/gill_data/images"
OUTPUT_FILE="/mnt/disk/lxh/gill_data/wukong_labeled.jsonl"
MODEL_NAME="/mnt/disk/lxh/models/Qwen2.5-VL-32B-Instruct-AWQ"
API_BASE="http://localhost:8000/v1"
MAX_CONCURRENCY=32  # 使用优化后的并发数

# 激活 conda 环境（如果需要）
CONDA_ENV="gill_chinese"

echo "============================================================"
echo "🚀 启动标注任务（tmux 会话）"
echo "============================================================"
echo ""

# 检查 vLLM 服务是否运行
if ! curl -s "$API_BASE/models" > /dev/null 2>&1; then
    echo "❌ 错误: vLLM 服务未运行"
    echo ""
    echo "标注任务需要 vLLM 服务才能运行。"
    echo "请先启动服务端，然后再运行标注任务。"
    echo ""
    echo "启动方式："
    echo "  方式 1: 使用完整启动脚本（推荐）"
    echo "    bash scripts/start_annotation_full.sh"
    echo ""
    echo "  方式 2: 手动启动"
    echo "    步骤 1: bash scripts/start_vllm_server.sh"
    echo "    步骤 2: 等待服务启动后，再运行此脚本"
    echo ""
    echo "  方式 3: 在 tmux 中启动服务端"
    echo "    tmux new-session -d -s vllm 'bash scripts/start_vllm_server.sh'"
    echo "    等待 1-2 分钟后，再运行此脚本"
    echo ""
    exit 1
fi

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 输入文件不存在: $INPUT_FILE"
    exit 1
fi

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
pkill -f "annotate_async_vllm" 2>/dev/null
sleep 2

# 询问是否删除旧文件
if [ -f "$OUTPUT_FILE" ]; then
    COMPLETED=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
    if [ "$COMPLETED" -gt 0 ]; then
        echo ""
        echo "📊 发现已存在的输出文件:"
        echo "   文件: $OUTPUT_FILE"
        echo "   已处理: $COMPLETED 条"
        echo ""
        echo "选项："
        echo "  1. 保留文件，继续处理（断点续传）"
        echo "  2. 删除文件，重新开始（使用新配置）"
        echo "  3. 取消"
        echo ""
        read -p "请选择 (1/2/3): " delete_choice
        
        case $delete_choice in
            1)
                echo "✅ 保留文件，将跳过已处理的数据"
                ;;
            2)
                echo "🗑️  删除输出文件..."
                rm -f "$OUTPUT_FILE"
                if [ -f "${OUTPUT_FILE%.jsonl}_errors.jsonl" ]; then
                    echo "🗑️  删除错误日志文件..."
                    rm -f "${OUTPUT_FILE%.jsonl}_errors.jsonl"
                fi
                echo "✅ 已删除，将重新开始处理"
                ;;
            3)
                echo "取消操作"
                exit 0
                ;;
            *)
                echo "无效选择，默认保留文件"
                ;;
        esac
    fi
fi

# 创建新的 tmux 会话
echo "📦 创建 tmux 会话: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -x 120 -y 30

# 准备运行命令（单行格式，避免反斜杠换行问题）
if command -v conda &> /dev/null; then
    RUN_CMD="conda activate $CONDA_ENV && "
else
    RUN_CMD=""
fi

# 构建单行命令（不使用反斜杠换行）
RUN_CMD+="cd /home/lxh/Project/gill-main && "
RUN_CMD+="python scripts/annotate_async_vllm.py "
RUN_CMD+="--input $INPUT_FILE "
RUN_CMD+="--image-root $IMAGE_ROOT "
RUN_CMD+="--output $OUTPUT_FILE "
RUN_CMD+="--model-name $MODEL_NAME "
RUN_CMD+="--api-base $API_BASE "
RUN_CMD+="--max-concurrency $MAX_CONCURRENCY"

# 在 tmux 会话中运行命令
echo "🚀 执行命令: $RUN_CMD"
tmux send-keys -t "$SESSION_NAME" "$RUN_CMD" C-m

echo ""
echo "✅ 标注任务已在 tmux 会话中启动"
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

