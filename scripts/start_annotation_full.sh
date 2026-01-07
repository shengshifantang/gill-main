#!/bin/bash
# 完整启动脚本：先启动 vLLM 服务，再启动标注任务

echo "============================================================"
echo "🚀 完整启动流程：vLLM 服务 + 标注任务"
echo "============================================================"
echo ""

# 检查是否有失败的数据需要重跑
ERROR_FILE="/mnt/disk/lxh/gill_data/wukong_labeled_errors.jsonl"
RETRY_MODE=false

if [ -f "$ERROR_FILE" ]; then
    ERROR_COUNT=$(wc -l < "$ERROR_FILE" 2>/dev/null || echo 0)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  发现 $ERROR_COUNT 条失败记录"
        echo ""
        echo "选项："
        echo "  1. 正常启动新的标注任务"
        echo "  2. 重跑失败的数据"
        echo ""
        read -p "请选择 (1/2，默认1): " choice
        choice=${choice:-1}
        
        if [ "$choice" = "2" ]; then
            RETRY_MODE=true
        fi
        echo ""
    fi
fi

# 检查 vLLM 服务是否已运行
echo "🔍 检查 vLLM 服务状态..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM 服务已在运行"
    SKIP_VLLM=true
else
    echo "❌ vLLM 服务未运行，需要启动"
    SKIP_VLLM=false
fi

echo ""

# 步骤 1: 启动 vLLM 服务（如果未运行）
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
            if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
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
            echo "⚠️  警告: 服务启动超时，但继续尝试启动标注任务"
            echo "   如果失败，请手动检查 vLLM 服务:"
            echo "     tmux attach -t vllm"
        fi
    fi
else
    echo "✅ 跳过 vLLM 服务启动（服务已在运行）"
fi

echo ""
echo "============================================================"
echo "步骤 2: 启动标注任务"
echo "============================================================"
echo ""

# 再次检查服务（确保服务可用）
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "❌ vLLM 服务未响应，无法启动标注任务"
    echo ""
    echo "请检查服务状态:"
    echo "  tmux attach -t vllm"
    echo "  或"
    echo "  curl http://localhost:8000/v1/models"
    echo ""
    exit 1
fi

# 运行标注任务启动脚本
if [ "$RETRY_MODE" = true ]; then
    echo "🔄 重跑失败的数据..."
    bash scripts/retry_failed_annotations.sh
else
echo "🚀 启动标注任务..."
bash scripts/run_annotation_tmux.sh
fi

echo ""
echo "============================================================"
echo "✅ 启动完成"
echo "============================================================"
echo ""
echo "📋 查看状态："
echo "  服务端: tmux attach -t vllm"
if [ "$RETRY_MODE" = true ]; then
    echo "  重试任务: tmux attach -t annotation_retry"
else
echo "  标注任务: tmux attach -t annotation"
fi
echo ""
echo "📊 查看进度:"
echo "  bash scripts/check_progress.sh"
echo ""
if [ "$RETRY_MODE" = false ]; then
    echo "🔄 重跑失败数据:"
    echo "  bash scripts/retry_failed_annotations.sh"
    echo ""
fi
echo "============================================================"

