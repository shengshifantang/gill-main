#!/bin/bash
# vLLM API Server 启动脚本
# 用于 Qwen2.5-VL-32B + 3x4090 环境

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0,1,2

# 模型路径（请根据实际情况修改）
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-VL-32B-Instruct-AWQ}"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    echo "请设置 MODEL_PATH 环境变量或修改脚本中的路径"
    exit 1
fi

echo "🚀 启动 vLLM API Server"
echo "模型路径: $MODEL_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 启动 vLLM API Server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq \
    --tensor-parallel-size 3 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --disable-log-requests

