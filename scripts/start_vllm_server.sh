#!/bin/bash
# vLLM API Server 启动脚本
# 用于 Qwen2.5-VL-32B + 2x4090 环境
# 注意：模型有 40 个注意力头，必须使用能整除 40 的 tensor-parallel-size（1, 2, 4, 5, 8, 10, 20, 40）

# 设置 GPU
# 注意：tensor-parallel-size 必须能整除模型的注意力头数（40）
# 40 的因数：1, 2, 4, 5, 8, 10, 20, 40
# 使用 2 个 GPU 进行张量并行
export CUDA_VISIBLE_DEVICES=0,1

# 内存优化：减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 模型路径（请根据实际情况修改）
# 如果设置了 MODEL_PATH 环境变量则使用它，否则使用默认路径
MODEL_PATH="${MODEL_PATH:-/mnt/disk/lxh/models/Qwen2.5-VL-32B-Instruct-AWQ}"

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
# 注意：AWQ 量化只支持 float16，不支持 bfloat16
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --disable-log-requests

