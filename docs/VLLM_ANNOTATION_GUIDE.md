# vLLM 异步标注指南

## 📋 概述

本指南介绍如何使用 **vLLM + AsyncIO** 架构进行高效的大规模图像标注。

### 架构优势

1. **服务端-客户端解耦**：服务端（vLLM）和客户端（标注脚本）分离，更稳定
2. **Continuous Batching**：vLLM 自动优化批处理，最大化 GPU 利用率
3. **Tensor Parallelism**：32B 模型分布在 3 张卡上，显存更稳定
4. **高并发**：单线程可处理数千个并发请求
5. **断点续传**：自动跳过已处理的数据

---

## 🚀 快速开始

### 第一步：启动 vLLM 服务端

在**独立的终端窗口**（或 `tmux`/`screen` 会话）中运行：

```bash
# 方式1：使用启动脚本（推荐）
bash scripts/start_vllm_server.sh

# 方式2：手动启动
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL_PATH=/root/models/Qwen2.5-VL-32B-Instruct-AWQ  # 根据实际情况修改

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq \
    --tensor-parallel-size 3 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --disable-log-requests
```

**等待看到**：`Uvicorn running on http://0.0.0.0:8000`

### 第二步：运行客户端标注脚本

在**另一个终端窗口**中运行：

```bash
conda activate gill_chinese

python scripts/annotate_async_vllm.py \
    --input /mnt/disk/lxh/gill_data/wukong_downloaded_500k.jsonl \
    --image-root /mnt/disk/lxh/gill_data/wukong_images \
    --output /mnt/disk/lxh/gill_data/wukong_labeled_vllm.jsonl \
    --api-base http://localhost:8000/v1 \
    --model-name /root/models/Qwen2.5-VL-32B-Instruct-AWQ \
    --max-concurrency 64
```

---

## 📊 参数说明

### vLLM 服务端参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型路径 | `/root/models/Qwen2.5-VL-32B-Instruct-AWQ` |
| `--quantization awq` | 量化方式 | `awq`（降低显存占用） |
| `--tensor-parallel-size` | Tensor 并行数 | `3`（3 张卡） |
| `--max-model-len` | 最大上下文长度 | `8192` |
| `--gpu-memory-utilization` | GPU 显存利用率 | `0.95`（最大化利用） |
| `--port` | API 服务端口 | `8000` |

### 客户端参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 JSONL 文件 | **必需** |
| `--output` | 输出 JSONL 文件 | **必需** |
| `--image-root` | 图片根目录 | **必需** |
| `--api-base` | vLLM API 地址 | `http://localhost:8000/v1` |
| `--model-name` | 模型名称（需与服务端一致） | **必需** |
| `--max-concurrency` | 最大并发数 | `64` |

---

## ⚙️ 性能调优

### 并发数调整

`--max-concurrency` 参数控制同时发送的请求数。建议根据显存负载调整：

- **3x4090 TP=3**：可以尝试 `50-100`
- **如果显存不足**：降低到 `32` 或 `16`
- **如果 GPU 利用率低**：增加到 `100` 或更高

### 监控 GPU 使用率

在另一个终端运行：

```bash
watch -n 1 nvidia-smi
```

观察：
- **GPU 利用率**：应该接近 100%
- **显存使用**：应该在 90%+ 但不超过 100%
- **如果显存不足**：降低 `--max-concurrency` 或 `--gpu-memory-utilization`

---

## 🔧 故障排查

### 问题1：vLLM 服务端无法启动

**症状**：启动后立即退出或报错

**解决方案**：
1. 检查模型路径是否正确
2. 检查 GPU 是否可用：`nvidia-smi`
3. 检查 CUDA 版本是否兼容
4. 尝试降低 `--gpu-memory-utilization` 到 `0.85`

### 问题2：客户端连接失败

**症状**：`Connection refused` 或 `Connection timeout`

**解决方案**：
1. 确认服务端已启动并显示 `Uvicorn running on http://0.0.0.0:8000`
2. 检查端口是否被占用：`netstat -tuln | grep 8000`
3. 如果服务端在其他机器，检查防火墙设置

### 问题3：显存不足（OOM）

**症状**：`CUDA out of memory`

**解决方案**：
1. 降低 `--max-concurrency`（客户端）
2. 降低 `--gpu-memory-utilization`（服务端）
3. 使用 AWQ 量化模型（如果还没用）

### 问题4：处理速度慢

**症状**：GPU 利用率低，处理速度慢

**解决方案**：
1. 增加 `--max-concurrency`（客户端）
2. 检查网络延迟（如果服务端在远程）
3. 确保使用 AWQ 量化模型

---

## 📈 性能对比

### 传统方式（多进程 + Transformers）

- **架构**：每个进程加载完整模型
- **显存占用**：每进程 ~24GB（32B 模型）
- **并发能力**：受限于进程数和显存
- **稳定性**：一个进程崩溃影响整体

### vLLM 方式（Client-Server）

- **架构**：单服务端 + 异步客户端
- **显存占用**：TP=3 共享显存，更高效
- **并发能力**：单线程可处理数千请求
- **稳定性**：服务端和客户端解耦，更稳定

**预期提升**：
- **吞吐量**：提升 2-3 倍
- **显存利用率**：提升 30-50%
- **稳定性**：显著提升

---

## 📝 输出格式

标注结果保存在 JSONL 文件中，每行格式：

```json
{
    "image_path": "/path/to/image.jpg",
    "caption": "图片描述",
    "vlm_output": "模型原始输出（包含 rationale 和 JSON）",
    "annotations": {
        "rationale": "空间推理说明",
        "objects": [
            {"name": "物体名", "bbox": [x1, y1, x2, y2]}
        ]
    },
    "objects": [
        {"name": "物体名", "bbox": [x1, y1, x2, y2]}
    ]
}
```

**注意**：
- `bbox` 坐标使用 0-1000 的归一化整数
- 如果解析失败，会有 `annotations_error: true` 字段

---

## 🔄 断点续传

脚本自动支持断点续传：

1. **启动时**：自动读取输出文件，收集已处理的图片路径
2. **处理时**：自动跳过已处理的图片
3. **写入时**：追加模式写入，不会覆盖已有数据

**使用场景**：
- 脚本中断后重新运行
- 分批处理大量数据
- 增量标注新数据

---

## 💡 最佳实践

1. **服务端常驻**：让 vLLM 服务端一直运行，避免重复加载模型
2. **监控资源**：使用 `nvidia-smi` 和 `htop` 监控 GPU 和 CPU
3. **分批处理**：如果数据量很大，可以分批处理，每批完成后检查结果
4. **日志记录**：保存服务端和客户端的日志，便于排查问题
5. **备份数据**：定期备份标注结果，防止数据丢失

---

## 📚 相关文档

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Qwen2.5-VL 模型文档](https://github.com/QwenLM/Qwen2-VL)
- [AsyncIO 文档](https://docs.python.org/3/library/asyncio.html)

