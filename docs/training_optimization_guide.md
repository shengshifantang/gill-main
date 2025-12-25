# 3x RTX 4090 训练优化指南

## 📊 显存分析

### 当前配置
- **GPU**: 3x RTX 4090 (24GB x 3 = 72GB 总显存)
- **模型**: Kolors (SDXL-based, ~2.6B UNet + 4B Text Encoder)
- **精度**: BF16 混合精度

### 显存占用估算

| 组件 | 参数量 | BF16 显存 | 备注 |
|------|--------|-----------|------|
| Kolors UNet | 2.6B | ~5.2GB | 冻结，仅推理 |
| Text Encoder (ChatGLM) | 4B | ~8GB | 冻结 |
| Spatial Adapter | ~50M | ~100MB | 可训练 |
| 优化器状态 (AdamW) | 50M × 2 | ~200MB | 2 份动量 |
| 梯度 | 50M | ~100MB | |
| 激活值 (batch=1) | - | ~8GB | 最大头 |
| **单卡总计** | - | **~21.6GB** | 接近极限 |

---

## ⚠️ 关键问题：你的代码会 OOM

### 问题代码
```python
# ❌ 你的 main.py (会爆显存)
for mode in ['captioning', 'retrieval', 'generation']:
    loss = forward(mode)
    total_loss += loss

total_loss.backward()  # 保存 3 份完整计算图！
```

**为什么会 OOM？**
1. PyTorch 的 `backward()` 需要保存整个计算图
2. 你累加了 3 个 mode 的 loss，计算图包含 3 次 UNet forward
3. 激活值显存 = 8GB × 3 = **24GB**（超出单卡容量）

---

## ✅ 解决方案

### 方案 A：分步 Backward（推荐）

```python
# ✅ 修改 main.py 的 train() 函数
optimizer.zero_grad()

for mode in ['captioning', 'retrieval', 'generation']:
    # Forward
    loss = forward(mode)
    
    # 立即 Backward（释放计算图）
    scaled_loss = loss / 3.0  # 平均 3 个 mode 的梯度
    scaler.scale(scaled_loss).backward()
    
    # 清理中间变量
    del loss
    torch.cuda.empty_cache()

# 梯度裁剪 + 优化器步进
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
scaler.step(optimizer)
scaler.update()
```

**优点**：
- 每次只保存 1 份计算图，显存占用 ~8GB
- 梯度自动累加（PyTorch 默认行为）
- 无需修改模型架构

---

### 方案 B：Gradient Checkpointing

```python
# ✅ 在加载 UNet 时启用
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "Kwai-Kolors/Kolors",
    subfolder="unet",
    torch_dtype=torch.bfloat16
)

# 启用 Gradient Checkpointing
unet.enable_gradient_checkpointing()
```

**原理**：
- 前向传播时不保存中间激活值
- 反向传播时重新计算（用时间换空间）
- 显存减少 ~40%，训练时间增加 ~20%

**⚠️ 注意**：
- 你的 Spatial Adapter 也需要支持 checkpointing
- 需要在 `SpatialControlAdapter.forward()` 中使用 `torch.utils.checkpoint.checkpoint()`

---

### 方案 C：DeepSpeed ZeRO-2（多卡场景）

```bash
# ✅ 安装 DeepSpeed
pip install deepspeed

# ✅ 创建配置文件 ds_config.json
{
  "train_batch_size": 3,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

```python
# ✅ 修改 main.py
import deepspeed

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=trainable_params,
    config="ds_config.json"
)

# 训练循环
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**优点**：
- 优化器状态分片到 3 张卡（每卡只存 1/3）
- 可选 CPU Offload（进一步节��显存）
- 支持更大的 batch size

**缺点**：
- 需要重构训练代码
- 通信开销（3 卡间需要同步梯度）

---

## 🎯 推荐配置（3x 4090）

### 数据清洗阶段
```bash
# 使用修复版脚本
python scripts/prepare_layout_dataset_fixed.py \
    --input-tsv data/wukong_train.tsv \
    --image-dir /data/wukong/images \
    --output-jsonl data/layout_dataset.jsonl \
    --num-gpus 3 \
    --batch-size 8 \
    --model-path Qwen/Qwen2-VL-7B-Instruct
```

**预期速度**：
- 单卡吞吐：~5 images/s
- 3 卡并行：~15 images/s
- 10 万张图：~2 小时

---

### 训练阶段

#### 配置 1：单卡训练（最简单）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset layout \
    --batch-size 1 \
    --grad-accumulation-steps 8 \
    --precision bf16 \
    --lr 1e-4 \
    --epochs 10
```

**特点**：
- 无需修改 DDP 代码
- 有效 batch size = 1 × 8 = 8
- 训练时间：~3 天（10 万样本）

---

#### 配置 2：3 卡 DDP（推荐）
```bash
# 使用方案 A（分步 Backward）
torchrun --nproc_per_node=3 main.py \
    --dataset layout \
    --batch-size 1 \
    --grad-accumulation-steps 3 \
    --precision bf16 \
    --lr 1e-4 \
    --epochs 10 \
    --multiprocessing-distributed
```

**特点**：
- 有效 batch size = 1 × 3 × 3 = 9
- 训练时间：~1 天
- 需要修改 `train()` 函数（见方案 A）

---

#### 配置 3：3 卡 DeepSpeed（最优）
```bash
deepspeed --num_gpus=3 main.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --dataset layout \
    --batch-size 2 \
    --precision bf16 \
    --lr 1e-4 \
    --epochs 10
```

**特点**：
- 有效 batch size = 2 × 3 = 6
- 优化器状态分片（节省 ~4GB 显存）
- 可以跑 batch_size=2（提升训练稳定性）

---

## 🔧 代码修改清单

### 1. 修改 `main.py` 的 `train()` 函数

```python
# 在 train() 函数中找到这段代码：
for i, batch in enumerate(train_loader):
    # ... 数据预处理 ...
    
    # ❌ 删除这段（原代码）
    # total_loss = 0
    # for mode in model_modes:
    #     loss = forward(mode)
    #     total_loss += loss
    # total_loss.backward()
    
    # ✅ 替换为（方案 A）
    optimizer.zero_grad()
    for mode_idx, mode in enumerate(model_modes):
        # Forward
        with torch.cuda.amp.autocast(enabled=(args.precision == 'fp16')):
            result = model(images, tgt_tokens, token_len, mode=mode, ...)
            loss = compute_loss(result, mode)  # 你的 loss 计算逻辑
        
        # Backward（立即释放计算图）
        scaled_loss = loss / len(model_modes)
        scaler.scale(scaled_loss).backward()
        
        # 清理
        del loss, result
        if mode_idx < len(model_modes) - 1:
            torch.cuda.empty_cache()
    
    # 梯度裁剪 + 优化器步进
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
```

---

### 2. 修改 `gill/spatial_adapter.py`

```bash
# 直接替换为修复版
cp gill/spatial_adapter_fixed.py gill/spatial_adapter.py
```

**关键修改**：
1. ✅ 注入位置：`attn2`（Cross-Attention）而非 `attn1`
2. ✅ 坐标验证：自动检查 BBox 是否归一化
3. ✅ 维度适配：支持 SDXL 的多层维度（320/640/1280/2048）

---

### 3. 修改数据清洗脚本

```bash
# 使用修复版（Ray 并行）
python scripts/prepare_layout_dataset_fixed.py \
    --input-tsv data/wukong_train.tsv \
    --image-dir /data/wukong/images \
    --output-jsonl data/layout_dataset.jsonl \
    --num-gpus 3
```

**依赖安装**：
```bash
pip install ray qwen-vl-utils
```

---

## 📈 性能对比

| 方案 | 显存/卡 | 训练速度 | 实现难度 | 推荐度 |
|------|---------|----------|----------|--------|
| 原代码（累加 loss） | **OOM** | - | - | ❌ |
| 方案 A（分步 backward） | 21GB | 1.0x | ⭐ | ⭐⭐⭐⭐⭐ |
| 方案 B（Gradient Checkpointing） | 15GB | 0.8x | ⭐⭐ | ⭐⭐⭐ |
| 方案 C（DeepSpeed ZeRO-2） | 18GB | 1.2x | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🐛 常见问题

### Q1: 训练时显存仍然不足？
**A**: 尝试以下组合：
```python
# 1. 启用 Gradient Checkpointing
unet.enable_gradient_checkpointing()

# 2. 减少 batch size
--batch-size 1 --grad-accumulation-steps 16

# 3. 冻结 Text Encoder（如果未冻结）
for param in text_encoder.parameters():
    param.requires_grad = False
```

---

### Q2: DDP 报错 "RuntimeError: Expected to mark a variable ready only once"？
**A**: 这是因为不同 mode 使用了不同的参数子集。解决方法：
```python
# 在 main.py 的 DDP 初始化时
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.gpu],
    find_unused_parameters=True  # ✅ 关键
)
```

---

### Q3: 数据清洗时 Qwen2-VL 输出格式不稳定？
**A**: 使用更严格的 prompt：
```python
prompt = """请严格按照以下 JSON 格式输出：
[
  {"label": "物体名称", "bbox": [x1, y1, x2, y2]}
]
坐标范围：0-1000
不要输出任何其他文字。"""
```

---

## 📚 参考资料

1. **GLIGEN 论文**: https://arxiv.org/abs/2301.07093
2. **Diffusers Attention Processor**: https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview
3. **DeepSpeed ZeRO**: https://www.deepspeed.ai/tutorials/zero/
4. **Gradient Checkpointing**: https://pytorch.org/docs/stable/checkpoint.html

---

## ✅ 总结

你的原始方案**思路正确**，但有 4 个关键问题：

1. ❌ **vLLM 多模态支持不完善** → ✅ 使用 Ray + Transformers
2. ❌ **Spatial Adapter 注入到 Self-Attention** → ✅ 改为 Cross-Attention
3. ❌ **累加 loss 导致显存爆炸** → ✅ 分步 backward
4. ❌ **缺少坐标验证** → ✅ 自动检查归一化

**立即行动**：
1. 替换 `spatial_adapter.py` 为修复版
2. 修改 `main.py` 的 `train()` 函数（方案 A）
3. 使用 `prepare_layout_dataset_fixed.py` 清洗数据
4. 开始训练！

祝你的 **GILL-Next-CN** 项目顺利！🚀

