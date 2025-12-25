# 快速行动指南

## 🎯 你需要做的 5 件事

### 1️⃣ 替换 Spatial Adapter（1 分钟）

```bash
cd /home/lxh/Project/gill-main
cp gill/spatial_adapter_fixed.py gill/spatial_adapter.py
```

---

### 2️⃣ 修改 main.py 的 train() 函数（5 分钟）

找到 `main.py` 第 600-700 行左右的训练循环，找到这段代码：

```python
# ❌ 原代码（搜索这段）
for mode_idx, model_mode in enumerate(model_modes):
    # ... forward 逻辑 ...
    mode_loss = 0
    # ... 计算 loss ...
    total_loss = total_loss + mode_loss

# 在所有 mode 结束后
loss_to_back = total_loss / args.grad_accumulation_steps
scaler.scale(loss_to_back).backward()
```

**替换为**：

```python
# ✅ 修复后
for mode_idx, model_mode in enumerate(model_modes):
    # ... forward 逻辑（保持不变）...
    mode_loss = 0
    # ... 计算 loss（保持不变）...
    
    # ✅ 关键修改：立即 backward
    loss_to_back = mode_loss / (len(model_modes) * args.grad_accumulation_steps)
    scaler.scale(loss_to_back).backward()
    
    # ✅ 清理中间变量
    del mode_loss
    if mode_idx < len(model_modes) - 1:
        torch.cuda.empty_cache()

# ✅ 删除原来的 total_loss.backward()
```

**完整的修改示例**（如果找不到确切位置，参考这个模板）：

```python
# 在 train() 函数中，找到主训练循环
for i, (_, images, caption_images, ret_tokens, ret_caption_len, gen_tokens, gen_caption_len, clip_emb) in enumerate(train_loader):
    # ... 数据预处理代码（保持不变）...
    
    model_modes = ['captioning', 'retrieval', 'generation']
    # ❌ 删除这行：total_loss = 0
    forward_success = True
    
    for mode_idx, model_mode in enumerate(model_modes):
        # ... 原有的 forward 逻辑（保持不变）...
        # ... 原有的 loss 计算（保持不变）...
        
        # ✅ 在每个 mode 的 loss 计算完成后，立即添加：
        if not forward_success:
            break
        
        # mode_loss 是当前 mode 的总 loss（ce_loss + cont_loss + gen_loss）
        loss_to_back = mode_loss / (len(model_modes) * args.grad_accumulation_steps)
        
        if scaler is not None:
            scaler.scale(loss_to_back).backward()
        else:
            loss_to_back.backward()
        
        # 清理
        del mode_loss
        if mode_idx < len(model_modes) - 1:
            torch.cuda.empty_cache()
    
    # ❌ 删除原来的这些行：
    # losses.update(total_loss.item(), images.size(0))
    # loss_to_back = total_loss / args.grad_accumulation_steps
    # scaler.scale(loss_to_back).backward()
    
    # ✅ 保留优化器步进逻辑（不变）
    successful_steps += 1
    if (successful_steps % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
        # ... 原有的梯度裁剪和优化器步进（保持不变）...
```

---

### 3️⃣ 运行测试脚本（2 分钟）

```bash
cd /home/lxh/Project/gill-main
python scripts/test_spatial_adapter.py
```

**预期输出**：
```
✅ 测试通过：只注入到 Cross-Attention 层
✅ 触发警告: BBox 坐标异常，自动归一化
✅ 测试通过：成功创建 X 个不同维度的 Adapter
✅ Forward 成功
✅ Backward 成功
✅ 显存占用在 RTX 4090 (24GB) 范围内
```

---

### 4️⃣ 清洗数据（2 小时）

```bash
# 安装依赖
pip install ray qwen-vl-utils

# 运行清洗
python scripts/prepare_layout_dataset_fixed.py \
    --input-tsv /path/to/wukong_train.tsv \
    --image-dir /path/to/wukong/images \
    --output-jsonl data/layout_dataset.jsonl \
    --num-gpus 3 \
    --batch-size 8 \
    --resume
```

**监控进度**：
```bash
# 另开一个终端
watch -n 1 'wc -l data/layout_dataset.jsonl'
```

---

### 5️⃣ 开始训练（1 天）

```bash
# 3 卡 DDP 训练
torchrun --nproc_per_node=3 main.py \
    --dataset layout \
    --dataset-dir data \
    --image-dir /path/to/images \
    --batch-size 1 \
    --grad-accumulation-steps 3 \
    --precision bf16 \
    --lr 1e-4 \
    --epochs 10 \
    --multiprocessing-distributed \
    --exp-name gill_spatial_control
```

**监控显存**：
```bash
# 另开一个终端
watch -n 1 nvidia-smi
```

**预期显存占用**：每卡 ~21GB（��全范围内）

---

## 🐛 如果遇到问题

### 问题 1：测试脚本报错 "No module named 'gill.spatial_adapter_fixed'"

**解决**：
```bash
# 确认文件存在
ls -lh gill/spatial_adapter_fixed.py

# 如果不存在，重新创建
# （文件内容已在前面生成）
```

---

### 问题 2：训练时仍然 OOM

**解决**：
```bash
# 方案 A：启用 Gradient Checkpointing
# 在 main.py 的 model 加载后添加：
# unet.enable_gradient_checkpointing()

# 方案 B：减少 batch size
--batch-size 1 --grad-accumulation-steps 8

# 方案 C：使用 DeepSpeed（需要额外配置）
```

---

### 问题 3：DDP 报错 "Expected to mark a variable ready only once"

**解决**：
在 `main.py` 中找到 DDP 初始化（约 400 行），确认有这行：
```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.gpu],
    find_unused_parameters=True  # ✅ 必须为 True
)
```

---

### 问题 4：数据清洗速度慢

**解决**：
```bash
# 增加 batch size（如果显存允许）
--batch-size 16

# 或者只清洗部分数据测试
head -n 1000 wukong_train.tsv > wukong_test.tsv
python scripts/prepare_layout_dataset_fixed.py --input-tsv wukong_test.tsv ...
```

---

## 📊 预期效果

### 数据清洗
- **速度**：~15 images/s（3 卡并行）
- **质量**：~60% 的图像能成功标注 BBox
- **时间**：10 万张图 ~2 小时

### 训练
- **显存**：~21GB/卡（batch_size=1）
- **速度**：~0.5 steps/s（3 卡 DDP）
- **时间**：10 万样本 10 epochs ~1 天

---

## 📚 参考文档

- **完整审查报告**：`docs/code_review_summary.md`
- **显存优化指南**：`docs/training_optimization_guide.md`
- **修复后的代码**：
  - `gill/spatial_adapter_fixed.py`
  - `scripts/prepare_layout_dataset_fixed.py`
  - `scripts/test_spatial_adapter.py`

---

## ✅ 检查清单

- [ ] 已替换 `gill/spatial_adapter.py`
- [ ] 已修改 `main.py` 的 `train()` 函数
- [ ] 测试脚本通过（所有 ✅）
- [ ] 数据清洗完成（生成 `.jsonl` 文件）
- [ ] 训练启动且显存在 24GB 以内

---

**祝训练顺利！** 🚀

如有问题，请查看 `docs/code_review_summary.md` 的常见问题部分。

