# 第一阶段训练验证指南

## 概述

本指南帮助你验证第一阶段（Spatial Adapter）训练的效果，确保训练逻辑正确，特别是 **Gate 参数正确更新并起作用**。

## 快速开始

### 1. 运行验证脚本

```bash
cd /home/lxh/Project/gill-main
bash scripts/run_validation.sh
```

脚本会自动：
- 查找最新的 checkpoint
- 分析 Gate 参数
- 分析 Adapter 权重
- 生成对比图（有/无 Adapter）

### 2. 手动运行（自定义参数）

```bash
python scripts/validate_training.py \
    --checkpoint ./checkpoints/spatial_adapter_fp16_fixed/checkpoint-2000.pt \
    --output-dir ./outputs/validation_step2000 \
    --kolors-path ./model/Kolors \
    --test-data ./data/coco2014_cn_val_clean.jsonl \
    --num-samples 10 \
    --device cuda:0
```

## 验证内容

### ✅ 1. Gate 参数分析

**验证目标**：确保 Gate 从初始值 0 更新，并在合理范围内。

#### 输出文件
- `gate_analysis.json`：详细的 Gate 统计信息
- `gate_distribution.png`：可视化分布图

#### 健康指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| **Tanh(Gate) 均值** | 0.1 ~ 0.7 | 太接近 0 说明未学习，太接近 1 说明饱和 |
| **接近零比例** | < 50% | 超过 50% 说明大部分 Gate 未起作用 |
| **饱和比例** | < 30% | 超过 30% 可能过拟合 |

#### 判断标准

✅ **训练成功**：
```
Tanh(Gate) 均值: 0.35
接近零比例: 15%
饱和比例: 5%
```

⚠️ **训练不足**：
```
Tanh(Gate) 均值: 0.02
接近零比例: 85%
饱和比例: 0%
```
→ 需要继续训练或检查学习率

⚠️ **过拟合**：
```
Tanh(Gate) 均值: 0.92
接近零比例: 5%
饱和比例: 60%
```
→ 降低学习率或增加正则化

### ✅ 2. 权重分析

**验证目标**：确保 Adapter 权重有效学习，不是随机初始化状态。

#### 输出文件
- `weight_analysis.json`：各层权重统计

#### 判断标准

✅ **正常学习**：
- Attention 层权重 Abs Mean > 0.01
- Output 层权重 Abs Mean > 0.005
- 权重标准差 > 0.01

⚠️ **未学习**：
- 所有权重接近 0
- 标准差极小

### ✅ 3. 生成效果对比

**验证目标**：确保 Adapter 能够控制物体在指定位置生成。

#### 输出文件
每个测试样本生成一个文件夹 `sample_XX/`，包含：
- `baseline.png`：无 Adapter 的生成结果
- `adapter.png`：有 Adapter 的生成结果
- `annotated.png`：标注了 bbox 的结果
- `comparison.png`：三图对比
- `metadata.json`：样本元数据

#### 判断标准

✅ **空间控制生效**：
1. **Baseline vs Adapter 有明显差异**
   - Baseline：物体位置随机
   - Adapter：物体出现在 bbox 指定位置

2. **Annotated 图中 bbox 与物体对齐**
   - 红框框住的区域包含对应物体
   - 物体大小与 bbox 大小匹配

3. **多物体场景**
   - 每个 bbox 对应一个物体
   - 物体之间不重叠（除非 bbox 重叠）

⚠️ **控制失效**：
- Baseline 和 Adapter 生成结果几乎相同
- bbox 位置没有对应物体
- 物体位置完全随机

## 常见问题排查

### Q1: Gate 接近零比例 > 80%

**原因**：
- 学习率过小
- 训练步数不足
- 数据质量问题（bbox 不准确）

**解决方案**：
```bash
# 1. 检查训练日志中的 loss 是否下降
tail -100 logs/training.log

# 2. 增加学习率重新训练
python scripts/train_spatial_adapter.py \
    --lr 2e-4 \
    ...

# 3. 延长训练时间
python scripts/train_spatial_adapter.py \
    --epochs 10 \
    --save-every 1000 \
    ...
```

### Q2: Gate 饱和比例 > 50%

**原因**：
- 学习率过大
- 训练过度

**解决方案**：
```bash
# 使用较早的 checkpoint
python scripts/validate_training.py \
    --checkpoint ./checkpoints/spatial_adapter_fp16_fixed/checkpoint-1000.pt \
    ...

# 或降低学习率重新训练
python scripts/train_spatial_adapter.py \
    --lr 5e-5 \
    ...
```

### Q3: 生成效果无差异

**原因**：
- Adapter 未正确注入
- Gate 值过小（接近 0）
- 训练数据与测试数据分布不匹配

**解决方案**：
```bash
# 1. 检查 Gate 值
cat outputs/validation_stepXXXX/gate_analysis.json

# 2. 手动设置 Gate scale 测试
# 修改 validate_training.py 中的 inject_spatial_control_to_unet 调用
# 添加 adapter_container.set_scale(1.0)

# 3. 使用训练集样本测试
python scripts/validate_training.py \
    --test-data ./data/coco2014_cn_train_clean.jsonl \
    --num-samples 5 \
    ...
```

### Q4: 部分 bbox 生效，部分失效

**原因**：
- 小物体学习困难
- 某些类别数据不足
- bbox 质量参差不齐

**解决方案**：
```bash
# 1. 分析失效的 bbox 特征
# 查看 metadata.json 中的 bboxes 和 obj_names

# 2. 过滤小 bbox 重新训练
python scripts/train_spatial_adapter.py \
    --min-bbox-area 0.03 \
    ...

# 3. 增加困难样本的训练权重（需修改训练代码）
```

## 进阶分析

### 1. 逐层 Gate 分析

查看 `gate_distribution.png` 中的 "各层 Gate 激活值" 图：
- **均匀分布**：各层学习均衡 ✅
- **某些层为 0**：该层未学习，可能需要调整架构 ⚠️
- **某些层饱和**：该层过拟合，考虑 dropout ⚠️

### 2. 时间步分析

不同 timestep 的控制效果可能不同：

```python
# 修改 validate_training.py，测试不同 timestep
for timestep in [0, 10, 20, 30]:
    # 在 scheduler.add_noise 时固定 timestep
    ...
```

### 3. Scale 敏感性测试

测试不同 scale 值的效果：

```python
# 在 inject_spatial_control_to_unet 后添加
for scale in [0.5, 1.0, 1.5, 2.0]:
    adapter_container.set_scale(scale)
    # 生成图像
    ...
```

## 验证通过标准

满足以下条件，说明第一阶段训练成功：

- [x] Gate 的 Tanh 均值在 0.2 ~ 0.8 之间
- [x] 接近零比例 < 50%
- [x] 饱和比例 < 30%
- [x] Adapter 权重 Abs Mean > 0.01
- [x] 对比图中，Adapter 版本在 bbox 位置生成了对应物体
- [x] 至少 70% 的测试样本控制生效

## 下一步

验证通过后，可以进行：

1. **第二阶段训练**：训练 Layout Planner
2. **超参数调优**：调整学习率、scale 范围等
3. **数据增强**：添加更多训练数据
4. **模型优化**：调整 Adapter 架构

## 参考

- 训练脚本：`scripts/train_spatial_adapter.py`
- 模型定义：`gill/spatial_adapter_fixed.py`
- 测试脚本：`scripts/test_checkpoint.sh`

