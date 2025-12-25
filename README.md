# Chinese-GILL: Layout-Controlled Multimodal Generation

中文布局控制的多模态生成系统，支持图文交错生成和精确的空间布局控制。

## 🎯 核心特性

- **CoT Layout Planner**: 基于 DeepSeek-7B 的思维链布局规划器
- **Spatial Adapter**: GLIGEN 风格的空间控制适配器
- **端到端生成**: 支持布局控制的图文交错生成

## 📁 项目结构

```
gill-main/
├── gill/                    # 核心代码
│   ├── models.py           # GILL 模型
│   ├── layout_planner.py  # Layout Planner
│   └── spatial_adapter.py # Spatial Adapter
├── scripts/                 # 实验脚本
│   ├── filter_wukong_candidates.py      # 筛选候选样本
│   ├── annotate_with_qwen_vl_plus.py    # Qwen-VL-Plus 标注
│   ├── prepare_cot_training_data.py    # CoT 数据转换
│   ├── prepare_mixed_training_data.py  # 混合数据准备
│   ├── train_layout_planner.py         # 训练 Layout Planner
│   ├── train_spatial_adapter.py        # 训练 Spatial Adapter
│   ├── create_test_set.py              # 创建测试集
│   └── evaluate_baselines.py           # Baseline 对比评估
├── docs/                    # 文档
│   ├── EXPERIMENT_GUIDE.md              # 实验执行指南（主文档）
│   ├── DATA_SCALE_ARGUMENTATION.md     # 数据规模论证（论文用）
│   ├── BASELINE_EVALUATION_GUIDE.md    # Baseline 对比指南
│   ├── MIXED_TRAINING_GUIDE.md         # 混合数据训练指南
│   ├── COMPLETE_PIPELINE.md            # 完整流程总结
│   ├── ARCHITECTURE_SUMMARY.md         # 架构总结
│   └── RESUME_PROTECTION.md            # 断点续传说明
├── data/                    # 数据
│   ├── wukong_release/     # WuKong 完整数据集（31GB）
│   ├── wukong_test/        # WuKong 测试数据
│   └── layout_dataset_*.jsonl  # 布局数据集
├── configs/                 # 配置文件
│   └── baseline_config.json # Baseline 配置
└── checkpoints/             # 模型检查点
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas tqdm dashscope torch transformers diffusers
```

### 2. 查看实验指南

**主要文档**: `docs/EXPERIMENT_GUIDE.md`

包含完整的实验流程：
- Phase 1: 数据扩充（18-32 小时）
- Phase 2: 混合数据准备（30 分钟）
- Phase 3: 模型训练（10-16 小时）
- Phase 4: 评估与对比（4-8 小时）

## 📚 文档说明

### 核心文档

- **`docs/EXPERIMENT_GUIDE.md`** - 实验执行指南（**从这里开始**）
- **`docs/DATA_SCALE_ARGUMENTATION.md`** - 数据规模论证（论文用）
- **`docs/BASELINE_EVALUATION_GUIDE.md`** - Baseline 对比评估指南

### 辅助文档

- **`docs/MIXED_TRAINING_GUIDE.md`** - 混合数据训练策略
- **`docs/COMPLETE_PIPELINE.md`** - 完整流程总结
- **`docs/ARCHITECTURE_SUMMARY.md`** - 架构设计总结
- **`docs/RESUME_PROTECTION.md`** - 断点续传机制说明

## 🎯 实验流程概览

```
1. 筛选候选样本 (1h)
   ↓
2. Qwen-VL-Plus 标注 (12-24h, 挂机)
   ↓
3. 转换为 CoT 格式 (10min)
   ↓
4. 准备混合数据 (30min)
   ↓
5. 训练 Layout Planner (4-6h)
   ↓
6. 训练 Spatial Adapter (6-10h)
   ↓
7. 评估与对比 (4-8h)
```

## 📊 核心创新

1. **CoT Layout Planning**: 思维链增强的布局规划
2. **混合数据训练**: 15k Layout + 50k General，保证泛化性
3. **参数高效**: LoRA/Adapter 微调，降低数据需求
4. **多层次对比**: 消融实验 + SOTA 对比 + 通用模型对比

## ⚠️ 注意事项

- **API 费用**: Qwen-VL-Plus 标注约需 ¥150-300（15K 图片）
- **断点续传**: 所有脚本支持断点续传，API 额度用完后可无缝恢复
- **图片路径**: 如果只有 URL，需要处理图片下载或路径映射

## 📝 引用

如果使用本项目，请引用相关论文（待发表）。

---

**详细实验指南请查看**: `docs/EXPERIMENT_GUIDE.md`

