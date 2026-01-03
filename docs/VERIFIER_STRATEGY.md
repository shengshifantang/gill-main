# 🔍 验证器策略文档（Verifier Strategy）

## 核心原则：避免"自循环验证"偏差

### 问题背景

如果你用 **Qwen3-VL-32B** 标注数据，训练出的 Adapter 本质上是在模仿 Qwen3-VL 的"视觉观"。如果你再用同一个模型去**验证**生成结果：

- **系统性偏差放大**：如果 Qwen3-VL 对"左上方"的理解有系统性偏差，你的生成模型学会了这个偏差，而验证模型也会认为这是对的。
- **性能瓶颈**：32B 模型在推理阶段速度较慢，多轮迭代等待时间过长。

---

## 🌟 推荐方案：异构验证器架构（Heterogeneous Verifier Architecture）

### 方案一：混合验证器（Hybrid Verifier）- **推荐用于在线推理**

**组合**：Grounding DINO + Qwen2-VL-7B

- **Grounding DINO**：检测位置准确性（Neuro-Symbolic Feedback）
  - 纯粹的物体检测模型，基于像素和特征
  - 速度极快，判定标准与 VLM 不同
  - 避免系统性偏差

- **Qwen2-VL-7B**：检测语义准确性（颜色、属性等）
  - 轻量级 VLM，速度快
  - 模型架构不同，减少系统性偏差
  - 用于语义层面的验证

**论文宣称**：MoE-based Self-Correction（专家混合模型自我修正）

**使用方式**：
```python
from gill.feedback_verifier import create_feedback_verifier

verifier = create_feedback_verifier(
    verifier_type="hybrid",  # 混合模式
    device="cuda"
)
```

---

### 方案二：仅 Grounding DINO（用于计算 Metrics）

**适用场景**：论文评估实验、量化指标计算

**优势**：
- 速度极快
- 基于像素和特征，客观性强
- 适合计算 Detection Accuracy、Layout IoU 等指标

**使用方式**：
```python
verifier = create_feedback_verifier(
    verifier_type="grounding_dino",
    device="cuda"
)
```

---

### 方案三：仅 Qwen2-VL-7B（轻量级语义验证）

**适用场景**：资源受限、仅需语义验证

**优势**：
- 速度快（相比 32B 模型）
- 模型架构不同，减少系统性偏差
- 适合语义层面的验证

**使用方式**：
```python
verifier = create_feedback_verifier(
    verifier_type="qwen2vl_7b",
    device="cuda"
)
```

---

### 方案四：GPT-4o / Claude 3.5 Sonnet（仅用于评估实验）

**适用场景**：论文的 Evaluation 章节，作为"金标准裁判"

**使用方式**：
- 随机抽取 500 个 Case
- 使用 GPT-4o 作为"金标准裁判"
- 评估闭环效果

**优势**：
- 公信力高
- 避免"用自己的模型评估自己"

---

## 📊 数据构建 vs 在线验证

### 数据构建（Training Data）- 保持现状 ✅

**流程**：Qwen2.5-7B (Filter) + Qwen3-VL-32B-Thinking (Annotate)

**理由**：
- 这是"由强模型蒸馏出的高质量数据集"
- 双重过滤机制保证数据质量
- CoT (Thinking) 加持处理复杂方位

**论文宣称**：SOTA VLM 蒸馏数据

---

### 在线闭环（Inference Loop）- 使用异构验证器 ✅

**推荐组合**：Grounding DINO + Qwen2-VL-7B

**理由**：
- 避免"自循环验证"偏差
- 速度快，支持多轮迭代
- 异构验证更具说服力

**论文宣称**：基于专家混合模型（MoE-based）的自我修正能力

---

## 🔧 代码实现

### 更新后的 FeedbackVerifier

```python
from gill.feedback_verifier import FeedbackVerifier, create_feedback_verifier

# 混合模式（推荐）
verifier = create_feedback_verifier(
    verifier_type="hybrid",
    device="cuda"
)

# 验证结果包含各验证器的详细结果
result = verifier.verify(
    image=generated_image,
    original_prompt=prompt,
    expected_layout=layout_objects
)

# result["verifier_details"] 包含：
# - "grounding_dino": Grounding DINO 的验证结果
# - "qwen2vl_7b": Qwen2-VL-7B 的验证结果
```

### 在 InferenceAgent 中使用

```python
from scripts.inference_agent import InferenceAgent

agent = InferenceAgent(
    verifier_type="hybrid",  # 使用混合验证器
    # ... 其他参数
)
```

---

## 📝 论文撰写建议

### Methodology 章节

**数据构建**：
> "We construct a high-quality Chinese Layout-Text-Image dataset through a two-stage distillation process: (1) Qwen2.5-7B filters semantically invalid captions, and (2) Qwen3-VL-32B-Thinking annotates bounding boxes with Chain-of-Thought reasoning. This ensures SOTA-level data quality."

**验证器架构**：
> "To avoid self-correction bias, we employ a heterogeneous verifier architecture combining Grounding DINO (for spatial accuracy) and Qwen2-VL-7B (for semantic accuracy). This MoE-based approach provides neuro-symbolic feedback, ensuring objective verification."

### Experiments 章节

**验证器消融实验**：
- Baseline: 使用 Qwen3-VL-32B 作为验证器（自循环）
- Ours: 使用混合验证器（Grounding DINO + Qwen2-VL-7B）
- 对比指标：Detection Accuracy, Layout IoU, Human Evaluation

---

## ✅ 总结

1. **数据构建**：保持 Qwen2.5-7B + Qwen3-VL-32B-Thinking（SOTA 级质量）
2. **在线验证**：使用混合验证器（Grounding DINO + Qwen2-VL-7B）
3. **论文宣称**：SOTA VLM 蒸馏数据 + MoE-based 自我修正

**避免"自循环验证"偏差，这是论文说服力的关键！**

