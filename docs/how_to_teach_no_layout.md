# 如何让模型学会"不画框"：训练数据格式设计

## 核心原理

模型通过 **Causal Language Modeling** 学习：给定输入，预测输出。

### 有对象的数据（教模型画框）

```json
{
  "caption": "桌子左边有一只猫",
  "objects": [
    {"name": "桌子", "bbox": [0.1, 0.5, 0.5, 0.9]},
    {"name": "猫", "bbox": [0.0, 0.3, 0.4, 0.7]}
  ]
}
```

**训练时的 input/output**：
```
Input:  <|im_start|>user\n桌子左边有一只猫<|im_end|>\n<|im_start|>assistant\n
Output: <obj>桌子</obj><box>[0.10,0.50,0.50,0.90]</box><obj>猫</obj><box>[0.00,0.30,0.40,0.70]</box><|im_end|>
```

模型学到：**有具体物体 → 输出 `<obj>...</obj><box>...</box>`**

### 无对象的数据（教模型不画框）

```json
{
  "caption": "美丽的风景",
  "objects": []
}
```

**关键问题：output 应该是什么？**

有三种方案：

## 方案对比

### 方案 1：输出空字符串（推荐）

```
Input:  <|im_start|>user\n美丽的风景<|im_end|>\n<|im_start|>assistant\n
Output: <|im_end|>
```

**优点**：
- ✅ 最简单直接
- ✅ 模型学会"什么都不说"
- ✅ 推理时直接检测输出是否为空

**缺点**：
- ⚠️ 可能与 EOS token 混淆

### 方案 2：输出特殊标记（明确）

```
Input:  <|im_start|>user\n美丽的风景<|im_end|>\n<|im_start|>assistant\n
Output: <no_layout><|im_end|>
```

**优点**：
- ✅ 语义明确
- ✅ 易于解析
- ✅ 不会与 EOS 混淆

**缺点**：
- ⚠️ 需要添加新的 special token

### 方案 3：输出自然语言解释（不推荐）

```
Input:  <|im_start|>user\n美丽的风景<|im_end|>\n<|im_start|>assistant\n
Output: 该描述中没有具体的前景物体，无需生成布局。<|im_end|>
```

**优点**：
- ✅ 可解释性强

**缺点**：
- ❌ 增加训练难度
- ❌ 输出不稳定
- ❌ 解析复杂

## 🎯 推荐方案：输出空字符串

### 实现方式

修改 `LayoutJsonlDataset` 类，处理无对象数据：

```python
class LayoutJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_samples: int = -1):
        self.samples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                inp = item.get("caption", "").strip()
                objs = item.get("objects", [])
                
                if len(objs) > 0:
                    # 有对象：生成标准格式
                    out_parts = []
                    for obj in objs:
                        name = obj.get("name", "").strip()
                        bbox = obj.get("bbox", [])
                        if name and len(bbox) == 4:
                            bbox_str = ",".join(f"{v:.2f}" for v in bbox)
                            out_parts.append(f"<obj>{name}</obj><box>[{bbox_str}]</box>")
                    
                    if out_parts:
                        out = "".join(out_parts)
                        self.samples.append({"input": inp, "output": out})
                
                else:
                    # 无对象：输出空字符串
                    self.samples.append({"input": inp, "output": ""})
```

### Label Masking 处理

在 `DataCollatorForLayoutPlanner` 中，需要正确处理空输出：

```python
@dataclass
class DataCollatorForLayoutPlanner:
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        
        for example in examples:
            messages = [
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}  # 可能为空
            ]
            
            # 完整对话
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # 只有 prompt 部分
            user_msg = [{"role": "user", "content": example["input"]}]
            prompt_text = self.tokenizer.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            full_ids = self.tokenizer(full_text, add_special_tokens=False, 
                                     max_length=self.max_length, truncation=True).input_ids
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, 
                                       max_length=self.max_length, truncation=True).input_ids
            
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = input_ids.clone()
            
            # Mask prompt 部分
            prompt_len = len(prompt_ids)
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                labels[:] = -100
            
            # 关键：即使 output 为空，也要让模型学习生成 EOS token
            # 这样模型会学会"在这种情况下，我应该立即结束"
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Padding...
        # (省略 padding 代码，与之前相同)
```

### 推理时的处理

```python
def generate_layout(self, prompt: str, **kwargs) -> Dict:
    # 生成
    result = self.model.generate(...)
    output_text = self.tokenizer.decode(...)
    
    # 清理
    output_text = output_text.strip()
    
    # 解析
    if not output_text or output_text == "":
        # 模型输出为空 → 无需布局
        return {
            "layout_text": "",
            "objects": []
        }
    
    # 否则正常解析
    objects = parse_layout_output(output_text)
    return {
        "layout_text": output_text,
        "objects": objects
    }
```

## 📊 训练数据示例

### 混合数据格式

```jsonl
{"caption": "桌子左边有一只猫", "objects": [{"name": "桌子", "bbox": [...]}, {"name": "猫", "bbox": [...]}]}
{"caption": "美丽的风景", "objects": []}
{"caption": "左边是树，右边是房子", "objects": [{"name": "树", "bbox": [...]}, {"name": "房子", "bbox": [...]}]}
{"caption": "抽象的艺术作品", "objects": []}
{"caption": "一个人在跑步", "objects": [{"name": "人", "bbox": [...]}]}
{"caption": "空荡荡的房间", "objects": []}
```

### 转换为训练样本

```python
# 有对象
{
  "input": "桌子左边有一只猫",
  "output": "<obj>桌子</obj><box>[0.10,0.50,0.50,0.90]</box><obj>猫</obj><box>[0.00,0.30,0.40,0.70]</box>"
}

# 无对象
{
  "input": "美丽的风景",
  "output": ""  # 空字符串
}
```

## 🔬 模型如何学习？

### 训练过程

1. **有对象样本**：
   ```
   Loss = CrossEntropy(predicted_tokens, target_tokens)
   Target: <obj>桌子</obj><box>[...]</box>...
   ```
   模型学会：输出结构化布局

2. **无对象样本**：
   ```
   Loss = CrossEntropy(predicted_tokens, EOS_token)
   Target: <|im_end|> (立即结束)
   ```
   模型学会：立即输出 EOS，不生成任何内容

### 决策边界

经过混合训练，模型会学到一个隐式的决策函数：

```python
def should_generate_layout(caption):
    # 模型内部学到的模式（简化表示）
    if has_concrete_objects(caption):
        return True  # 生成布局
    else:
        return False  # 立即 EOS
```

这个决策是通过大量样本学习到的，比规则更智能。

## 💡 关键点总结

### 1. 数据格式

```python
# 有对象
{"input": "桌子左边有一只猫", "output": "<obj>...</obj><box>...</box>"}

# 无对象
{"input": "美丽的风景", "output": ""}  # 空字符串
```

### 2. Label Masking

```python
# 只计算 assistant 部分的 loss
# 即使 output 为空，也要让模型学习生成 EOS
labels[:prompt_len] = -100  # Mask prompt
# labels[prompt_len:] 保留（包括 EOS token）
```

### 3. 推理解析

```python
if output_text == "" or not output_text:
    return {"objects": []}  # 无需布局
else:
    return parse_layout_output(output_text)
```

## 🚀 实施步骤

### 步骤 1：确认数据格式正确

```bash
# 检查混合数据
python3 -c "
import json

with open('data/layout_planner_mixed_80_20.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        item = json.loads(line)
        caption = item.get('caption', '')
        objects = item.get('objects', [])
        print(f'{i+1}. {caption[:30]}... → {len(objects)} 个对象')
"
```

### 步骤 2：确认训练脚本正确处理空输出

检查 `LayoutJsonlDataset` 是否正确处理 `objects: []` 的情况。

### 步骤 3：训练并验证

```bash
# 训练
CUDA_VISIBLE_DEVICES=2 python scripts/train_layout_planner.py \
    --layout-json data/layout_planner_mixed_80_20.jsonl \
    --epochs 3 \
    --use-format-metric

# 验证（测试无对象场景）
python3 -c "
from gill.layout_planner import LayoutPlanner

planner = LayoutPlanner(...)

# 测试有对象
result1 = planner.generate_layout('桌子左边有一只猫')
print(f'有对象: {len(result1[\"objects\"])} 个')

# 测试无对象
result2 = planner.generate_layout('美丽的风景')
print(f'无对象: {len(result2[\"objects\"])} 个')  # 应该是 0
"
```

## ✅ 预期效果

训练后，模型会：

| 输入 | 预期输出 | 实际效果 |
|------|----------|----------|
| "桌子左边有一只猫" | `<obj>...</obj><box>...</box>` | ✅ 正确生成 |
| "美丽的风景" | `""` (空) | ✅ 不生成 |
| "抽象的艺术作品" | `""` (空) | ✅ 不生成 |
| "左边是树，右边是房子" | `<obj>...</obj><box>...</box>` | ✅ 正确生成 |

**关键**：模型通过训练数据学会了"决策边界"，知道何时该画框、何时不该画框。
