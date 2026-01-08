#!/usr/bin/env python3
"""
Layout Planner 验证脚本 (Fixed Tokenizer Loading)

修正点：强制从 adapter_path 加载训练好的 Tokenizer，确保特殊 Token 和 Embeddings 对齐。
"""

import os
import sys
import argparse
import torch
import re
from typing import List, Dict

# 保证可以 import gill
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注意：这里我们不直接用 LayoutPlanner 类，而是手动加载以确保控制权
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Planner 验证脚本")
    parser.add_argument("--base-model", type=str, default="./model/qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="./checkpoints/layout_planner/final")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-refinement", action="store_true")
    return parser.parse_args()

def parse_layout_output(text: str) -> List[Dict]:
    """解析布局输出 (<obj>...</obj><box>...</box>)"""
    objects = []
    # 匹配 <obj>...</obj><box>...</box>
    pattern = r'<obj>(.*?)</obj><box>\[(.*?)\]</box>'
    matches = re.findall(pattern, text)
    
    for name, bbox_str in matches:
        try:
            bbox = [float(x.strip()) for x in bbox_str.split(',')]
            if len(bbox) == 4:
                # 兼容 0-1000 格式
                if max(bbox) > 1.5:
                    bbox = [b/1000.0 for b in bbox]
                objects.append({"name": name.strip(), "bbox": bbox})
        except:
            continue
    return objects

def load_model_and_tokenizer(base_path, adapter_path, device):
    print(f"🚀 正在加载模型...")
    print(f"  - 基座: {base_path}")
    print(f"  - Adapter: {adapter_path}")

    # 1. 优先从 Adapter 路径加载 Tokenizer (这是关键！确保 ID 一致)
    try:
        print("📦 尝试从 Adapter 加载训练好的 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        print("✓ 成功从 Adapter 加载 Tokenizer")
    except Exception as e:
        print(f"⚠️ Adapter 中无 Tokenizer，回退到基座 (可能导致乱码): {e}")
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>"]})

    # 2. 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 3. 调整 Embedding 大小 (必须与 Tokenizer 一致)
    model.resize_token_embeddings(len(tokenizer))

    # 4. 加载 LoRA
    # 注意：如果训练时保存了 embedding layer，PeftModel 会自动加载它
    print("📦 正在加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("✓ LoRA Adapter 加载成功！")
    
    return model, tokenizer

def format_prompt(tokenizer, prompt):
    # 使用 Chat Template 构建输入
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    args = parse_args()
    
    # 设置设备
    if "cuda" in args.device and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    # 加载
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path, device)

    # 测试用例
    test_prompts = [
        "画一只在桌子左边的猫",
        "天空中有飞鸟，下面是广阔的草地",
        "左边是一个红色的苹果，右边是一个黄色的香蕉",
        "一个在玩飞盘的狗，背景是海滩",
        "上方是蓝天，下方是草地",
        "中间有一朵花，左边是树，右边是房子",
    ]
    
    print("\n" + "=" * 60)
    print("🧐 开始验证推理效果 (Fixed Version)")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 测试 {i}: {prompt}")
        
        # 构造输入
        input_text = format_prompt(tokenizer, prompt)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, # 验证时建议贪婪解码，看最稳的结果
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=False) # 不跳过特殊 token，我们要看 <obj>
            
            # 清理 Qwen 特殊 token 方便显示
            clean_text = output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            print(f"🤖 输出: {clean_text}")
            
            # 解析
            objects = parse_layout_output(clean_text)
            if objects:
                for obj in objects:
                    bbox_str = ",".join([f"{x:.2f}" for x in obj['bbox']])
                    print(f"   - {obj['name']}: [{bbox_str}]")
            else:
                print("   ⚠️ 未解析到对象 (可能是格式仍有问题)")
                
        except Exception as e:
            print(f"❌ 出错: {e}")

if __name__ == "__main__":
    main()
