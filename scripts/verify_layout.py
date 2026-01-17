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
from gill.layout_planner import parse_layout_output, format_layout_input

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Planner 验证脚本")
    parser.add_argument("--base-model", type=str, default="./model/qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="./checkpoints/layout_planner/final")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--input-jsonl", type=str, default=None, help="可选：从 JSONL 读取测试样本")
    parser.add_argument("--max-samples", type=int, default=6, help="最大测试样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于采样）")
    parser.add_argument("--output-jsonl", type=str, default=None, help="可选：保存逐条验证结果")
    return parser.parse_args()

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
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]})

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
    return format_layout_input(tokenizer, prompt, enable_cot=False, feedback=None)

def _get_prompt(item: Dict) -> str:
    for key in ("caption", "prompt", "text"):
        val = item.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""

def _reservoir_sample_prompts(path: str, k: int, seed: int) -> List[str]:
    import json
    import random
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            prompt = _get_prompt(item)
            if not prompt:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(prompt)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = prompt
    return reservoir

def main():
    args = parse_args()
    
    # 设置设备
    if "cuda" in args.device and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    # 加载
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path, device)

    # 测试用例（可从 JSONL 采样）
    if args.input_jsonl and os.path.exists(args.input_jsonl):
        test_prompts = _reservoir_sample_prompts(args.input_jsonl, args.max_samples, args.seed)
        if not test_prompts:
            print("⚠️ 从 JSONL 未采样到有效 prompt，回退到内置样例")
    else:
        test_prompts = []
    if not test_prompts:
        test_prompts = [
            "画一只在桌子左边的猫",
            "天空中有飞鸟，下面是广阔的草地",
            "左边是一个红色的苹果，右边是一个黄色的香蕉",
            "一个在玩飞盘的狗，背景是海滩",
            "上方是蓝天，下方是草地",
            "中间有一朵花，左边是树，右边是房子",
        ]
    if args.max_samples > 0:
        test_prompts = test_prompts[:args.max_samples]
    
    print("\n" + "=" * 60)
    print("🧐 开始验证推理效果 (Fixed Version)")
    print("=" * 60)
    
    total = 0
    format_ok = 0
    parse_ok = 0  # includes <no_layout>
    parse_obj_ok = 0  # only when objects parsed
    no_layout_count = 0
    results = []

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
            total += 1
            is_format_ok = ("<no_layout>" in clean_text) or ("<obj>" in clean_text and "<box>" in clean_text)
            if is_format_ok:
                format_ok += 1
            if "<no_layout>" in clean_text:
                print("   - <no_layout> (无布局输出)")
                no_layout_count += 1
                parse_ok += 1
                results.append({
                    "prompt": prompt,
                    "layout_text": clean_text,
                    "format_ok": is_format_ok,
                    "parsed_ok": True,
                    "parsed_objects_ok": False,
                    "no_layout": True,
                    "parsed": []
                })
                continue
            objects = parse_layout_output(clean_text)
            if objects:
                parse_ok += 1
                parse_obj_ok += 1
                for obj in objects:
                    bbox_str = ",".join([f"{x:.2f}" for x in obj['bbox']])
                    print(f"   - {obj['name']}: [{bbox_str}]")
            else:
                print("   ⚠️ 未解析到对象 (可能是格式仍有问题)")
            results.append({
                "prompt": prompt,
                "layout_text": clean_text,
                "format_ok": is_format_ok,
                "parsed_ok": bool(objects),
                "parsed_objects_ok": bool(objects),
                "no_layout": False,
                "parsed": objects
            })
                
        except Exception as e:
            print(f"❌ 出错: {e}")

    # Summary
    if total > 0:
        print("\n" + "=" * 60)
        print("📊 格式统计")
        print("=" * 60)
        print(f"Total: {total}")
        print(f"Format OK: {format_ok} ({format_ok/total*100:.1f}%)")
        print(f"Parsed OK (incl. <no_layout>): {parse_ok} ({parse_ok/total*100:.1f}%)")
        print(f"Parsed Objects OK: {parse_obj_ok} ({parse_obj_ok/total*100:.1f}%)")
        print(f"No-layout: {no_layout_count} ({no_layout_count/total*100:.1f}%)")

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        import json
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
