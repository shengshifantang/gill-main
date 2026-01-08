#!/usr/bin/env python3
"""
比较不同 checkpoint 的输出格式
用于验证"最后一个 epoch"是否格式更好
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model_and_tokenizer(base_path, adapter_path, device_str):
    """加载模型和 tokenizer"""
    print(f"📦 加载: {adapter_path}")
    
    # 检查实际可用的 GPU 数量
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，使用 CPU")
        device_obj = torch.device("cpu")
        device_map_str = "cpu"
    else:
        available_gpus = torch.cuda.device_count()
        print(f"  📍 检测到 {available_gpus} 个可用 GPU")
        
        # 解析设备字符串
        if device_str.startswith("cuda:"):
            requested_idx = int(device_str.split(":")[1])
            # 如果使用了 CUDA_VISIBLE_DEVICES，实际索引会被重新映射
            # 例如 CUDA_VISIBLE_DEVICES=2 时，PyTorch 只能看到 1 个 GPU（索引为 0）
            # 所以如果请求的是 cuda:2，但只有 1 个可见 GPU，应该使用 cuda:0
            if requested_idx >= available_gpus:
                print(f"  ⚠️  请求的设备 {requested_idx} 超出范围，使用 cuda:0")
                device_idx = 0
            else:
                device_idx = requested_idx
            device_obj = torch.device(f"cuda:{device_idx}")
            device_map_str = f"cuda:{device_idx}"
        elif device_str == "cuda":
            device_obj = torch.device("cuda:0")
            device_map_str = "cuda:0"
        else:
            device_obj = torch.device(device_str)
            device_map_str = device_str
    
    # 1. 优先从 Adapter 路径加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        print("  ✓ 从 Adapter 加载 Tokenizer")
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>"]})
        print("  ⚠️  从基座加载 Tokenizer")
    
    # 2. 加载基座模型（使用明确的设备映射，避免 auto 模式）
    print(f"  📍 使用设备: {device_map_str}")
    # 对于单 GPU，使用字典格式的 device_map 更安全
    # {"": device_str} 表示所有层都放在指定设备上
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map={"": device_map_str},  # 使用字典格式，确保所有层都在同一设备
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. 加载 LoRA（PeftModel 会自动处理设备）
    print("  📦 加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("  ✓ 模型加载完成\n")
    
    return model, tokenizer

def test_checkpoint(base_path, adapter_path, device_str, test_prompts):
    """测试单个 checkpoint"""
    model, tokenizer = load_model_and_tokenizer(base_path, adapter_path, device_str)
    
    # 解析设备字符串为 torch.device 对象（与 load_model_and_tokenizer 中的逻辑保持一致）
    if not torch.cuda.is_available():
        device_obj = torch.device("cpu")
    elif device_str.startswith("cuda:"):
        requested_idx = int(device_str.split(":")[1])
        available_gpus = torch.cuda.device_count()
        # 如果使用了 CUDA_VISIBLE_DEVICES，实际索引会被重新映射
        device_idx = 0 if requested_idx >= available_gpus else requested_idx
        device_obj = torch.device(f"cuda:{device_idx}")
    elif device_str == "cuda":
        device_obj = torch.device("cuda:0")
    else:
        device_obj = torch.device(device_str)
    
    results = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device_obj)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        
        # 分析格式
        has_obj = '<obj>' in generated
        has_box = '<box>' in generated
        has_tool_call = '</tool_call>' in generated or 'useRal' in generated
        has_garbage = any(x in generated for x in ['<|im_start|>', '<|im_end|>', '<|endoftext|>'])
        
        results.append({
            'prompt': prompt,
            'output': generated[:200],
            'has_obj': has_obj,
            'has_box': has_box,
            'has_tool_call': has_tool_call,
            'has_garbage': has_garbage,
            'format_ok': has_obj and has_box and not has_tool_call and not has_garbage
        })
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="比较不同 checkpoint 的输出格式")
    parser.add_argument("--base-model", type=str, default="./model/qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--checkpoints", type=str, nargs="+", 
                       default=["checkpoints/layout_planner/final", 
                               "checkpoints/layout_planner/checkpoint-22500"])
    args = parser.parse_args()
    
    test_prompts = [
        "画一只在桌子左边的猫",
        "天空中有飞鸟，下面是广阔的草地",
        "左边是一个红色的苹果，右边是一个黄色的香蕉"
    ]
    
    print("=" * 70)
    print("🔍 比较不同 Checkpoint 的输出格式")
    print("=" * 70)
    print()
    
    all_results = {}
    for checkpoint_path in args.checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n{'='*70}")
        print(f"📊 测试: {checkpoint_name}")
        print(f"{'='*70}\n")
        
        try:
            results = test_checkpoint(args.base_model, checkpoint_path, args.device, test_prompts)
            all_results[checkpoint_name] = results
            
            # 打印结果
            for r in results:
                print(f"📝 Prompt: {r['prompt']}")
                print(f"🤖 输出: {r['output']}")
                print(f"   格式检查:")
                print(f"     <obj>: {'✅' if r['has_obj'] else '❌'}")
                print(f"     <box>: {'✅' if r['has_box'] else '❌'}")
                print(f"     乱码: {'❌ 有' if r['has_tool_call'] or r['has_garbage'] else '✅ 无'}")
                print(f"   总体: {'✅ 格式正确' if r['format_ok'] else '⚠️  格式有问题'}")
                print()
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 总结")
    print("=" * 70)
    for name, results in all_results.items():
        format_ok_count = sum(1 for r in results if r['format_ok'])
        print(f"{name}: {format_ok_count}/{len(results)} 个样本格式正确")
    
    print("\n💡 建议:")
    if len(all_results) >= 2:
        names = list(all_results.keys())
        r1 = all_results[names[0]]
        r2 = all_results[names[1]]
        ok1 = sum(1 for r in r1 if r['format_ok'])
        ok2 = sum(1 for r in r2 if r['format_ok'])
        
        if ok2 > ok1:
            print(f"   ✅ {names[1]} 格式更好，建议使用该 checkpoint")
        elif ok1 > ok2:
            print(f"   ✅ {names[0]} 格式更好，建议使用该 checkpoint")
        else:
            print(f"   ⚠️  两个 checkpoint 格式相似，建议使用最后一个 epoch 的模型")

if __name__ == "__main__":
    main()
