#!/usr/bin/env python3
"""
诊断 Layout Planner 训练问题

检查：
1. 训练数据格式是否正确
2. Chat Template 格式化后的文本格式
3. Label Masking 是否正确
4. 特殊 Token 是否被正确学习
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_training_data(data_path: str):
    """诊断训练数据格式"""
    print("=" * 60)
    print("1. 检查训练数据格式")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    count = 0
    valid_input_output_count = 0
    valid_caption_objects_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            try:
                data = json.loads(line)
                
                # 情况1: 检查 input/output 格式（预处理好的格式）
                if 'input' in data and 'output' in data:
                    output = data['output']
                    if '<obj>' in output and '<box>' in output:
                        valid_input_output_count += 1
                    if count <= 3:
                        print(f"\n样本 {count} (input/output 格式):")
                        print(f"  Input: {data['input'][:80]}")
                        print(f"  Output: {data['output'][:150]}")
                        print(f"  包含 <obj>: {'<obj>' in output}")
                        print(f"  包含 <box>: {'<box>' in output}")
                
                # 情况2: 检查 caption + objects 格式（原始格式，会被 LayoutJsonlDataset 转换）
                elif 'caption' in data and 'objects' in data:
                    caption = data.get('caption', '').strip()
                    objects = data.get('objects', [])
                    if caption and objects and len(objects) > 0:
                        # 检查是否有有效的 bbox
                        has_valid_bbox = False
                        for obj in objects:
                            bbox = obj.get('bbox', [])
                            bbox_1000 = obj.get('bbox_1000', [])
                            if (bbox and len(bbox) == 4) or (bbox_1000 and len(bbox_1000) == 4):
                                has_valid_bbox = True
                                break
                        
                        if has_valid_bbox:
                            valid_caption_objects_count += 1
                        
                        if count <= 3:
                            print(f"\n样本 {count} (caption/objects 格式):")
                            print(f"  Caption: {caption[:80]}")
                            print(f"  Objects 数量: {len(objects)}")
                            print(f"  有有效 bbox: {has_valid_bbox}")
                            if objects:
                                first_obj = objects[0]
                                print(f"  第一个对象: name={first_obj.get('name', 'N/A')}, bbox={first_obj.get('bbox', 'N/A')}")
            except Exception as e:
                if count <= 3:
                    print(f"\n样本 {count}: 解析错误 - {e}")
                continue
    
    total_valid = valid_input_output_count + valid_caption_objects_count
    print(f"\n总计: {count} 条样本")
    print(f"input/output 格式: {valid_input_output_count} 条")
    print(f"caption/objects 格式: {valid_caption_objects_count} 条")
    print(f"有效格式总计: {total_valid} 条 ({total_valid/count*100:.1f}%)")
    
    return total_valid > 0


def diagnose_chat_template(base_model_path: str):
    """诊断 Chat Template 格式化"""
    print("\n" + "=" * 60)
    print("2. 检查 Chat Template 格式化")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>"]})
        
        # 测试样本
        sample_input = "画一只在桌子左边的猫"
        sample_output = "<obj>猫</obj><box>[0.10,0.20,0.40,0.50]</box><obj>桌子</obj><box>[0.50,0.30,0.90,0.80]</box>"
        
        # 构建完整对话
        messages = [
            {"role": "user", "content": sample_input},
            {"role": "assistant", "content": sample_output}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 构建 Prompt 部分
        user_msg = [{"role": "user", "content": sample_input}]
        prompt_text = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
        
        print(f"输入: {sample_input}")
        print(f"\n完整对话文本（最后 200 字符）:")
        print(full_text[-200:])
        print(f"\nPrompt 文本（最后 100 字符）:")
        print(prompt_text[-100:])
        
        # Tokenize 检查
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        
        assistant_ids = full_ids[len(prompt_ids):]
        assistant_text = tokenizer.decode(assistant_ids, skip_special_tokens=False)
        
        print(f"\nAssistant 部分 Token 数量: {len(assistant_ids)}")
        print(f"Assistant 部分文本: {assistant_text[:200]}")
        print(f"包含 <obj>: {'<obj>' in assistant_text}")
        print(f"包含 <box>: {'<box>' in assistant_text}")
        
        # 检查特殊 Token ID
        obj_id = tokenizer.convert_tokens_to_ids("<obj>")
        box_id = tokenizer.convert_tokens_to_ids("<box>")
        print(f"\n特殊 Token ID:")
        print(f"  <obj>: {obj_id}")
        print(f"  <box>: {box_id}")
        print(f"  <obj> 在 Assistant 部分: {obj_id in assistant_ids}")
        print(f"  <box> 在 Assistant 部分: {box_id in assistant_ids}")
        
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_label_masking(base_model_path: str, data_path: str):
    """诊断 Label Masking"""
    print("\n" + "=" * 60)
    print("3. 检查 Label Masking")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>"]})
        
        # 加载一个样本并转换为 input/output 格式
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    break
        
        # 处理两种数据格式
        if 'input' in data and 'output' in data:
            sample_input = data['input']
            sample_output = data['output']
        elif 'caption' in data and 'objects' in data:
            # 模拟 LayoutJsonlDataset 的转换逻辑
            sample_input = str(data.get('caption', '')).strip()
            objs = data.get('objects', [])
            
            # 获取图像尺寸
            width = float(data.get('width', 0))
            height = float(data.get('height', 0))
            has_dim = width > 0 and height > 0
            
            out_parts = []
            for obj in objs:
                # 名称处理
                name = str(obj.get("name", "")).strip()
                if not name:
                    # 尝试从 category_id 转换（简化版，实际代码中有完整映射）
                    category_id = obj.get("category_id")
                    if category_id:
                        name = f"物体{category_id}"
                    else:
                        name = "物体"
                
                # 坐标处理
                bbox = obj.get("bbox", [])
                bbox_1000 = obj.get("bbox_1000", [])
                bbox_final = None
                
                if bbox_1000 and len(bbox_1000) == 4:
                    bbox_final = [float(v) / 1000.0 for v in bbox_1000]
                elif bbox and len(bbox) == 4:
                    bbox_raw = [float(v) for v in bbox]
                    max_val = max(bbox_raw)
                    
                    if max_val <= 1.05:
                        bbox_final = bbox_raw
                    elif has_dim:
                        bbox_final = [
                            bbox_raw[0] / width,
                            bbox_raw[1] / height,
                            bbox_raw[2] / width,
                            bbox_raw[3] / height
                        ]
                    elif max_val <= 1000:
                        bbox_final = [v / 1000.0 for v in bbox_raw]
                
                if bbox_final:
                    bbox_final = [max(0.0, min(1.0, v)) for v in bbox_final]
                    bbox_str = ",".join(f"{v:.2f}" for v in bbox_final)
                    out_parts.append(f"<obj>{name}</obj><box>[{bbox_str}]</box>")
            
            sample_output = "".join(out_parts)
            
            if not sample_input or not sample_output:
                print("❌ 无法从 caption/objects 格式生成有效的 input/output")
                return False
        else:
            print("❌ 数据格式不支持：既没有 input/output，也没有 caption/objects")
            return False
        
        # 构建文本
        messages = [
            {"role": "user", "content": sample_input},
            {"role": "assistant", "content": sample_output}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        user_msg = [{"role": "user", "content": sample_input}]
        prompt_text = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        full_ids = tokenizer(full_text, add_special_tokens=False, max_length=512, truncation=True).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, max_length=512, truncation=True).input_ids
        
        # Label Masking（模拟训练代码逻辑）
        labels = full_ids.copy()
        prompt_len = len(prompt_ids)
        
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        else:
            labels = [-100] * len(labels)
        
        # 检查
        print(f"Full IDs 长度: {len(full_ids)}")
        print(f"Prompt IDs 长度: {prompt_len}")
        print(f"Label 中非 -100 的数量: {sum(1 for x in labels if x != -100)}")
        print(f"Label 中 -100 的数量: {sum(1 for x in labels if x == -100)}")
        
        # 检查 Assistant 部分的标签
        assistant_labels = labels[prompt_len:]
        assistant_tokens = full_ids[prompt_len:]
        assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=False)
        
        print(f"\nAssistant 部分:")
        print(f"  Token 数量: {len(assistant_tokens)}")
        print(f"  Label 数量: {len(assistant_labels)}")
        print(f"  需要计算 Loss 的 Token 数: {sum(1 for x in assistant_labels if x != -100)}")
        print(f"  文本: {assistant_text[:200]}")
        print(f"  包含 <obj>: {'<obj>' in assistant_text}")
        print(f"  包含 <box>: {'<box>' in assistant_text}")
        
        # 检查特殊 Token 的 Label
        obj_id = tokenizer.convert_tokens_to_ids("<obj>")
        box_id = tokenizer.convert_tokens_to_ids("<box>")
        
        obj_indices = [i for i, tid in enumerate(assistant_tokens) if tid == obj_id]
        box_indices = [i for i, tid in enumerate(assistant_tokens) if tid == box_id]
        
        print(f"\n特殊 Token 位置检查:")
        print(f"  <obj> 出现在位置: {obj_indices[:5]}... (共 {len(obj_indices)} 个)")
        print(f"  <box> 出现在位置: {box_indices[:5]}... (共 {len(box_indices)} 个)")
        
        if obj_indices:
            obj_label = assistant_labels[obj_indices[0]]
            print(f"  第一个 <obj> 的 Label: {obj_label} (应为非 -100)")
        
        if box_indices:
            box_label = assistant_labels[box_indices[0]]
            print(f"  第一个 <box> 的 Label: {box_label} (应为非 -100)")
        
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/layout_planner_train.jsonl")
    parser.add_argument("--base-model", type=str, default="./model/qwen2.5-7B-Instruct")
    args = parser.parse_args()
    
    print("🔍 Layout Planner 训练问题诊断")
    print("=" * 60)
    
    # 1. 检查训练数据
    data_ok = diagnose_training_data(args.data_path)
    
    # 2. 检查 Chat Template
    if data_ok:
        template_ok = diagnose_chat_template(args.base_model)
        
        # 3. 检查 Label Masking
        if template_ok:
            masking_ok = diagnose_label_masking(args.base_model, args.data_path)
            
            # 总结
            print("\n" + "=" * 60)
            print("诊断总结")
            print("=" * 60)
            print(f"训练数据格式: {'✅' if data_ok else '❌'}")
            print(f"Chat Template: {'✅' if template_ok else '❌'}")
            print(f"Label Masking: {'✅' if masking_ok else '❌'}")


if __name__ == "__main__":
    main()
