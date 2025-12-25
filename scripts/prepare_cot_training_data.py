#!/usr/bin/env python3
"""
将 Qwen-VL-Plus 标注的黄金数据转换为 CoT (Chain-of-Thought) 训练格式

这是论文的核心创新点：让 Layout Planner 不仅输出坐标，还输出推理过程。

训练数据格式：
Input: "左边一只猫"
Output: "分析：用户要求在左边放置猫。通常左侧区域的横坐标范围是 0 到 500。考虑到猫的常规比例，我将其放置在左侧居中位置。<obj>猫</obj><box>[200, 50, 800, 450]</box>"

Usage:
    python scripts/prepare_cot_training_data.py \
        --input-jsonl data/wukong_golden_15k_cot.jsonl \
        --output-jsonl data/layout_dataset_cot_15k.jsonl
"""

import argparse
import json
import os
from typing import Dict, List
from tqdm import tqdm


def format_bbox_1000(bbox_1000: List[int]) -> str:
    """将 0-1000 坐标格式化为字符串"""
    if len(bbox_1000) != 4:
        return ""
    return f"[{bbox_1000[0]},{bbox_1000[1]},{bbox_1000[2]},{bbox_1000[3]}]"


def format_bbox_01(bbox_01: List[float]) -> str:
    """将 0-1 坐标格式化为字符串（保留2位小数）"""
    if len(bbox_01) != 4:
        return ""
    return f"[{bbox_01[0]:.2f},{bbox_01[1]:.2f},{bbox_01[2]:.2f},{bbox_01[3]:.2f}]"


def create_cot_output(caption: str, reasoning: str, objects: List[Dict], use_1000: bool = True) -> str:
    """
    创建 CoT 格式的输出
    
    Args:
        caption: 原始描述
        reasoning: Qwen-VL-Plus 的推理过程
        objects: 物体列表（包含 bbox）
        use_1000: 是否使用 0-1000 格式（True）还是 0-1 格式（False）
    
    Returns:
        CoT 格式的输出文本
    """
    # 构造推理部分（可以进一步优化，让模型学习如何从 caption 生成推理）
    # 这里我们直接使用 Qwen-VL-Plus 的推理，作为"教师"知识
    cot_text = f"分析：{reasoning}\n\n"
    
    # 构造布局输出
    layout_parts = []
    for obj in objects:
        name = obj.get('name', '物体')
        if use_1000 and 'bbox_1000' in obj:
            bbox_str = format_bbox_1000(obj['bbox_1000'])
        else:
            bbox_str = format_bbox_01(obj['bbox'])
        
        layout_parts.append(f"<obj>{name}</obj><box>{bbox_str}</box>")
    
    cot_text += "".join(layout_parts)
    
    return cot_text


def convert_to_cot_format(input_file: str, output_file: str, use_1000: bool = True):
    """
    将黄金数据转换为 CoT 训练格式
    """
    print("=" * 60)
    print("🔄 转换为 CoT 训练格式")
    print("=" * 60)
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print(f"坐标格式: {'0-1000 (整数)' if use_1000 else '0-1 (浮点数)'}")
    print()
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 读取输入数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                samples.append(item)
            except:
                continue
    
    print(f"✓ 加载 {len(samples)} 条数据")
    
    # 转换为 CoT 格式
    cot_samples = []
    stats = {
        'total': 0,
        'has_reasoning': 0,
        'no_reasoning': 0,
    }
    
    for item in tqdm(samples, desc="转换中"):
        stats['total'] += 1
        
        caption = item.get('caption', '').strip()
        reasoning = item.get('cot_reasoning', '').strip()
        objects = item.get('objects', [])
        
        if not caption or not objects:
            continue
        
        # 如果没有推理文本，生成一个简单的
        if not reasoning:
            reasoning = f"根据描述'{caption}'，分析物体位置关系。"
            stats['no_reasoning'] += 1
        else:
            stats['has_reasoning'] += 1
        
        # 创建 CoT 输出
        cot_output = create_cot_output(caption, reasoning, objects, use_1000)
        
        # 构造训练样本
        cot_sample = {
            'input': caption,
            'output': cot_output,
            'reasoning': reasoning,
            'objects': objects,
            'bbox_source': item.get('bbox_source', 'unknown'),
        }
        
        cot_samples.append(cot_sample)
    
    # 保存
    print(f"\n💾 保存 CoT 训练数据...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in cot_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 60)
    print("✅ 转换完成!")
    print("=" * 60)
    print(f"总样本数: {stats['total']}")
    print(f"有推理文本: {stats['has_reasoning']}")
    print(f"无推理文本（已生成）: {stats['no_reasoning']}")
    print(f"输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="将黄金数据转换为 CoT 训练格式")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/wukong_golden_15k_cot.jsonl",
        help="输入的黄金数据 JSONL 文件",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/layout_dataset_cot_15k.jsonl",
        help="输出的 CoT 训练数据 JSONL 文件",
    )
    parser.add_argument(
        "--use-1000",
        action='store_true',
        help="使用 0-1000 整数坐标格式（推荐，提升精度）",
    )
    args = parser.parse_args()
    
    convert_to_cot_format(args.input_jsonl, args.output_jsonl, args.use_1000)


if __name__ == "__main__":
    main()

