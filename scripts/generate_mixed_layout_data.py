#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成混合 Layout 训练数据（推荐方案）

包含：
- 80% 有对象的数据（学习如何生成布局）
- 20% 无对象的数据（学习何时不生成布局）
"""

import json
import random
import argparse
import os
from typing import List, Dict


def load_layout_data(jsonl_path: str) -> List[Dict]:
    """加载有对象的 Layout 数据"""
    layout_data = []
    
    print(f"📖 读取 Layout 数据: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                
                # 只保留有对象的数据
                if 'objects' in item and len(item.get('objects', [])) > 0:
                    if not item.get('annotations_error') and not item.get('error_type'):
                        layout_data.append(item)
            except:
                continue
    
    print(f"✅ 提取到 {len(layout_data)} 条 Layout 数据")
    return layout_data


def load_no_object_data(jsonl_path: str) -> List[Dict]:
    """加载无对象的数据"""
    no_object_data = []
    
    print(f"📖 读取无对象数据: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                
                # 只保留无对象的数据
                if item.get('no_objects', False) or len(item.get('objects', [])) == 0:
                    if not item.get('annotations_error') and not item.get('error_type'):
                        # 确保 objects 为空
                        item['objects'] = []
                        no_object_data.append(item)
            except:
                continue
    
    print(f"✅ 提取到 {len(no_object_data)} 条无对象数据")
    return no_object_data


def prepare_mixed_data(
    layout_data: List[Dict],
    no_object_data: List[Dict],
    layout_ratio: float = 0.8,
    total_size: int = None,
    seed: int = 42
) -> List[Dict]:
    """
    准备混合训练数据
    
    Args:
        layout_data: 有对象的数据
        no_object_data: 无对象的数据
        layout_ratio: Layout 数据占比（默认 0.8）
        total_size: 总数据量（默认使用所有 Layout 数据）
        seed: 随机种子
    """
    random.seed(seed)
    
    # 如果没有指定总数据量，使用所有 Layout 数据
    if total_size is None:
        total_size = len(layout_data)
    
    # 计算需要的数量
    layout_needed = int(total_size * layout_ratio)
    no_object_needed = total_size - layout_needed
    
    # 检查数据是否充足
    if layout_needed > len(layout_data):
        print(f"⚠️  警告: 需要 {layout_needed} 条 Layout 数据，但只有 {len(layout_data)} 条")
        layout_needed = len(layout_data)
        no_object_needed = total_size - layout_needed
    
    if no_object_needed > len(no_object_data):
        print(f"⚠️  警告: 需要 {no_object_needed} 条无对象数据，但只有 {len(no_object_data)} 条")
        no_object_needed = len(no_object_data)
        total_size = layout_needed + no_object_needed
    
    # 采样
    sampled_layout = random.sample(layout_data, layout_needed)
    sampled_no_object = random.sample(no_object_data, no_object_needed)
    
    # 混合并打乱
    mixed_data = sampled_layout + sampled_no_object
    random.shuffle(mixed_data)
    
    print(f"\n✅ 混合数据准备完成:")
    print(f"   总数据量: {len(mixed_data)}")
    print(f"   Layout 数据: {layout_needed} ({layout_needed/len(mixed_data)*100:.1f}%)")
    print(f"   无对象数据: {no_object_needed} ({no_object_needed/len(mixed_data)*100:.1f}%)")
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description="生成混合 Layout 训练数据（推荐方案）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

1. 推荐配比（80% Layout + 20% 无对象）：
   python scripts/generate_mixed_layout_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_mixed_80_20.jsonl \\
       --layout-ratio 0.8

2. 保守配比（70% Layout + 30% 无对象）：
   python scripts/generate_mixed_layout_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_mixed_70_30.jsonl \\
       --layout-ratio 0.7

3. 指定总数据量：
   python scripts/generate_mixed_layout_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_mixed.jsonl \\
       --layout-ratio 0.8 \\
       --total-size 200000
        """
    )
    
    parser.add_argument(
        "--labeled",
        type=str,
        required=True,
        help="已标注数据文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出混合数据文件路径"
    )
    parser.add_argument(
        "--layout-ratio",
        type=float,
        default=0.8,
        help="Layout 数据占比（默认 0.8，即 80%%）"
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=None,
        help="总数据量（默认使用所有 Layout 数据）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0.0 < args.layout_ratio < 1.0:
        print("❌ 错误: layout-ratio 必须在 0.0-1.0 之间")
        return
    
    if not os.path.exists(args.labeled):
        print(f"❌ 错误: 标注数据文件不存在: {args.labeled}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📦 生成混合 Layout 训练数据")
    print("=" * 60)
    print()
    
    # 1. 加载有对象的数据
    layout_data = load_layout_data(args.labeled)
    
    if len(layout_data) == 0:
        print("❌ 错误: 没有找到可用的 Layout 数据")
        return
    
    # 2. 加载无对象的数据
    no_object_data = load_no_object_data(args.labeled)
    
    if len(no_object_data) == 0:
        print("❌ 错误: 没有找到可用的无对象数据")
        return
    
    # 3. 准备混合数据
    print()
    print("=" * 60)
    print("🔀 混合数据")
    print("=" * 60)
    print()
    
    mixed_data = prepare_mixed_data(
        layout_data=layout_data,
        no_object_data=no_object_data,
        layout_ratio=args.layout_ratio,
        total_size=args.total_size,
        seed=args.seed
    )
    
    # 4. 保存
    print()
    print(f"💾 保存到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(mixed_data)} 条混合数据")
    print()
    print("=" * 60)
    print("💡 训练建议")
    print("=" * 60)
    print(f"   训练命令:")
    print(f"   CUDA_VISIBLE_DEVICES=2 python scripts/train_layout_planner.py \\")
    print(f"       --layout-json {args.output} \\")
    print(f"       --val-json data/coco-cn/coco-cn_val.jsonl \\")
    print(f"       --base-model ./model/qwen2.5-7B-Instruct \\")
    print(f"       --output-dir ./checkpoints/layout_planner_mixed \\")
    print(f"       --epochs 3 \\")
    print(f"       --use-format-metric")
    print("=" * 60)


if __name__ == "__main__":
    main()
