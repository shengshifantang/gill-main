#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备混合训练数据（Layout 数据 + 通用数据）

功能：
1. 从标注数据中提取 Layout 数据（含对象的）
2. 从输入数据中提取通用数据（未标注的）
3. 按指定比例混合
4. 打乱并保存
"""

import json
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any


def load_layout_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """加载 Layout 数据（含对象的标注）"""
    layout_data = []
    
    print(f"📖 读取 Layout 数据: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                
                # 只保留含对象的标注
                if 'objects' in item and len(item.get('objects', [])) > 0:
                    # 排除标注错误的数据
                    if not item.get('annotations_error') and not item.get('error_type'):
                        item['has_layout'] = True
                        layout_data.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"✅ 提取到 {len(layout_data)} 条 Layout 数据")
    return layout_data


def load_generic_data(
    labeled_jsonl: str,
    unlabeled_jsonl: str,
    include_no_objects: bool = True
) -> List[Dict[str, Any]]:
    """加载通用数据（未标注 + 无对象）"""
    generic_data = []
    
    # 1. 从未标注数据中提取
    if os.path.exists(unlabeled_jsonl):
        print(f"📖 读取未标注数据: {unlabeled_jsonl}")
        processed_paths = set()
        
        # 先读取已标注的路径（用于过滤）
        if os.path.exists(labeled_jsonl):
            with open(labeled_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        image_path = item.get('image_path', '')
                        if image_path:
                            processed_paths.add(image_path)
                    except:
                        pass
        
        # 读取未标注的数据
        unlabeled_count = 0
        with open(unlabeled_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    image_path = item.get('image_path', '')
                    
                    # 只保留未标注的数据
                    if image_path and image_path not in processed_paths:
                        item['has_layout'] = False
                        item['objects'] = []
                        generic_data.append(item)
                        unlabeled_count += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ 提取到 {unlabeled_count} 条未标注数据")
    
    # 2. 从已标注数据中提取无对象的数据
    if include_no_objects and os.path.exists(labeled_jsonl):
        print(f"📖 读取无对象数据: {labeled_jsonl}")
        no_objects_count = 0
        with open(labeled_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # 只保留无对象且无错误的数据
                    if (item.get('no_objects', False) or 
                        ('objects' in item and len(item.get('objects', [])) == 0)):
                        if not item.get('annotations_error') and not item.get('error_type'):
                            item['has_layout'] = False
                            item['objects'] = []  # 确保无对象
                            generic_data.append(item)
                            no_objects_count += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ 提取到 {no_objects_count} 条无对象数据")
    
    print(f"✅ 总计通用数据: {len(generic_data)} 条")
    return generic_data


def prepare_mixed_data(
    layout_data: List[Dict[str, Any]],
    generic_data: List[Dict[str, Any]],
    layout_ratio: float,
    total_size: int = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    准备混合训练数据
    
    Args:
        layout_data: Layout 数据列表
        generic_data: 通用数据列表
        layout_ratio: Layout 数据占比 (0.0-1.0)
        total_size: 总数据量（如果指定，会按比例采样）
        seed: 随机种子
    """
    random.seed(seed)
    
    # 如果没有指定总数据量，使用所有 Layout 数据
    if total_size is None:
        total_size = len(layout_data)
    
    # 计算需要的数量
    layout_needed = int(total_size * layout_ratio)
    generic_needed = total_size - layout_needed
    
    # 检查数据是否充足
    if layout_needed > len(layout_data):
        print(f"⚠️  警告: 需要 {layout_needed} 条 Layout 数据，但只有 {len(layout_data)} 条")
        layout_needed = len(layout_data)
        generic_needed = total_size - layout_needed
    
    if generic_needed > len(generic_data):
        print(f"⚠️  警告: 需要 {generic_needed} 条通用数据，但只有 {len(generic_data)} 条")
        generic_needed = len(generic_data)
        total_size = layout_needed + generic_needed
    
    # 采样
    sampled_layout = random.sample(layout_data, layout_needed)
    sampled_generic = random.sample(generic_data, generic_needed)
    
    # 混合并打乱
    mixed_data = sampled_layout + sampled_generic
    random.shuffle(mixed_data)
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description="准备混合训练数据（Layout + Generic）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python scripts/prepare_mixed_training_data.py \\
      --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
      --unlabeled /mnt/disk/lxh/gill_data/wukong_downloaded_500k_fixed.jsonl \\
      --output data/mixed_train_20pct.jsonl \\
      --layout-ratio 0.2 \\
      --total-size 100000
        """
    )
    
    parser.add_argument(
        "--labeled",
        type=str,
        required=True,
        help="已标注数据文件路径（包含 Layout 和无对象数据）"
    )
    parser.add_argument(
        "--unlabeled",
        type=str,
        required=True,
        help="未标注数据文件路径（通用数据来源）"
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
        default=0.2,
        help="Layout 数据占比 (0.0-1.0，默认 0.2 即 20%%)"
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=None,
        help="总数据量（如果指定，会按比例采样；默认使用所有 Layout 数据）"
    )
    parser.add_argument(
        "--include-no-objects",
        action="store_true",
        default=True,
        help="是否包含无对象的标注数据（默认 True）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0.0 <= args.layout_ratio <= 1.0:
        print("❌ 错误: layout-ratio 必须在 0.0-1.0 之间")
        return
    
    if not os.path.exists(args.labeled):
        print(f"❌ 错误: 已标注数据文件不存在: {args.labeled}")
        return
    
    if not os.path.exists(args.unlabeled):
        print(f"❌ 错误: 未标注数据文件不存在: {args.unlabeled}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📦 准备混合训练数据")
    print("=" * 60)
    print()
    
    # 1. 加载 Layout 数据
    layout_data = load_layout_data(args.labeled)
    
    if len(layout_data) == 0:
        print("❌ 错误: 没有找到可用的 Layout 数据")
        return
    
    # 2. 加载通用数据
    generic_data = load_generic_data(
        args.labeled,
        args.unlabeled,
        include_no_objects=args.include_no_objects
    )
    
    if len(generic_data) == 0:
        print("❌ 错误: 没有找到可用的通用数据")
        return
    
    # 3. 准备混合数据
    print()
    print("=" * 60)
    print("🔀 混合数据")
    print("=" * 60)
    print()
    
    mixed_data = prepare_mixed_data(
        layout_data=layout_data,
        generic_data=generic_data,
        layout_ratio=args.layout_ratio,
        total_size=args.total_size,
        seed=args.seed
    )
    
    # 4. 统计信息
    layout_count = sum(1 for item in mixed_data if item.get('has_layout', False))
    generic_count = len(mixed_data) - layout_count
    
    print(f"✅ 混合数据准备完成:")
    print(f"   总数据量: {len(mixed_data)}")
    print(f"   Layout 数据: {layout_count} ({layout_count/len(mixed_data)*100:.1f}%)")
    print(f"   通用数据: {generic_count} ({generic_count/len(mixed_data)*100:.1f}%)")
    print()
    
    # 5. 保存
    print(f"💾 保存到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(mixed_data)} 条混合数据")
    print()
    print("=" * 60)
    print("💡 使用建议")
    print("=" * 60)
    print(f"   训练命令:")
    print(f"   python scripts/train_spatial_adapter.py \\")
    print(f"       --mixed-data {args.output} \\")
    print(f"       --kolors-model /path/to/Kolors \\")
    print(f"       --output-dir checkpoints/spatial_adapter_{int(args.layout_ratio*100)}pct")
    print("=" * 60)


if __name__ == "__main__":
    main()
