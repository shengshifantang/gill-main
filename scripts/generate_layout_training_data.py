#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Layout Planner 训练数据

从标注数据中提取有对象的样本，生成纯 Layout 训练数据。
"""

import json
import argparse
import os
from typing import List, Dict, Any


def extract_layout_data(
    labeled_jsonl: str,
    output_jsonl: str,
    min_objects: int = 1,
    max_objects: int = None,
    filter_errors: bool = True
) -> int:
    """
    从标注数据中提取 Layout 数据
    
    Args:
        labeled_jsonl: 已标注数据文件路径
        output_jsonl: 输出文件路径
        min_objects: 最少对象数（默认 1）
        max_objects: 最多对象数（默认无限制）
        filter_errors: 是否过滤标注错误的数据（默认 True）
    
    Returns:
        提取的数据条数
    """
    layout_data = []
    
    print(f"📖 读取标注数据: {labeled_jsonl}")
    
    total_lines = 0
    skipped_no_objects = 0
    skipped_errors = 0
    skipped_too_many = 0
    
    with open(labeled_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            total_lines += 1
            
            try:
                item = json.loads(line.strip())
                
                # 检查是否有对象
                objects = item.get('objects', [])
                num_objects = len(objects)
                
                if num_objects < min_objects:
                    skipped_no_objects += 1
                    continue
                
                if max_objects and num_objects > max_objects:
                    skipped_too_many += 1
                    continue
                
                # 过滤标注错误
                if filter_errors:
                    if item.get('annotations_error') or item.get('error_type'):
                        skipped_errors += 1
                        continue
                
                # 添加到 layout_data
                layout_data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"⚠️  警告: 第 {line_num} 行处理失败: {e}")
                continue
    
    print(f"\n📊 数据统计:")
    print(f"   总行数: {total_lines}")
    print(f"   提取成功: {len(layout_data)}")
    print(f"   跳过（对象数 < {min_objects}）: {skipped_no_objects}")
    if max_objects:
        print(f"   跳过（对象数 > {max_objects}）: {skipped_too_many}")
    if filter_errors:
        print(f"   跳过（标注错误）: {skipped_errors}")
    
    # 保存
    print(f"\n💾 保存到: {output_jsonl}")
    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in layout_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(layout_data)} 条 Layout 数据")
    
    return len(layout_data)


def main():
    parser = argparse.ArgumentParser(
        description="生成 Layout Planner 训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

1. 生成纯 Layout 训练数据（推荐）：
   python scripts/generate_layout_training_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_train.jsonl

2. 限制对象数量（避免过于复杂的场景）：
   python scripts/generate_layout_training_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_train.jsonl \\
       --min-objects 1 \\
       --max-objects 10

3. 包含标注错误的数据（用于调试）：
   python scripts/generate_layout_training_data.py \\
       --labeled /mnt/disk/lxh/gill_data/wukong_labeled.jsonl \\
       --output data/layout_planner_train_with_errors.jsonl \\
       --no-filter-errors
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
        help="输出训练数据文件路径"
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=1,
        help="最少对象数（默认 1）"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="最多对象数（默认无限制）"
    )
    parser.add_argument(
        "--no-filter-errors",
        action="store_true",
        help="不过滤标注错误的数据"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.labeled):
        print(f"❌ 错误: 标注数据文件不存在: {args.labeled}")
        return
    
    if args.min_objects < 0:
        print(f"❌ 错误: min-objects 必须 >= 0")
        return
    
    if args.max_objects and args.max_objects < args.min_objects:
        print(f"❌ 错误: max-objects 必须 >= min-objects")
        return
    
    print("=" * 60)
    print("📦 生成 Layout Planner 训练数据")
    print("=" * 60)
    print()
    
    # 提取数据
    count = extract_layout_data(
        labeled_jsonl=args.labeled,
        output_jsonl=args.output,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        filter_errors=not args.no_filter_errors
    )
    
    if count == 0:
        print("\n❌ 错误: 没有提取到任何数据")
        return
    
    print()
    print("=" * 60)
    print("💡 使用建议")
    print("=" * 60)
    print(f"   训练命令:")
    print(f"   CUDA_VISIBLE_DEVICES=2 python scripts/train_layout_planner.py \\")
    print(f"       --layout-json {args.output} \\")
    print(f"       --val-json data/coco-cn/coco-cn_val.jsonl \\")
    print(f"       --base-model ./model/qwen2.5-7B-Instruct \\")
    print(f"       --output-dir ./checkpoints/layout_planner \\")
    print(f"       --epochs 3 \\")
    print(f"       --batch-size 2 \\")
    print(f"       --gradient-accumulation-steps 4 \\")
    print(f"       --lr 1e-4 \\")
    print(f"       --use-format-metric")
    print("=" * 60)


if __name__ == "__main__":
    main()
