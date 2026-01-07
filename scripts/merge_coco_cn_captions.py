#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并 COCO-CN 两种 caption 类型的数据集

功能：
1. 合并 manually-translated 和 human-written 两种 caption 类型的数据
2. 保留所有数据（不去重，因为同一图片的不同描述是天然的数据增强）
3. 添加 source_caption_type 字段标识数据来源

用法：
python scripts/merge_coco_cn_captions.py \
    --translated-file /mnt/disk/lxh/gill_data/coco-cn-translated/coco-cn_train.jsonl \
    --human-file /mnt/disk/lxh/gill_data/coco-cn-human/coco-cn_train.jsonl \
    --output-file /mnt/disk/lxh/gill_data/coco-cn/coco-cn_train_merged.jsonl
"""

import json
import argparse
import os
from tqdm import tqdm
from collections import defaultdict

def merge_captions(translated_file, human_file, output_file):
    """
    合并两种 caption 类型的数据
    
    Args:
        translated_file: manually-translated 类型的 JSONL 文件
        human_file: human-written 类型的 JSONL 文件
        output_file: 输出的合并 JSONL 文件
    """
    
    translated_count = 0
    human_count = 0
    total_count = 0
    
    # 统计信息
    stats = {
        'translated': 0,
        'human': 0,
        'total': 0
    }
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 读取 manually-translated 数据
        if translated_file and os.path.exists(translated_file):
            print(f"📖 读取 manually-translated 数据: {translated_file}")
            with open(translated_file, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="处理 translated"):
                    try:
                        data = json.loads(line.strip())
                        # 添加标识字段
                        data['source_caption_type'] = 'manually-translated'
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        stats['translated'] += 1
                        stats['total'] += 1
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"⚠️  manually-translated 文件不存在或未指定: {translated_file}")
        
        # 读取 human-written 数据
        if human_file and os.path.exists(human_file):
            print(f"\n📖 读取 human-written 数据: {human_file}")
            with open(human_file, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="处理 human-written"):
                    try:
                        data = json.loads(line.strip())
                        # 添加标识字段
                        data['source_caption_type'] = 'human-written'
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        stats['human'] += 1
                        stats['total'] += 1
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"⚠️  human-written 文件不存在或未指定: {human_file}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("✅ 合并完成！")
    print(f"{'='*60}")
    print(f"📊 统计信息:")
    print(f"   manually-translated: {stats['translated']} 条")
    print(f"   human-written:       {stats['human']} 条")
    print(f"   总计:                {stats['total']} 条")
    print(f"\n💾 保存到: {output_file}")
    
    if stats['total'] == 0:
        print("\n⚠️  警告: 没有合并任何数据，请检查输入文件路径")

def main():
    parser = argparse.ArgumentParser(
        description="合并 COCO-CN 两种 caption 类型的数据集"
    )
    parser.add_argument(
        "--translated-file",
        type=str,
        default=None,
        help="manually-translated 类型的 JSONL 文件（可选）"
    )
    parser.add_argument(
        "--human-file",
        type=str,
        default=None,
        help="human-written 类型的 JSONL 文件（可选）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="输出的合并 JSONL 文件"
    )
    
    args = parser.parse_args()
    
    # 至少需要一个输入文件
    if not args.translated_file and not args.human_file:
        print("❌ 错误: 至少需要指定 --translated-file 或 --human-file 之一")
        return
    
    # 检查输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    merge_captions(args.translated_file, args.human_file, args.output_file)

if __name__ == "__main__":
    main()

