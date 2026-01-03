#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSONL 数据集平衡脚本（基于实际下载的图像）
功能：
 1. 读取标注后的 JSONL 文件
 2. 按比例采样各类型数据
 3. 生成平衡的 JSONL 文件
 4. 输出需要删除的图像列表
"""

import json
import random
import argparse
import os
from collections import defaultdict

def balance_jsonl_dataset(input_jsonl, output_jsonl, strong_count, weak_count, negative_count, 
                          delete_list=None, seed=42):
    """
    从输入 JSONL 中按指定数量采样各类型数据
    
    Args:
        input_jsonl: 输入的标注数据 JSONL
        output_jsonl: 输出的平衡数据 JSONL
        strong_count: Strong 样本数量
        weak_count: Weak 样本数量
        negative_count: Negative 样本数量
        delete_list: 输出需要删除的图像路径列表文件
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取所有数据并按类型分组
    data_by_type = defaultdict(list)
    
    print(f"📖 读取数据: {input_jsonl}")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                spatial_type = data.get('spatial_type', 'unknown')
                
                # 检查图像文件是否存在
                image_path = data.get('image_path')
                if image_path and os.path.exists(image_path):
                    data_by_type[spatial_type].append(data)
                else:
                    print(f"  ⚠️  图像不存在，跳过: {image_path}")
            except json.JSONDecodeError:
                continue
    
    # 统计原始数据
    print(f"\n📊 原始数据统计（实际下载成功的图像）:")
    total_available = 0
    for type_name, data in sorted(data_by_type.items()):
        count = len(data)
        total_available += count
        print(f"  {type_name}: {count} 条")
    print(f"  总计: {total_available} 条")
    
    # 采样
    target_counts = {
        'strong': strong_count,
        'weak': weak_count,
        'negative': negative_count
    }
    
    sampled_data = []
    discarded_data = []  # 未被选中的数据
    
    print(f"\n🎲 开始采样:")
    
    for type_name, target_count in target_counts.items():
        available = len(data_by_type[type_name])
        
        if available == 0:
            print(f"  ⚠️  {type_name}: 需要 {target_count} 条，但没有可用数据")
            continue
        
        if available < target_count:
            print(f"  ⚠️  {type_name}: 需要 {target_count} 条，但只有 {available} 条，将使用全部数据")
            sampled = data_by_type[type_name]
            discarded = []
        else:
            # 随机采样
            sampled = random.sample(data_by_type[type_name], target_count)
            # 找出未被选中的数据
            sampled_set = set(id(item) for item in sampled)
            discarded = [item for item in data_by_type[type_name] if id(item) not in sampled_set]
            print(f"  ✅ {type_name}: 从 {available} 条中采样 {target_count} 条，丢弃 {len(discarded)} 条")
        
        sampled_data.extend(sampled)
        discarded_data.extend(discarded)
    
    # 打乱顺序
    random.shuffle(sampled_data)
    
    # 写入输出文件
    print(f"\n💾 保存平衡数据到: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for data in sampled_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 写入删除列表
    if delete_list and discarded_data:
        print(f"💾 保存删除列表到: {delete_list}")
        with open(delete_list, 'w', encoding='utf-8') as f:
            for data in discarded_data:
                image_path = data.get('image_path', '')
                if image_path:
                    f.write(image_path + '\n')
        
        # 计算可节省的磁盘空间
        total_size = 0
        for data in discarded_data:
            image_path = data.get('image_path', '')
            if image_path and os.path.exists(image_path):
                total_size += os.path.getsize(image_path)
        
        print(f"   将删除 {len(discarded_data)} 张图像，节省约 {total_size / 1024 / 1024 / 1024:.2f} GB 磁盘空间")
    
    # 最终统计
    print(f"\n✅ 完成！")
    print(f"   总计: {len(sampled_data)} 条")
    
    type_counts = defaultdict(int)
    for data in sampled_data:
        type_counts[data.get('spatial_type', 'unknown')] += 1
    
    for type_name in ['strong', 'weak', 'negative']:
        count = type_counts[type_name]
        percentage = count / len(sampled_data) * 100 if sampled_data else 0
        print(f"   {type_name.capitalize()}: {count} 条 ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="JSONL 数据集平衡脚本")
    parser.add_argument("--input", type=str, required=True, help="输入的标注数据 JSONL")
    parser.add_argument("--output", type=str, required=True, help="输出的平衡数据 JSONL")
    parser.add_argument("--strong", type=int, default=100000, help="Strong 样本数量（默认 100000）")
    parser.add_argument("--weak", type=int, default=50000, help="Weak 样本数量（默认 50000）")
    parser.add_argument("--negative", type=int, default=100000, help="Negative 样本数量（默认 100000）")
    parser.add_argument("--delete-list", type=str, default=None, 
                       help="输出需要删除的图像路径列表文件（可选）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--preset", type=str, choices=['balanced', 'spatial', 'test'], 
                       help="预设方案：balanced(平衡), spatial(空间增强), test(测试)")
    
    args = parser.parse_args()
    
    # 应用预设方案
    if args.preset == 'balanced':
        print("📋 使用预设方案：平衡训练（推荐）")
        args.strong = 100000
        args.weak = 50000
        args.negative = 100000
    elif args.preset == 'spatial':
        print("📋 使用预设方案：空间增强")
        args.strong = 150000
        args.weak = 50000
        args.negative = 50000
    elif args.preset == 'test':
        print("📋 使用预设方案：轻量级测试")
        args.strong = 20000
        args.weak = 10000
        args.negative = 20000
    
    balance_jsonl_dataset(
        args.input,
        args.output,
        args.strong,
        args.weak,
        args.negative,
        args.delete_list,
        args.seed
    )

if __name__ == "__main__":
    main()

