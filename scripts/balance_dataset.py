#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集平衡脚本
功能：从筛选后的数据中按比例采样，生成平衡的训练数据集
"""

import csv
import random
import argparse
from collections import defaultdict

def balance_dataset(input_csv, output_csv, strong_count, weak_count, negative_count, seed=42):
    """
    从输入 CSV 中按指定数量采样各类型数据
    
    Args:
        input_csv: 输入的筛选数据 CSV
        output_csv: 输出的平衡数据 CSV
        strong_count: Strong 样本数量
        weak_count: Weak 样本数量
        negative_count: Negative 样本数量
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取所有数据并按类型分组
    data_by_type = defaultdict(list)
    
    print(f"📖 读取数据: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) == 4:
                spatial_type = row[2]
                data_by_type[spatial_type].append(row)
    
    # 统计原始数据
    print(f"\n📊 原始数据统计:")
    for type_name, data in data_by_type.items():
        print(f"  {type_name}: {len(data)} 条")
    
    # 采样
    target_counts = {
        'strong': strong_count,
        'weak': weak_count,
        'negative': negative_count
    }
    
    sampled_data = []
    print(f"\n🎲 开始采样:")
    
    for type_name, target_count in target_counts.items():
        available = len(data_by_type[type_name])
        
        if available < target_count:
            print(f"  ⚠️  {type_name}: 需要 {target_count} 条，但只有 {available} 条，将使用全部数据")
            sampled = data_by_type[type_name]
        else:
            sampled = random.sample(data_by_type[type_name], target_count)
            print(f"  ✅ {type_name}: 从 {available} 条中采样 {target_count} 条")
        
        sampled_data.extend(sampled)
    
    # 打乱顺序
    random.shuffle(sampled_data)
    
    # 写入输出文件
    print(f"\n💾 保存到: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled_data)
    
    # 最终统计
    print(f"\n✅ 完成！")
    print(f"   总计: {len(sampled_data)} 条")
    print(f"   Strong: {sum(1 for row in sampled_data if row[2] == 'strong')} 条 ({sum(1 for row in sampled_data if row[2] == 'strong')/len(sampled_data)*100:.1f}%)")
    print(f"   Weak: {sum(1 for row in sampled_data if row[2] == 'weak')} 条 ({sum(1 for row in sampled_data if row[2] == 'weak')/len(sampled_data)*100:.1f}%)")
    print(f"   Negative: {sum(1 for row in sampled_data if row[2] == 'negative')} 条 ({sum(1 for row in sampled_data if row[2] == 'negative')/len(sampled_data)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="数据集平衡脚本")
    parser.add_argument("--input", type=str, required=True, help="输入的筛选数据 CSV")
    parser.add_argument("--output", type=str, required=True, help="输出的平衡数据 CSV")
    parser.add_argument("--strong", type=int, default=100000, help="Strong 样本数量（默认 100000）")
    parser.add_argument("--weak", type=int, default=50000, help="Weak 样本数量（默认 50000）")
    parser.add_argument("--negative", type=int, default=100000, help="Negative 样本数量（默认 100000）")
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
    
    balance_dataset(
        args.input,
        args.output,
        args.strong,
        args.weak,
        args.negative,
        args.seed
    )

if __name__ == "__main__":
    main()

