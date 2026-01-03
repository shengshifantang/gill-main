#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并 spatial_type 信息到已下载的 JSONL
功能：
 1. 读取筛选 CSV 文件，建立 URL -> spatial_type 的映射
 2. 读取已下载的 JSONL 文件
 3. 根据 URL 匹配，添加 spatial_type 字段
 4. 输出新的 JSONL 文件
"""

import csv
import json
import argparse
import os
from tqdm import tqdm

def merge_spatial_type(csv_file, input_jsonl, output_jsonl):
    """
    合并 spatial_type 信息
    
    Args:
        csv_file: 筛选后的 CSV 文件（包含 url, caption, spatial_type, reason）
        input_jsonl: 已下载的 JSONL 文件（包含 image_path, caption, url）
        output_jsonl: 输出的 JSONL 文件（添加 spatial_type 字段）
    """
    
    # 步骤 1：读取 CSV，建立 URL -> spatial_type 映射
    print(f"📖 读取筛选 CSV: {csv_file}")
    url_to_type = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        
        for row in reader:
            if len(row) >= 3:
                url = row[0]
                spatial_type = row[2]
                url_to_type[url] = spatial_type
    
    print(f"   共读取 {len(url_to_type)} 条 URL 映射")
    
    # 步骤 2：读取 JSONL，添加 spatial_type
    print(f"\n📖 读取已下载 JSONL: {input_jsonl}")
    
    matched_count = 0
    unmatched_count = 0
    type_counts = {'strong': 0, 'weak': 0, 'negative': 0, 'unknown': 0}
    
    with open(input_jsonl, 'r', encoding='utf-8') as f_in:
        with open(output_jsonl, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="处理进度"):
                try:
                    data = json.loads(line.strip())
                    url = data.get('url', '')
                    
                    # 查找对应的 spatial_type
                    if url in url_to_type:
                        spatial_type = url_to_type[url]
                        matched_count += 1
                    else:
                        spatial_type = 'unknown'
                        unmatched_count += 1
                    
                    # 添加 spatial_type 字段
                    data['spatial_type'] = spatial_type
                    type_counts[spatial_type] = type_counts.get(spatial_type, 0) + 1
                    
                    # 写入输出文件
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    continue
    
    # 统计结果
    print(f"\n✅ 合并完成！")
    print(f"   匹配成功: {matched_count} 条")
    print(f"   匹配失败: {unmatched_count} 条")
    print(f"\n📊 类型分布:")
    total = sum(type_counts.values())
    for type_name in ['strong', 'weak', 'negative', 'unknown']:
        count = type_counts.get(type_name, 0)
        percentage = count / total * 100 if total > 0 else 0
        print(f"   {type_name}: {count} 条 ({percentage:.1f}%)")
    print(f"   总计: {total} 条")
    print(f"\n💾 保存到: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="合并 spatial_type 信息到已下载的 JSONL")
    parser.add_argument("--csv", type=str, required=True, 
                       help="筛选后的 CSV 文件")
    parser.add_argument("--input", type=str, required=True, 
                       help="已下载的 JSONL 文件")
    parser.add_argument("--output", type=str, required=True, 
                       help="输出的 JSONL 文件")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv):
        print(f"❌ CSV 文件不存在: {args.csv}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ JSONL 文件不存在: {args.input}")
        return
    
    merge_spatial_type(args.csv, args.input, args.output)

if __name__ == "__main__":
    main()

