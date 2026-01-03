#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计下载成功的图片中 strong、weak、negative 的分类比例
"""
import json
import csv
from collections import Counter

def main():
    # 读取下载成功的JSONL文件
    downloaded_urls = set()
    jsonl_file = '/mnt/disk/lxh/gill_data/wukong_downloaded_500k.jsonl'
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        url = data.get('url', '')
                        if url:
                            downloaded_urls.add(url)
                    except:
                        continue
        downloaded_count = len(downloaded_urls)
        print(f"✅ 已读取 {downloaded_count:,} 条成功下载的记录")
    except FileNotFoundError:
        print(f"❌ JSONL文件不存在: {jsonl_file}")
        return
    except Exception as e:
        print(f"❌ 读取JSONL文件失败: {e}")
        return

    # 读取原始筛选CSV，匹配分类
    csv_file = '/mnt/disk/lxh/gill_data/wukong_filtered_spatial_500k.csv'
    classifications = Counter()
    matched_count = 0

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url', '').strip()
                if url in downloaded_urls:
                    spatial_type = row.get('spatial_type', 'unknown').strip()
                    classifications[spatial_type] += 1
                    matched_count += 1
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return

    print(f"\n{'='*60}")
    print(f"📊 下载成功的图片分类统计")
    print(f"{'='*60}")
    print(f"   总下载成功: {downloaded_count:,} 张")
    print(f"   匹配到分类: {matched_count:,} 张")
    print(f"   未匹配记录: {downloaded_count - matched_count:,} 张")
    
    if matched_count > 0:
        print(f"\n分类分布:")
        print(f"{'─'*60}")
        for cls in ['strong', 'weak', 'negative']:
            count = classifications.get(cls, 0)
            pct = count / matched_count * 100
            bar = '█' * int(pct / 2)
            print(f"   {cls:10s}: {count:6,} ({pct:5.2f}%) {bar}")
        
        print(f"\n{'─'*60}")
        print(f"   总计:      {matched_count:6,} (100.00%)")
        
        # 计算与原始数据的对比
        print(f"\n{'='*60}")
        print(f"📈 与原始筛选数据的对比")
        print(f"{'='*60}")
        
        # 读取原始CSV的总体分布
        original_classifications = Counter()
        total_original = 0
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spatial_type = row.get('spatial_type', 'unknown').strip()
                    original_classifications[spatial_type] += 1
                    total_original += 1
        except:
            pass
        
        if total_original > 0:
            print(f"\n原始筛选数据分布:")
            for cls in ['strong', 'weak', 'negative']:
                orig_count = original_classifications.get(cls, 0)
                orig_pct = orig_count / total_original * 100
                dl_count = classifications.get(cls, 0)
                dl_pct = dl_count / matched_count * 100 if matched_count > 0 else 0
                diff = dl_pct - orig_pct
                print(f"   {cls:10s}: 原始 {orig_pct:5.2f}% → 下载 {dl_pct:5.2f}% (差异: {diff:+5.2f}%)")

if __name__ == '__main__':
    main()
