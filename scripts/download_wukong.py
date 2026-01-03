#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wukong 数据集图片下载脚本 (流式优化版)
功能：
 1. 分块读取 CSV，立即开始下载，不再等待大文件加载
 2. 多线程下载图片到本地
 3. 自动生成 JSONL 索引文件
 4. 增强的超时处理和进度显示
"""

import os
import json
import pandas as pd
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import hashlib
import time

# 全局统计
TOTAL_DOWNLOADED = 0
TOTAL_PROCESSED = 0

def download_one_image(row, save_dir, timeout=3):
    """
    下载单张图片并返回元数据
    """
    url = row.get('url')
    caption = row.get('caption')
    
    if not url or not isinstance(url, str):
        return None

    try:
        # 生成唯一文件名 (使用 URL hash 防止文件名冲突)
        img_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        img_filename = f"{img_hash}.jpg"
        img_path = os.path.join(save_dir, img_filename)

        # 如果文件已存在且大小正常，跳过下载
        if os.path.exists(img_path) and os.path.getsize(img_path) > 1024:
            return {
                "image_path": os.path.abspath(img_path),
                "caption": caption,
                "url": url
            }

        # 请求图片 (设置更短的连接超时)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # timeout=(连接超时, 读取超时)
        # 禁用代理，避免代理服务未运行时导致连接失败
        response = requests.get(url, headers=headers, stream=True, timeout=(3, 5), proxies={'http': None, 'https': None})
        
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # 再次检查文件是否有效
            if os.path.getsize(img_path) > 1024:
                return {
                    "image_path": os.path.abspath(img_path),
                    "caption": caption,
                    "url": url
                }
            else:
                if os.path.exists(img_path):
                    os.remove(img_path)
    except Exception:
        # Wukong 数据集死链很多，这是正常现象，直接忽略
        pass
    
    return None

def process_chunk(chunk, args, executor, f_out, chunk_idx):
    """
    处理一个小的数据块
    """
    global TOTAL_DOWNLOADED, TOTAL_PROCESSED
    
    # 列名适配
    if 'url' not in chunk.columns:
        if len(chunk.columns) >= 2:
            chunk.rename(columns={chunk.columns[0]: 'url', chunk.columns[1]: 'caption'}, inplace=True)
    
    if 'text' in chunk.columns and 'caption' not in chunk.columns:
        chunk.rename(columns={'text': 'caption'}, inplace=True)
    
    records = chunk.to_dict('records')
    chunk_size = len(records)
    
    # 立即显示开始处理的信息
    print(f"  📥 Chunk {chunk_idx}: Processing {chunk_size} URLs...", flush=True)
    
    futures = [executor.submit(download_one_image, rec, args.save_dir) for rec in records]
    
    # 使用tqdm显示进度
    with tqdm(total=chunk_size, desc=f"  Chunk {chunk_idx}", leave=False, ncols=80) as pbar:
        for future in as_completed(futures):
            result = future.result()
            TOTAL_PROCESSED += 1
            pbar.update(1)
            
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                TOTAL_DOWNLOADED += 1
                if TOTAL_DOWNLOADED % 50 == 0:
                    f_out.flush()
                pbar.set_postfix({"✅": TOTAL_DOWNLOADED, "📊": TOTAL_PROCESSED})
            
            # 每处理100个就输出一次总体进度（包括成功率）
            if TOTAL_PROCESSED % 100 == 0:
                success_rate = (TOTAL_DOWNLOADED / TOTAL_PROCESSED * 100) if TOTAL_PROCESSED > 0 else 0
                print(f"\r📊 Processed: {TOTAL_PROCESSED}, Downloaded: {TOTAL_DOWNLOADED} ({success_rate:.1f}%)", end="", flush=True)
            
            if args.max_samples and TOTAL_DOWNLOADED >= args.max_samples:
                return True # Stop signal
            
    return False # Continue signal

def main(args):
    global TOTAL_DOWNLOADED, TOTAL_PROCESSED
    
    # 1. 准备目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. 读取已有进度（断点续传支持）
    processed_urls = set()
    if os.path.exists(args.output_jsonl):
        print(f"📖 Reading existing progress from {args.output_jsonl}...")
        try:
            with open(args.output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'url' in data:
                            processed_urls.add(data['url'])
                    except:
                        pass
            print(f"✅ Found {len(processed_urls)} already downloaded images.")
            TOTAL_DOWNLOADED = len(processed_urls)  # 更新计数器
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing progress: {e}")
            processed_urls = set()
    
    # 3. 读取 CSV 文件列表
    # 支持两种方式：
    # - 如果 csv_dir 是文件，直接使用
    # - 如果 csv_dir 是目录，递归查找所有 CSV 文件
    if os.path.isfile(args.csv_dir):
        csv_files = [args.csv_dir]
        print(f"📦 Using single CSV file: {os.path.basename(args.csv_dir)}")
    else:
        csv_files = sorted([str(p) for p in Path(args.csv_dir).rglob("*.csv")])
        print(f"📦 Found {len(csv_files)} CSV files in {args.csv_dir}")
    
    if args.limit_csvs:
        csv_files = csv_files[:args.limit_csvs]

    # 创建线程池 (复用，不要反复创建销毁)
    executor = ThreadPoolExecutor(max_workers=args.workers)

    # --- 安全补丁：确保文件以换行符结尾（防止异常中断导致格式错误）---
    if os.path.exists(args.output_jsonl):
        try:
            with open(args.output_jsonl, 'rb+') as f:
                f.seek(0, 2)  # 移动到文件末尾
                if f.tell() > 0:  # 如果文件不为空
                    f.seek(-1, 2)  # 移动到倒数第一个字节
                    last_char = f.read(1)
                    if last_char != b'\n':
                        print("🔧 检测到上次运行未正常换行，正在自动修复...")
                        f.write(b'\n')
        except Exception as e:
            print(f"⚠️ 文件修复检查失败 (不影响运行): {e}")
    # ------------------------------------------------------------------------

    # 使用追加模式，避免清空已有数据
    file_mode = 'a' if os.path.exists(args.output_jsonl) else 'w'
    with open(args.output_jsonl, file_mode, encoding='utf-8') as f_out:
        for csv_file in csv_files:
            print(f"\n🚀 Processing {os.path.basename(csv_file)} in chunks...")
            
            try:
                # 关键修改：使用 chunksize 分块读取
                # 减小chunksize到1000，让进度更快出现
                chunk_iter = pd.read_csv(csv_file, on_bad_lines='skip', chunksize=1000)
                
                chunk_idx = 0
                for chunk in chunk_iter:
                    chunk_idx += 1
                    
                    # 过滤掉已经处理过的 URL（提升重启后的速度）
                    if 'url' in chunk.columns:
                        original_size = len(chunk)
                        chunk = chunk[~chunk['url'].isin(processed_urls)]
                        filtered_count = original_size - len(chunk)
                        if filtered_count > 0:
                            print(f"  ⏭️  Skipped {filtered_count} already processed URLs in chunk {chunk_idx}")
                    
                    # 如果 chunk 为空，跳过
                    if chunk.empty:
                        continue
                    
                    stop = process_chunk(chunk, args, executor, f_out, chunk_idx)
                    if stop:
                        print(f"\n🛑 Reached max samples: {args.max_samples}")
                        break
                
                if args.max_samples and TOTAL_DOWNLOADED >= args.max_samples:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error reading {csv_file}: {e}")

    executor.shutdown()
    print(f"\n\n🎉 Done! Processed: {TOTAL_PROCESSED}, Downloaded: {TOTAL_DOWNLOADED} images")
    print(f"💾 Metadata saved to {args.output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, 
                       help="Path to CSV file or directory containing CSV files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save images")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output path for wukong_raw.jsonl")
    parser.add_argument("--workers", type=int, default=64, help="Number of download threads")
    parser.add_argument("--limit_csvs", type=int, default=None, help="Only process first N csv files")
    parser.add_argument("--max_samples", type=int, default=None, help="Max images to download")
    args = parser.parse_args()
    main(args)
