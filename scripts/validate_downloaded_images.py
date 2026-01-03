#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证已下载图片的完整性脚本

功能：
1. 读取 JSONL 文件，获取所有已下载的图片路径
2. 使用 PIL 验证每张图片的完整性
3. 删除损坏的图片，并更新 JSONL 文件（移除损坏图片的记录）

用法：
python scripts/validate_downloaded_images.py \
    --input /mnt/disk/lxh/gill_data/wukong_downloaded_500k.jsonl \
    --output /mnt/disk/lxh/gill_data/wukong_downloaded_validated.jsonl
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def validate_image_file(image_path: str) -> bool:
    """验证单张图片的完整性"""
    try:
        if not os.path.exists(image_path):
            return False
        
        # 检查文件大小
        if os.path.getsize(image_path) < 1024:
            return False
        
        # 使用 PIL 验证图片结构
        with Image.open(image_path) as img:
            img.verify()  # 验证文件结构
        
        # verify() 后需要重新打开才能读取像素数据
        with Image.open(image_path) as img:
            img.load()  # 加载像素数据，确保图片完整
        
        return True
    except Exception:
        return False


def main(args):
    """主函数"""
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    print(f"📖 读取输入文件: {args.input}")
    
    # 读取所有记录
    all_records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                all_records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"✅ 读取到 {len(all_records)} 条记录")
    
    # 验证图片
    valid_records = []
    invalid_count = 0
    missing_count = 0
    
    print(f"\n🔍 开始验证图片完整性...")
    for record in tqdm(all_records, desc="验证进度", unit="img"):
        image_path = record.get('image_path', '')
        
        if not image_path:
            invalid_count += 1
            continue
        
        # 处理相对路径
        if not os.path.isabs(image_path):
            if args.image_root:
                image_path = os.path.join(args.image_root, image_path)
            else:
                # 尝试从 JSONL 中的路径推断
                pass
        
        if not os.path.exists(image_path):
            missing_count += 1
            if args.remove_missing:
                continue  # 跳过缺失的图片
            else:
                valid_records.append(record)  # 保留记录（可能路径问题）
        elif validate_image_file(image_path):
            valid_records.append(record)
        else:
            # 图片损坏，删除文件
            invalid_count += 1
            try:
                os.remove(image_path)
                print(f"  🗑️  删除损坏图片: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"  ⚠️  删除失败 {image_path}: {e}")
    
    # 写入验证后的记录
    print(f"\n💾 写入验证后的记录到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"📊 验证完成统计")
    print(f"{'='*60}")
    print(f"总记录数: {len(all_records)}")
    print(f"有效图片: {len(valid_records)}")
    print(f"损坏图片: {invalid_count}")
    print(f"缺失图片: {missing_count}")
    if len(all_records) > 0:
        valid_rate = (len(valid_records) / len(all_records)) * 100
        print(f"有效率: {valid_rate:.2f}%")
    print(f"{'='*60}")
    
    if args.backup_original and args.input != args.output:
        import shutil
        backup_path = args.input + ".backup"
        print(f"\n💾 备份原文件到: {backup_path}")
        shutil.copy2(args.input, backup_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="验证已下载图片的完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径（只包含有效图片）"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="图片根目录（用于解析相对路径）"
    )
    parser.add_argument(
        "--remove-missing",
        action="store_true",
        help="移除缺失图片的记录（默认保留）"
    )
    parser.add_argument(
        "--backup-original",
        action="store_true",
        help="备份原始文件（添加 .backup 后缀）"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    main(args)

