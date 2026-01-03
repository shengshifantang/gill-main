#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量删除图像脚本
功能：根据删除列表文件批量删除图像
"""

import os
import argparse
from tqdm import tqdm

def delete_images(delete_list, dry_run=False):
    """
    根据删除列表删除图像
    
    Args:
        delete_list: 包含图像路径的文件（每行一个路径）
        dry_run: 如果为 True，只显示将要删除的文件，不实际删除
    """
    if not os.path.exists(delete_list):
        print(f"❌ 删除列表文件不存在: {delete_list}")
        return
    
    # 读取删除列表
    print(f"📖 读取删除列表: {delete_list}")
    with open(delete_list, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"📊 共有 {len(image_paths)} 张图像待删除")
    
    # 计算总大小
    total_size = 0
    existing_count = 0
    for path in image_paths:
        if os.path.exists(path):
            total_size += os.path.getsize(path)
            existing_count += 1
    
    print(f"   其中 {existing_count} 张图像存在")
    print(f"   将释放约 {total_size / 1024 / 1024 / 1024:.2f} GB 磁盘空间")
    
    if dry_run:
        print(f"\n🔍 DRY RUN 模式：只显示前 10 个将要删除的文件")
        for i, path in enumerate(image_paths[:10]):
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024  # KB
                print(f"   {i+1}. {path} ({size:.1f} KB)")
        if len(image_paths) > 10:
            print(f"   ... 还有 {len(image_paths) - 10} 个文件")
        print(f"\n💡 使用 --confirm 参数执行实际删除")
        return
    
    # 确认删除
    print(f"\n⚠️  警告：即将删除 {existing_count} 张图像！")
    confirm = input("确认删除？(yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 取消删除")
        return
    
    # 执行删除
    print(f"\n🗑️  开始删除...")
    deleted_count = 0
    failed_count = 0
    
    for path in tqdm(image_paths, desc="删除进度"):
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"   ⚠️  删除失败: {path} - {e}")
    
    print(f"\n✅ 删除完成！")
    print(f"   成功删除: {deleted_count} 张")
    if failed_count > 0:
        print(f"   删除失败: {failed_count} 张")
    print(f"   释放空间: {total_size / 1024 / 1024 / 1024:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="批量删除图像脚本")
    parser.add_argument("--delete-list", type=str, required=True, 
                       help="包含图像路径的删除列表文件")
    parser.add_argument("--confirm", action="store_true", 
                       help="确认删除（不加此参数则为 dry-run 模式）")
    
    args = parser.parse_args()
    
    delete_images(args.delete_list, dry_run=not args.confirm)

if __name__ == "__main__":
    main()

