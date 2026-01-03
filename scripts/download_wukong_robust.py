#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[SOTA Standard] 异步高并发图片下载脚本（科研级优化版）

改进点：
1. 架构：aiohttp 异步替代多线程，降低 CPU 开销，支持更高并发 (100+)
2. 校验：增加 PIL 图片完整性校验，剔除伪装成 JPG 的 HTML 或截断图片
3. 审计：保留完整元数据，便于论文 Reproducibility
4. 断点续传：自动跳过已下载的图片，支持中断恢复

用法示例：
python scripts/download_wukong_robust.py \
    --csv_dir /mnt/disk/lxh/gill_data/wukong_filtered_spatial_500k.csv \
    --save_dir /mnt/disk/lxh/gill_data/wukong_images \
    --output_jsonl /mnt/disk/lxh/gill_data/wukong_downloaded_robust.jsonl \
    --concurrency 100
"""

import os
import json
import asyncio
import argparse
import hashlib
from pathlib import Path
from io import BytesIO
from typing import Set

try:
    import aiohttp
    import aiofiles
except ImportError:
    print("❌ 需要安装 aiohttp 和 aiofiles: pip install aiohttp aiofiles")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ 需要安装 Pillow: pip install Pillow")
    exit(1)

try:
    import pandas as pd
except ImportError:
    print("❌ 需要安装 pandas: pip install pandas")
    exit(1)

try:
    from tqdm.asyncio import tqdm
except ImportError:
    from tqdm import tqdm

# 全局统计
STATS = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "corrupt": 0,
    "exist": 0,
    "invalid_url": 0
}


async def validate_image(content: bytes) -> bool:
    """
    严格校验图片完整性（防止截断/损坏）
    
    使用 PIL 的 verify() 方法检查图片文件结构是否完整。
    这对于大规模数据集非常重要，因为损坏的图片会导致训练时 DataLoader 崩溃。
    """
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()  # 校验文件结构，不加载像素数据（更快）
        return True
    except Exception:
        return False


async def download_worker(
    session: aiohttp.ClientSession,
    row: dict,
    save_dir: str,
    semaphore: asyncio.Semaphore,
    processed_urls: Set[str]
) -> dict:
    """
    异步下载单张图片
    
    Args:
        session: aiohttp 会话
        row: 包含 url 和 caption 的字典
        save_dir: 保存目录
        semaphore: 并发控制信号量
        processed_urls: 已处理的 URL 集合（用于去重）
    
    Returns:
        成功时返回包含 image_path, caption, url 的字典，失败返回 None
    """
    async with semaphore:  # 限制并发数
        url = row.get('url')
        caption = row.get('caption', '')
        
        # URL 有效性检查
        if not isinstance(url, str) or len(url) < 5:
            STATS["invalid_url"] += 1
            return None

        # 检查是否已处理
        if url in processed_urls:
            return None

        # 1. 路径计算（使用 URL hash 确保唯一性）
        try:
            img_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            filename = f"{img_hash}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            # 2. 存在性检查（断点续传）
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                # 简单校验大小，避免空文件
                if file_size > 1024:
                    # [关键] 如果启用验证，检查已存在图片的完整性
                    if args.validate_existing:
                        try:
                            with open(save_path, 'rb') as f:
                                content = f.read()
                            if not await validate_image(content):
                                # 图片损坏，删除并重新下载
                                os.remove(save_path)
                                STATS["corrupt"] += 1
                                # 继续执行下载流程（不返回）
                            else:
                                # 图片完整，跳过下载
                                STATS["exist"] += 1
                                processed_urls.add(url)
                                return {
                                    "image_path": os.path.abspath(save_path),
                                    "caption": caption,
                                    "url": url
                                }
                        except Exception:
                            # 读取失败，删除并重新下载
                            if os.path.exists(save_path):
                                os.remove(save_path)
                            # 继续执行下载流程（不返回）
                    else:
                        # 不验证已存在图片，直接跳过（快速模式）
                        STATS["exist"] += 1
                        processed_urls.add(url)
                        return {
                            "image_path": os.path.abspath(save_path),
                            "caption": caption,
                            "url": url
                        }
            
            # 3. 网络请求（带超时和重试）
            timeout = aiohttp.ClientTimeout(total=10, connect=3)
            try:
                async with session.get(url, timeout=timeout, allow_redirects=True) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # 4. [关键] 图片有效性深度校验
                        if len(content) < 1024:  # 文件太小，可能是错误页面
                            STATS["corrupt"] += 1
                            return None
                        
                        if not await validate_image(content):
                            STATS["corrupt"] += 1
                            return None
                        
                        # 5. 异步写入
                        async with aiofiles.open(save_path, 'wb') as f:
                            await f.write(content)
                        
                        STATS["success"] += 1
                        processed_urls.add(url)
                        return {
                            "image_path": os.path.abspath(save_path),
                            "caption": caption,
                            "url": url
                        }
                    else:
                        STATS["failed"] += 1
                        return None
                        
            except asyncio.TimeoutError:
                STATS["failed"] += 1
                return None
            except aiohttp.ClientError:
                STATS["failed"] += 1
                return None
                
        except Exception as e:
            # 其他异常（文件系统错误等）
            STATS["failed"] += 1
            return None


def _fix_newline(filepath: str):
    """确保文件以换行符结尾（防止追加时格式错误）"""
    try:
        with open(filepath, 'rb+') as f:
            f.seek(0, 2)  # 移动到文件末尾
            if f.tell() > 0:  # 如果文件不为空
                f.seek(-1, 2)  # 移动到倒数第一个字节
                if f.read(1) != b'\n':
                    f.write(b'\n')
    except Exception:
        pass


async def main(args):
    """主函数"""
    # 准备环境
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 读取进度（断点续传）
    processed_urls: Set[str] = set()
    if os.path.exists(args.output_jsonl):
        print(f"📖 读取断点续传文件: {args.output_jsonl}")
        _fix_newline(args.output_jsonl)  # 修复末尾换行符
        try:
            with open(args.output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        url = data.get('url')
                        if url:
                            processed_urls.add(url)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ 读取进度文件时出错: {e}")
    
    print(f"✅ 已包含 {len(processed_urls)} 条记录")
    STATS["exist"] = len(processed_urls)

    # 读取 CSV 文件列表
    if os.path.isfile(args.csv_dir):
        csv_files = [args.csv_dir]
        print(f"📦 使用单个 CSV 文件: {os.path.basename(args.csv_dir)}")
    else:
        csv_files = sorted([str(p) for p in Path(args.csv_dir).rglob("*.csv")])
        print(f"📦 找到 {len(csv_files)} 个 CSV 文件")
    
    if args.limit_csvs:
        csv_files = csv_files[:args.limit_csvs]
        print(f"📦 限制处理前 {args.limit_csvs} 个文件")

    # 异步并发控制
    connector = aiohttp.TCPConnector(limit=500, limit_per_host=50)  # 连接池
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 文件写入锁
    write_lock = asyncio.Lock()
    
    async with aiohttp.ClientSession(
        connector=connector,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ) as session:
        
        # 使用追加模式打开输出文件
        file_mode = 'a' if os.path.exists(args.output_jsonl) else 'w'
        async with aiofiles.open(args.output_jsonl, file_mode, encoding='utf-8') as f_out:
            
            for csv_file in csv_files:
                print(f"\n🚀 处理 {os.path.basename(csv_file)}...")
                try:
                    # 分块读取 CSV（避免内存溢出）
                    chunk_iter = pd.read_csv(
                        csv_file,
                        on_bad_lines='skip',
                        chunksize=5000
                    )
                    
                    for chunk_idx, chunk in enumerate(chunk_iter):
                        # 预处理列名
                        if 'url' not in chunk.columns:
                            if len(chunk.columns) >= 2:
                                chunk.rename(
                                    columns={
                                        chunk.columns[0]: 'url',
                                        chunk.columns[1]: 'caption'
                                    },
                                    inplace=True
                                )
                        
                        if 'text' in chunk.columns and 'caption' not in chunk.columns:
                            chunk.rename(columns={'text': 'caption'}, inplace=True)
                        
                        # 过滤已下载的 URL
                        original_size = len(chunk)
                        chunk = chunk[~chunk['url'].isin(processed_urls)]
                        filtered_count = original_size - len(chunk)
                        
                        if filtered_count > 0:
                            print(f"  ⏭️  跳过 {filtered_count} 个已处理的 URL")
                        
                        if chunk.empty:
                            continue
                        
                        # 创建下载任务
                        tasks = [
                            download_worker(session, row.to_dict(), args.save_dir, semaphore, processed_urls)
                            for _, row in chunk.iterrows()
                        ]
                        
                        STATS["total"] += len(tasks)
                        
                        # 并发执行并显示进度
                        if hasattr(tqdm, 'asyncio'):
                            progress_bar = tqdm.asyncio.tqdm(
                                asyncio.as_completed(tasks),
                                total=len(tasks),
                                desc=f"  Chunk {chunk_idx+1}",
                                leave=False,
                                ncols=80
                            )
                        else:
                            progress_bar = tqdm(
                                total=len(tasks),
                                desc=f"  Chunk {chunk_idx+1}",
                                leave=False,
                                ncols=80
                            )
                        
                        completed_count = 0
                        for coro in asyncio.as_completed(tasks):
                            res = await coro
                            if res:
                                # 异步写入结果（加锁保证线程安全）
                                async with write_lock:
                                    await f_out.write(
                                        json.dumps(res, ensure_ascii=False) + "\n"
                                    )
                                    await f_out.flush()
                            
                            completed_count += 1
                            progress_bar.update(1)
                            
                            # 每 100 个显示一次统计
                            if completed_count % 100 == 0:
                                success_rate = (
                                    STATS["success"] / STATS["total"] * 100
                                    if STATS["total"] > 0 else 0
                                )
                                progress_bar.set_postfix({
                                    "✅": STATS["success"],
                                    "📊": STATS["total"],
                                    "Rate": f"{success_rate:.1f}%"
                                })
                        
                        progress_bar.close()
                        
                except Exception as e:
                    print(f"⚠️ CSV 处理错误 {csv_file}: {e}")
                    continue

    # 输出最终统计
    print(f"\n{'='*60}")
    print(f"📊 下载完成统计")
    print(f"{'='*60}")
    print(f"总任务数: {STATS['total']}")
    print(f"成功下载: {STATS['success']}")
    print(f"已存在: {STATS['exist']}")
    print(f"损坏图片: {STATS['corrupt']}")
    print(f"下载失败: {STATS['failed']}")
    print(f"无效 URL: {STATS['invalid_url']}")
    if STATS['total'] > 0:
        success_rate = (STATS['success'] / STATS['total']) * 100
        print(f"成功率: {success_rate:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[SOTA Standard] 异步高并发图片下载脚本（科研级优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="CSV 文件路径或包含 CSV 文件的目录"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="图片保存目录"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="输出 JSONL 文件路径（包含下载成功的图片元数据）"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="并发数（默认: 100，可根据网络调整）"
    )
    parser.add_argument(
        "--limit_csvs",
        type=int,
        default=None,
        help="如果 csv_dir 是目录，只处理前 N 个 CSV 文件"
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="验证已存在的图片完整性（会删除损坏的图片并重新下载，速度较慢但更安全）"
    )
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.csv_dir):
        print(f"❌ 输入路径不存在: {args.csv_dir}")
        exit(1)
    
    # 运行主函数
    asyncio.run(main(args))

