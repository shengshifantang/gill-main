#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速过滤图片（尺寸 >= min_size）并生成索引

用法示例：
python scripts/filter_wukong_images.py \
  --root /mnt/disk/lxh/gill_data \
  --out /mnt/disk/lxh/gill_data/filter_index.txt \
  --min-size 256 \
  --workers 8
"""

import argparse
import glob
import os
from multiprocessing import Pool
from typing import Optional, Tuple

from PIL import Image


def check_image(path: str, root: str, min_size: int) -> Optional[str]:
    if not path.lower().endswith(("jpg", "jpeg", "png")):
        return None
    try:
        with Image.open(path) as img:
            w, h = img.size
            if min(w, h) >= min_size:
                return os.path.relpath(path, root)
    except Exception:
        return None
    return None


def worker(args) -> Optional[str]:
    path, root, min_size = args
    return check_image(path, root, min_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="图片根目录")
    parser.add_argument("--out", required=True, help="输出索引文件路径")
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pattern", default="**/*.*", help="glob 模式")
    args = parser.parse_args()

    print(f"扫描目录: {args.root}")
    all_files = glob.glob(os.path.join(args.root, args.pattern), recursive=True)
    print(f"总文件数: {len(all_files)}，开始过滤 (min_size={args.min_size}) ...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    kept = 0
    with Pool(processes=args.workers) as pool, open(args.out, "w") as fw:
        for idx, res in enumerate(pool.imap_unordered(worker, ((f, args.root, args.min_size) for f in all_files), chunksize=200)):
            if res:
                fw.write(res + "\n")
                kept += 1
            if (idx + 1) % 5000 == 0:
                print(f"进度: {idx+1}/{len(all_files)}, 已保留 {kept}")

    print(f"完成。保留 {kept} / {len(all_files)}，输出 -> {args.out}")


if __name__ == "__main__":
    main()

