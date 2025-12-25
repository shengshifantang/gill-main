#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从多个悟空子数据集（TSV）构建统一的 caption JSONL：
  - 读取 wukong_train.tsv / wukong_val.tsv
  - 生成包含 image_path(相对 DATA_ROOT) + caption 的 JSONL
  - 自动去重（同一 image_path 只保留第一条）

示例用法：
python scripts/build_wukong_caption_jsonl.py \
  --data-root /mnt/disk/lxh/gill_data \
  --train-tsv /mnt/disk/lxh/gill_data/wukong_500k/wukong_train.tsv \
  --val-tsv /mnt/disk/lxh/gill_data/wukong_500k/wukong_val.tsv \
  --out /mnt/disk/lxh/gill_data/wukong_with_caption.jsonl
"""

import argparse
import csv
import json
import os
from typing import Set


def process_tsv(tsv_path: str, prefix: str, seen: Set[str], fout, data_root: str):
    if not os.path.exists(tsv_path):
        print(f"[警告] 找不到 TSV 文件: {tsv_path}")
        return

    print(f"读取 TSV: {tsv_path} (prefix={prefix})")
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        total, used = 0, 0
        for row in reader:
            total += 1
            caption = row.get("caption", "").strip()
            img_name = row.get("image_path") or row.get("image") or ""
            img_name = img_name.strip()
            if not img_name:
                continue
            # 统一为相对 DATA_ROOT 的路径，例如 "wukong_500k/images/00012345.jpg"
            rel_path = os.path.join(prefix, img_name)
            # 去重：同一相对路径仅保留第一条
            if rel_path in seen:
                continue
            # 可选：检查文件是否真实存在；避免无意义条目
            abs_path = os.path.join(data_root, rel_path)
            if not os.path.exists(abs_path):
                continue

            item = {
                "image_path": rel_path,
                "caption": caption,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            seen.add(rel_path)
            used += 1

        print(f"  总行数: {total}, 写入: {used}, 当前累计去重后样本数: {len(seen)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="图片根目录，对应脚本中 DATA_ROOT（例如 /mnt/disk/lxh/gill_data）",
    )
    parser.add_argument(
        "--train-tsv",
        required=True,
        help="训练集 TSV 路径（如 wukong_train.tsv）",
    )
    parser.add_argument(
        "--val-tsv",
        required=False,
        help="验证集 TSV 路径（如 wukong_val.tsv），可选",
    )
    parser.add_argument(
        "--images-prefix",
        default="wukong_500k/images",
        help="TSV 中 image_path 前需要拼接的前缀（相对 data-root），默认 wukong_500k/images",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出 JSONL 路径（例如 /mnt/disk/lxh/gill_data/wukong_with_caption.jsonl）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen: Set[str] = set()

    with open(args.out, "w", encoding="utf-8") as fout:
        process_tsv(
            args.train_tsv,
            prefix=args.images_prefix,
            seen=seen,
            fout=fout,
            data_root=args.data_root,
        )

        if args.val_tsv:
            process_tsv(
                args.val_tsv,
                prefix=args.images_prefix,
                seen=seen,
                fout=fout,
                data_root=args.data_root,
            )

    print(f"✅ 完成构建 caption JSONL，最终样本数: {len(seen)}")
    print(f"输出文件: {args.out}")


if __name__ == "__main__":
    main()


