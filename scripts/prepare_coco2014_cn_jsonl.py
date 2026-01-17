#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare COCO2014 jsonl for layout planner + spatial adapter.

Output schema (per line):
{
  "image_path": "train2014/COCO_train2014_000000xxxx.jpg",
  "caption": "<zh caption>",
  "width": 640,
  "height": 480,
  "objects": [
    {"name": "人", "name_en": "person", "category_id": 1,
     "bbox": [x1,y1,x2,y2], "bbox_1000":[...]}
  ]
}
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Any


# COCO 80 category id -> Chinese name (same as train_layout_planner.py)
COCO_ID_TO_ZH = {
    1: "人", 2: "自行车", 3: "汽车", 4: "摩托车", 5: "飞机", 6: "公交车", 7: "火车", 8: "卡车",
    9: "船", 10: "交通灯", 11: "消防栓", 12: "停止标志", 13: "停车计时器", 14: "长椅", 15: "鸟",
    16: "猫", 17: "狗", 18: "马", 19: "羊", 20: "牛", 21: "大象", 22: "熊", 23: "斑马",
    24: "长颈鹿", 25: "背包", 26: "雨伞", 27: "手提包", 28: "领带", 29: "行李箱", 30: "飞盘",
    31: "滑雪板", 32: "滑雪板", 33: "运动球", 34: "风筝", 35: "棒球棒", 36: "棒球手套",
    37: "滑板", 38: "冲浪板", 39: "网球拍", 40: "瓶子", 41: "酒杯", 42: "杯子", 43: "叉子",
    44: "刀", 45: "勺子", 46: "碗", 47: "香蕉", 48: "苹果", 49: "三明治", 50: "橙子",
    51: "西兰花", 52: "胡萝卜", 53: "热狗", 54: "披萨", 55: "甜甜圈", 56: "蛋糕", 57: "椅子",
    58: "沙发", 59: "盆栽", 60: "床", 61: "餐桌", 62: "厕所", 63: "电视", 64: "笔记本电脑",
    65: "鼠标", 66: "遥控器", 67: "键盘", 68: "手机", 69: "微波炉", 70: "烤箱", 71: "烤面包机",
    72: "水槽", 73: "冰箱", 74: "书", 75: "时钟", 76: "花瓶", 77: "剪刀", 78: "泰迪熊",
    79: "吹风机", 80: "牙刷"
}


def load_coco_captions(path: str) -> Dict[int, List[str]]:
    out = defaultdict(list)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                img_id = item.get("image_id")
                if img_id is None:
                    continue
                cap = item.get("caption_zh") or item.get("caption") or ""
                cap = str(cap).strip()
                if cap:
                    out[int(img_id)].append(cap)
        return out

    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict) and "annotations" in data:
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cap = ann.get("caption", "")
            if img_id is not None and cap:
                out[int(img_id)].append(str(cap))
    elif isinstance(data, dict):
        # mapping image_id -> caption(s)
        for k, v in data.items():
            try:
                img_id = int(k)
            except Exception:
                continue
            if isinstance(v, list):
                out[img_id].extend([str(x) for x in v if str(x).strip()])
            elif isinstance(v, str):
                if v.strip():
                    out[img_id].append(v.strip())
    return out


def load_coco_instances(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    images = {int(x["id"]): x for x in data.get("images", [])}
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann.get("image_id"))
        anns_by_img[img_id].append(ann)
    cat_id_to_en = {int(c["id"]): str(c["name"]).strip() for c in data.get("categories", [])}
    return images, anns_by_img, cat_id_to_en


def coco_bbox_xyxy(bbox_xywh, w, h):
    x, y, bw, bh = bbox_xywh
    x1 = max(0.0, min(float(x), float(w)))
    y1 = max(0.0, min(float(y), float(h)))
    x2 = max(0.0, min(float(x + bw), float(w)))
    y2 = max(0.0, min(float(y + bh), float(h)))
    return [x1, y1, x2, y2]


def norm_xyxy(bbox_xyxy, w, h):
    x1, y1, x2, y2 = bbox_xyxy
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True, help="COCO root with train2014/val2014 and annotations")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--captions-en", required=True, help="captions_*.json")
    parser.add_argument("--instances", required=True, help="instances_*.json")
    parser.add_argument("--captions-zh", default=None, help="optional translated captions json")
    parser.add_argument("--output", required=True, help="output jsonl")
    parser.add_argument("--max-captions-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=0.0, help="min normalized area (0-1)")
    parser.add_argument("--min-side", type=float, default=0.0, help="min normalized side length (0-1)")
    parser.add_argument("--max-objects", type=int, default=0, help="0 means no limit")
    parser.add_argument("--relative-path", action="store_true", help="store image_path relative to coco-root")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images, anns_by_img, cat_id_to_en = load_coco_instances(args.instances)
    captions_en = load_coco_captions(args.captions_en)
    captions_zh = load_coco_captions(args.captions_zh) if args.captions_zh else {}

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    kept = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for img_id, info in images.items():
            total += 1
            file_name = info.get("file_name", "")
            if not file_name:
                continue
            w = float(info.get("width", 0))
            h = float(info.get("height", 0))
            if w <= 0 or h <= 0:
                continue

            image_path = os.path.join(args.split + "2014", file_name)
            if not args.relative_path:
                image_path = os.path.join(args.coco_root, image_path)

            caps = captions_zh.get(img_id) or captions_en.get(img_id) or []
            if not caps:
                continue
            if args.max_captions_per_image > 0 and len(caps) > args.max_captions_per_image:
                caps = rng.sample(caps, args.max_captions_per_image)

            objects = []
            for ann in anns_by_img.get(img_id, []):
                bbox_xywh = ann.get("bbox", [])
                if not bbox_xywh or len(bbox_xywh) != 4:
                    continue
                bbox_xyxy = coco_bbox_xyxy(bbox_xywh, w, h)
                bbox_norm = norm_xyxy(bbox_xyxy, w, h)
                bw = max(0.0, bbox_norm[2] - bbox_norm[0])
                bh = max(0.0, bbox_norm[3] - bbox_norm[1])
                area = bw * bh
                if args.min_area and area < args.min_area:
                    continue
                if args.min_side and (bw < args.min_side or bh < args.min_side):
                    continue

                cat_id = int(ann.get("category_id", 0))
                name_en = cat_id_to_en.get(cat_id, "object")
                name_zh = COCO_ID_TO_ZH.get(cat_id, name_en)

                objects.append({
                    "name": name_zh,
                    "name_en": name_en,
                    "category_id": cat_id,
                    "bbox": [round(x, 6) for x in bbox_norm],
                    "bbox_1000": [round(x * 1000.0, 2) for x in bbox_norm],
                })

            if args.max_objects and args.max_objects > 0 and len(objects) > args.max_objects:
                objects = objects[: args.max_objects]

            if not objects:
                continue

            for cap in caps:
                cap = str(cap).strip()
                if not cap:
                    continue
                rec = {
                    "image_path": image_path,
                    "caption": cap,
                    "width": w,
                    "height": h,
                    "objects": objects,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[ok] total_images={total} samples_written={kept}")


if __name__ == "__main__":
    main()
