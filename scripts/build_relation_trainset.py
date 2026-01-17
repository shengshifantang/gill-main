#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build relation-based training samples for Layout Planner.

Input: JSONL with GT objects (COCO-CN etc.)
Output: JSONL where each line has:
  - "prompt": relation instruction (e.g., "猫在桌子左边。")
  - "objects": GT boxes for the mentioned objects (x1,y1,x2,y2 in [0,1])

This is designed to improve relation accuracy by explicit supervision.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple


def _normalize_gt_objects(item: Dict, scale_mode: str = "auto") -> List[Dict]:
    objs = item.get("objects", []) or []
    if not isinstance(objs, list):
        return []
    width = float(item.get("width", 0) or 0)
    height = float(item.get("height", 0) or 0)
    has_dim = width > 0 and height > 0

    normed = []
    for idx, obj in enumerate(objs):
        if not isinstance(obj, dict):
            continue
        bbox_1000 = obj.get("bbox_1000")
        bbox = bbox_1000
        if not (isinstance(bbox, list) and len(bbox) == 4):
            bbox = obj.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            b = [float(v) for v in bbox]
        except Exception:
            continue

        max_val = max(b) if b else 0
        if max_val <= 1.5:
            b_norm = b
        else:
            if scale_mode == "pixel":
                if not has_dim:
                    continue
                b_norm = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
            elif scale_mode == "1000":
                b_norm = [v / 1000.0 for v in b]
            else:
                if isinstance(bbox_1000, list) and len(bbox_1000) == 4:
                    b_norm = [v / 1000.0 for v in b]
                elif max_val <= 1000:
                    if has_dim and max(width, height) > 0:
                        ratio = max_val / max(width, height)
                        if ratio > 1.2:
                            b_norm = [v / 1000.0 for v in b]
                        else:
                            b_norm = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
                    else:
                        b_norm = [v / 1000.0 for v in b]
                elif has_dim:
                    b_norm = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
                else:
                    continue

        b_norm = [max(0.0, min(1.0, v)) for v in b_norm]
        normed.append({
            "idx": idx,
            "name": obj.get("name", ""),
            "bbox": b_norm,
        })
    return normed


def _center(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _relation_from_gt(a: Dict, b: Dict, min_sep: float = 0.1) -> Optional[str]:
    ax, ay = _center(a["bbox"])
    bx, by = _center(b["bbox"])
    dx = ax - bx
    dy = ay - by
    if abs(dx) >= abs(dy) and abs(dx) >= min_sep:
        return "left" if dx < 0 else "right"
    if abs(dy) >= min_sep:
        return "above" if dy < 0 else "below"
    return None


def _relation_text(rel: str) -> str:
    if rel == "left":
        return "左边"
    if rel == "right":
        return "右边"
    if rel == "above":
        return "上方"
    if rel == "below":
        return "下方"
    return "附近"

def _relation_templates(rel: str) -> List[str]:
    if rel == "left":
        return [
            "{a}在{b}左边。",
            "{a}在{b}左侧。",
            "{a}在{b}左面。",
            "{a}位于{b}左侧。",
            "{b}的左边是{a}。",
            "{b}左侧是{a}。",
        ]
    if rel == "right":
        return [
            "{a}在{b}右边。",
            "{a}在{b}右侧。",
            "{a}在{b}右面。",
            "{a}位于{b}右侧。",
            "{b}的右边是{a}。",
            "{b}右侧是{a}。",
        ]
    if rel == "above":
        return [
            "{a}在{b}上方。",
            "{a}在{b}上面。",
            "{a}位于{b}上方。",
            "{b}的上方是{a}。",
            "{b}上方是{a}。",
        ]
    if rel == "below":
        return [
            "{a}在{b}下方。",
            "{a}在{b}下面。",
            "{a}位于{b}下方。",
            "{b}的下方是{a}。",
            "{b}下方是{a}。",
        ]
    return ["{a}在{b}附近。"]


def _make_prompt(a_name: str, b_name: str, rel: str, mode: str, rng: random.Random) -> str:
    if mode == "single":
        return f"{a_name}在{b_name}{_relation_text(rel)}。"
    templates = _relation_templates(rel)
    tmpl = templates[rng.randrange(len(templates))]
    return tmpl.format(a=a_name, b=b_name)


def _iter_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relation training set for Layout Planner")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with GT objects")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL")
    parser.add_argument("--gt-field", default="objects", choices=["objects", "annotations"])
    parser.add_argument("--gt-bbox-scale", default="auto", choices=["auto", "pixel", "1000"])
    parser.add_argument("--max-per-image", type=int, default=2)
    parser.add_argument("--min-sep", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-same-class", action="store_true", help="Drop pairs with identical class names")
    parser.add_argument("--dedup-global", action="store_true", help="Deduplicate by prompt + bboxes")
    parser.add_argument("--max-per-name", type=int, default=-1, help="Cap samples per object name (optional)")
    parser.add_argument(
        "--template-mode",
        choices=["single", "multi"],
        default="multi",
        help="Prompt template mode. 'multi' adds template variation.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_items = list(_iter_items(args.input_jsonl))
    if args.shuffle:
        rng.shuffle(all_items)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    kept = 0
    total_pairs = 0
    total_images = 0
    total_pairs_generated = 0
    rel_count = {"left": 0, "right": 0, "above": 0, "below": 0}
    name_count = {}
    seen = set()
    with open(args.output_jsonl, "w", encoding="utf-8") as w:
        for item in all_items:
            if args.max_samples > 0 and kept >= args.max_samples:
                break
            total_images += 1

            ann = item.get("annotations", {}) if args.gt_field == "annotations" else {}
            objs = ann.get("objects", []) if args.gt_field == "annotations" else item.get("objects", [])
            gt_item = {
                "objects": objs,
                "width": item.get("width", 0),
                "height": item.get("height", 0),
            }
            gt_objects = _normalize_gt_objects(gt_item, scale_mode=args.gt_bbox_scale)
            if len(gt_objects) < 2:
                continue

            pairs = []
            for i in range(len(gt_objects)):
                for j in range(len(gt_objects)):
                    if i == j:
                        continue
                    rel = _relation_from_gt(gt_objects[i], gt_objects[j], min_sep=args.min_sep)
                    if rel is None:
                        continue
                    pairs.append((gt_objects[i], gt_objects[j], rel))

            if not pairs:
                continue
            rng.shuffle(pairs)
            pairs = pairs[: args.max_per_image]
            total_pairs_generated += len(pairs)
            for a, b, rel in pairs:
                if args.max_samples > 0 and kept >= args.max_samples:
                    break
                if args.no_same_class and (a["name"] == b["name"]):
                    continue
                prompt = _make_prompt(a["name"], b["name"], rel, args.template_mode, rng)
                record = {
                    "prompt": prompt,
                    "objects": [
                        {"name": a["name"], "bbox": a["bbox"]},
                        {"name": b["name"], "bbox": b["bbox"]},
                    ],
                    "source": "relation_synthetic",
                }
                if args.dedup_global:
                    key = (prompt, tuple(round(x, 4) for o in record["objects"] for x in o["bbox"]))
                    if key in seen:
                        continue
                    seen.add(key)
                if args.max_per_name and args.max_per_name > 0:
                    name_count[a["name"]] = name_count.get(a["name"], 0) + 1
                    if name_count[a["name"]] > args.max_per_name:
                        continue
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                total_pairs += 1
                rel_count[rel] = rel_count.get(rel, 0) + 1

    print(f"Done. Wrote {kept} samples.")
    if total_images > 0:
        print(f"Images scanned: {total_images}")
        print(f"Pairs generated (pre-filter): {total_pairs_generated}")
        print(f"Avg pairs per image: {total_pairs_generated / total_images:.3f}")
    if kept > 0:
        print("Relation distribution:")
        for k in ["left", "right", "above", "below"]:
            v = rel_count.get(k, 0)
            print(f"  {k}: {v} ({v / kept:.3f})")
        if name_count:
            top = sorted(name_count.items(), key=lambda x: x[1], reverse=True)[:10]
            print("Top names:")
            for n, c in top:
                print(f"  {n}: {c}")


if __name__ == "__main__":
    main()
