#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter/normalize JSONL data for Spatial Adapter training.

Expected input format per line:
{
  "image_path": "...",
  "caption": "...",
  "objects": [{"name": "...", "bbox": [x1,y1,x2,y2]}]  # or bbox_1000
}

This script:
  - drops samples with missing images (optional)
  - drops samples with empty caption
  - normalizes bbox to [0,1] if values look like 0-1000
  - removes invalid boxes and optionally small/huge boxes
  - optionally caps max objects per sample (keeps largest boxes)
  - can drop samples with no valid objects
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


def _load_bbox(obj: Dict) -> List[float]:
    bbox = obj.get("bbox", None)
    if bbox is None:
        bbox = obj.get("bbox_1000", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(x) for x in bbox]
        except Exception:
            return []
    return []


def _normalize_bbox(bbox: List[float]) -> List[float]:
    if not bbox or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    max_v = max(x1, y1, x2, y2)
    min_v = min(x1, y1, x2, y2)
    # If looks like 0-1000, normalize.
    if max_v > 1.5 and max_v <= 1000 and min_v >= 0:
        return [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
    return [x1, y1, x2, y2]


def _valid_box(bbox: List[float], min_area: float, max_area: float, min_side: float) -> Tuple[bool, float]:
    if not bbox or len(bbox) != 4:
        return False, 0.0
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return False, 0.0
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    w = x2 - x1
    h = y2 - y1
    area = w * h
    if area < min_area or area > max_area:
        return False, 0.0
    if w < min_side or h < min_side:
        return False, 0.0
    return True, area


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Spatial Adapter training JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--image-dir", default=None, help="Optional image root dir")
    parser.add_argument("--require-image", action="store_true", default=True, help="Drop samples with missing images")
    parser.add_argument("--keep-no-layout", action="store_true", default=False, help="Keep samples without valid boxes")
    parser.add_argument("--min-area", type=float, default=0.02, help="Min box area in normalized coords")
    parser.add_argument("--max-area", type=float, default=0.90, help="Max box area in normalized coords")
    parser.add_argument("--min-side", type=float, default=0.03, help="Min box side length in normalized coords")
    parser.add_argument("--max-objects", type=int, default=20, help="Max objects per sample")
    args = parser.parse_args()

    total = 0
    kept = 0
    dropped_no_caption = 0
    dropped_no_image = 0
    dropped_no_objects = 0
    dropped_invalid = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                dropped_invalid += 1
                continue

            caption = item.get("caption") or item.get("text") or item.get("prompt") or ""
            caption = caption.strip() if isinstance(caption, str) else ""
            if not caption:
                dropped_no_caption += 1
                continue

            image_path = item.get("image_path", "")
            if isinstance(image_path, str) and image_path and args.image_dir and not os.path.isabs(image_path):
                image_path = os.path.join(args.image_dir, image_path)
            if args.require_image:
                if not image_path or not os.path.exists(image_path):
                    dropped_no_image += 1
                    continue

            objects = item.get("objects", []) or []
            filtered: List[Dict] = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                bbox_raw = _load_bbox(obj)
                if not bbox_raw:
                    continue
                bbox = _normalize_bbox(bbox_raw)
                ok, area = _valid_box(bbox, args.min_area, args.max_area, args.min_side)
                if not ok:
                    continue
                name = obj.get("name", "")
                filtered.append({"name": name, "bbox": bbox, "_area": area})

            if not filtered and not args.keep_no_layout:
                dropped_no_objects += 1
                continue

            # keep largest boxes if too many
            if args.max_objects > 0 and len(filtered) > args.max_objects:
                filtered.sort(key=lambda x: x["_area"], reverse=True)
                filtered = filtered[: args.max_objects]

            for obj in filtered:
                obj.pop("_area", None)

            out = {
                "image_path": item.get("image_path", ""),
                "caption": caption,
                "objects": filtered,
                "has_layout": len(filtered) > 0,
            }
            # carry through extra fields if needed
            if "url" in item:
                out["url"] = item["url"]
            if "spatial_type" in item:
                out["spatial_type"] = item["spatial_type"]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print("==== Filter Summary ====")
    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Dropped (no caption): {dropped_no_caption}")
    print(f"Dropped (missing image): {dropped_no_image}")
    print(f"Dropped (no valid objects): {dropped_no_objects}")
    print(f"Dropped (invalid json): {dropped_invalid}")


if __name__ == "__main__":
    main()
