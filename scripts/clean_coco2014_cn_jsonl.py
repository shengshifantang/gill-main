#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean COCO2014 CN jsonl by removing invalid/tiny boxes.

Usage:
  python scripts/clean_coco2014_cn_jsonl.py \
    --input data/coco2014_cn_train.jsonl \
    --output data/coco2014_cn_train_clean.jsonl \
    --min-area 1e-4
"""

import argparse
import json


def _valid_bbox(b):
    if not isinstance(b, list) or len(b) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return False
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return False
    return True


def _area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-area", type=float, default=1e-4, help="drop boxes with area < min-area (normalized)")
    parser.add_argument("--keep-empty", action="store_true", help="keep samples with zero valid objects")
    args = parser.parse_args()

    total = 0
    parse_err = 0
    samples_out = 0
    objects_in = 0
    objects_out = 0
    bad_bbox = 0
    tiny_bbox = 0
    dropped_empty = 0

    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                parse_err += 1
                continue

            objs = item.get("objects", [])
            if not isinstance(objs, list):
                continue

            objects_in += len(objs)
            kept = []
            for obj in objs:
                b = obj.get("bbox")
                if not _valid_bbox(b):
                    bad_bbox += 1
                    continue
                b = [float(v) for v in b]
                if _area(b) < args.min_area:
                    tiny_bbox += 1
                    continue
                obj["bbox"] = b
                kept.append(obj)

            if not kept and not args.keep_empty:
                dropped_empty += 1
                continue

            item["objects"] = kept
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            samples_out += 1
            objects_out += len(kept)

    print(f"[ok] total={total} parse_err={parse_err}")
    print(f"[ok] objects_in={objects_in} objects_out={objects_out}")
    print(f"[ok] bad_bbox={bad_bbox} tiny_bbox={tiny_bbox} dropped_empty={dropped_empty}")
    print(f"[ok] samples_out={samples_out} -> {args.output}")


if __name__ == "__main__":
    main()
