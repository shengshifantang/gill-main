#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter results.jsonl by a zh->en map and optionally drop layout_text.

Usage:
  python scripts/filter_results_by_map.py \
    --input outputs/overfit_freeze_gate_phrase/results.jsonl \
    --output outputs/overfit_freeze_gate_phrase/results.filtered.jsonl \
    --zh-en-map outputs/zh_en_map.json \
    --drop-layout-text
"""

import argparse
import json
import os
import re

_MEASURE_RE = re.compile(r"^(一|二|三|四|五|六|七|八|九|十|两|几|多|每)?(个|只|条|张|把|台|部|辆|块|片|件|根|位|名|对|双|群)")
_NOISE_RE = re.compile(r"(正在|位于|看着|站在|坐在|躺在|趴在|穿着|拿着|走在|骑着)")


def clean_object_name(name: str, max_len: int = 10, min_len: int = 1) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"[\"'“”‘’（）()《》【】\[\]{}<>]", "", name)
    name = re.sub(r"[，,。\.、;；:：!?！？~`·•]", "", name)
    name = re.sub(r"\s+", "", name)
    if not name:
        return ""
    name = _MEASURE_RE.sub("", name)
    if not name:
        return ""
    if "的" in name:
        parts = [p for p in name.split("的") if p]
        if parts:
            name = parts[-1]
    if len(name) < min_len or len(name) > max_len:
        return ""
    if _NOISE_RE.search(name):
        return ""
    return name


def _load_zh_en_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        kz = clean_object_name(str(k), 10, 1)
        vz = str(v).strip().lower()
        if kz and vz:
            out[kz] = vz
    return out


def _filter_objects(objects, zh_en_map):
    kept = []
    for obj in objects or []:
        if not isinstance(obj, dict):
            continue
        name_zh = clean_object_name(obj.get("name", ""), 10, 1)
        if not name_zh:
            continue
        name_en = zh_en_map.get(name_zh)
        if not name_en:
            continue
        new_obj = dict(obj)
        new_obj["name_en"] = name_en
        kept.append(new_obj)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input results.jsonl")
    parser.add_argument("--output", required=True, help="Output filtered results.jsonl")
    parser.add_argument("--zh-en-map", required=True, help="zh-en map json")
    parser.add_argument("--drop-layout-text", action="store_true", help="Drop layout_text to force using layout.objects")
    args = parser.parse_args()

    zh_en_map = _load_zh_en_map(args.zh_en_map)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    kept_records = 0
    kept_objects = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            total += 1
            layout = record.get("layout") or {}
            objs = layout.get("objects") or record.get("objects") or []
            filtered = _filter_objects(objs, zh_en_map)
            if not filtered:
                continue
            if layout:
                if args.drop_layout_text and "layout_text" in layout:
                    layout.pop("layout_text", None)
                layout["objects"] = filtered
                record["layout"] = layout
            else:
                record["objects"] = filtered
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_records += 1
            kept_objects += len(filtered)

    print(f"[ok] total_records={total} kept_records={kept_records} kept_objects={kept_objects}")


if __name__ == "__main__":
    main()
