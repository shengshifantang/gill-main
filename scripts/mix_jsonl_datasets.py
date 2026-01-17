#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mix two JSONL datasets with a target ratio.

Example:
  python scripts/mix_jsonl_datasets.py \
    --primary data/layout_planner_mixed_80_20.jsonl \
    --secondary data/relation_train.jsonl \
    --ratio-secondary 0.2 \
    --output data/layout_planner_with_relation.jsonl \
    --shuffle \
    --seed 42
"""

import argparse
import json
import os
import random
from typing import List


def _load_jsonl(path: str, max_rows: int = -1) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix two JSONL datasets with a target ratio")
    parser.add_argument("--primary", required=True, help="Primary JSONL path")
    parser.add_argument("--secondary", required=True, help="Secondary JSONL path")
    parser.add_argument("--ratio-secondary", type=float, default=0.2, help="Target fraction of secondary in output")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-total", type=int, default=-1, help="Optional max total samples")
    parser.add_argument("--max-primary", type=int, default=-1, help="Optional cap for primary samples")
    parser.add_argument("--max-secondary", type=int, default=-1, help="Optional cap for secondary samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.ratio_secondary < 1.0):
        raise ValueError("--ratio-secondary must be in (0,1)")

    primary = _load_jsonl(args.primary, args.max_primary)
    secondary = _load_jsonl(args.secondary, args.max_secondary)

    if not primary or not secondary:
        raise ValueError("Primary/secondary dataset is empty")

    # Decide total size
    if args.max_total and args.max_total > 0:
        total = args.max_total
    else:
        # Fit secondary to ratio relative to primary size
        total = len(primary) + int(len(primary) * args.ratio_secondary / (1 - args.ratio_secondary))

    target_secondary = int(total * args.ratio_secondary)
    target_primary = total - target_secondary

    rng = random.Random(args.seed)
    rng.shuffle(primary)
    rng.shuffle(secondary)

    primary = primary[:target_primary]
    secondary = secondary[:target_secondary]

    mixed = primary + secondary
    if args.shuffle:
        rng.shuffle(mixed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line.strip() + "\n")

    print(f"Primary: {len(primary)}")
    print(f"Secondary: {len(secondary)}")
    print(f"Total: {len(mixed)}")
    print(f"Secondary ratio: {len(secondary) / len(mixed):.3f}")


if __name__ == "__main__":
    main()
