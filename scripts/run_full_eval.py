#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click evaluation runner for Layout Planner.

Runs:
  1) Relation evaluation
  2) Format/parse verification
  3) Caption-only diagnostic (optional)
"""

import argparse
import os
import subprocess
import sys


def _run(cmd, env=None):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--base-model", required=True, help="LLM base model path")
    parser.add_argument("--layout-adapter", required=True, help="Layout Planner LoRA adapter path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=["bf16", "fp16"])
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES override")

    # relation eval
    parser.add_argument("--relation-input", default="/mnt/disk/lxh/gill_data/coco-cn/coco-cn_val_filtered.jsonl")
    parser.add_argument("--relation-output", default="results/relation_eval.jsonl")
    parser.add_argument("--relation-max-samples", type=int, default=500)
    parser.add_argument("--relation-max-per-image", type=int, default=2)
    parser.add_argument("--relation-min-sep", type=float, default=0.12)
    parser.add_argument("--gt-bbox-scale", default="1000", choices=["auto", "pixel", "1000"])

    # verify
    parser.add_argument("--verify-input", default="data/layout_planner_mixed_80_20.jsonl")
    parser.add_argument("--verify-output", default="results/verify_layout.jsonl")
    parser.add_argument("--verify-max-samples", type=int, default=200)

    # caption-only diagnostic
    parser.add_argument("--caption-input", default="data/coco-cn_val_caption_only.jsonl")
    parser.add_argument("--caption-max-samples", type=int, default=300)
    parser.add_argument("--caption-print-examples", type=int, default=5)

    parser.add_argument("--skip-relation", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    python = sys.executable

    if not args.skip_relation:
        cmd = [
            python, "scripts/relation_eval_layout_planner.py",
            "--input-jsonl", args.relation_input,
            "--base-model", args.base_model,
            "--layout-adapter", args.layout_adapter,
            "--device", args.device,
            "--dtype", args.dtype,
            "--gt-bbox-scale", args.gt_bbox_scale,
            "--max-samples", str(args.relation_max_samples),
            "--max-per-image", str(args.relation_max_per_image),
            "--min-sep", str(args.relation_min_sep),
            "--output-jsonl", args.relation_output,
        ]
        if args.dry_run:
            print(">>", " ".join(cmd))
        else:
            _run(cmd, env=env)

    if not args.skip_verify:
        cmd = [
            python, "scripts/verify_layout.py",
            "--base-model", args.base_model,
            "--adapter-path", args.layout_adapter,
            "--input-jsonl", args.verify_input,
            "--max-samples", str(args.verify_max_samples),
            "--output-jsonl", args.verify_output,
            "--device", args.device,
        ]
        if args.dry_run:
            print(">>", " ".join(cmd))
        else:
            _run(cmd, env=env)

    if not args.skip_caption:
        cmd = [
            python, "scripts/one_shot_layout_eval_debug.py",
            "--input-jsonl", args.caption_input,
            "--base-model", args.base_model,
            "--layout-adapter", args.layout_adapter,
            "--device", args.device,
            "--dtype", args.dtype,
            "--gt-bbox-scale", args.gt_bbox_scale,
            "--max-samples", str(args.caption_max_samples),
            "--print-examples", str(args.caption_print_examples),
        ]
        if args.dry_run:
            print(">>", " ".join(cmd))
        else:
            _run(cmd, env=env)


if __name__ == "__main__":
    main()
