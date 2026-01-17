#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a simple overfit ablation: adapter-scale=0 vs adapter-scale=1
using layout_jsonl (fixed layouts), then evaluate with name_en.
"""

import argparse
import os
import subprocess
import sys


def _run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Overfit ablation runner")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--layout-adapter", required=True)
    parser.add_argument("--layout-jsonl", required=True)
    parser.add_argument("--spatial-adapter", required=True)
    parser.add_argument("--kolors-model", default="./model/Kolors")
    parser.add_argument("--output-root", default="outputs/overfit_clean")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-scale-on", type=float, default=1.0)
    parser.add_argument("--adapter-scale-off", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling", type=float, default=1.0)
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--sd-device", default="cuda:1")
    parser.add_argument("--gill-device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--visual-encoder", default="auto")
    parser.add_argument("--smart-expand", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--force-gate", action="store_true", help="Override adapter gate values for debug")
    parser.add_argument("--gate-value", type=float, default=3.0, help="Gate value when --force-gate is set")
    parser.add_argument("--train-freeze-gate", action="store_true",
                        help="Run train_spatial_adapter.py with --freeze-gate before ablation")
    parser.add_argument("--train-data", default=None, help="Training jsonl (default: --layout-jsonl)")
    parser.add_argument("--train-image-dir", default=None, help="Image dir for training (required if --train-freeze-gate)")
    parser.add_argument("--train-output-dir", default=None, help="Output dir for training (default: output_root + '_train')")
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--train-scale-min", type=float, default=1.0)
    parser.add_argument("--train-scale-max", type=float, default=1.0)
    parser.add_argument("--train-phrase-dropout", type=float, default=0.0)
    parser.add_argument("--train-save-every", type=int, default=200)
    parser.add_argument("--train-gate-value", type=float, default=1.0)

    parser.add_argument("--dino-backend", default="official", choices=["auto", "hf", "official"])
    parser.add_argument("--dino-config", required=True)
    parser.add_argument("--dino-ckpt", required=True)
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--dino-text-threshold", type=float, default=0.2)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--use-name-en", action="store_true", default=True)
    args = parser.parse_args()

    out_no = f"{args.output_root}_noadapter"
    out_on = f"{args.output_root}_adapter"
    res_no = os.path.join(out_no, "results.jsonl")
    res_on = os.path.join(out_on, "results.jsonl")

    adapter_ckpt = args.spatial_adapter
    if args.train_freeze_gate:
        train_data = args.train_data or args.layout_jsonl
        if not args.train_image_dir:
            raise SystemExit("--train-image-dir is required when --train-freeze-gate is set")
        train_out = args.train_output_dir or (args.output_root + "_train")
        os.makedirs(train_out, exist_ok=True)
        train_cmd = [
            sys.executable, "scripts/train_spatial_adapter.py",
            "--mixed-data", train_data,
            "--kolors-model", args.kolors_model,
            "--output-dir", train_out,
            "--batch-size", str(args.train_batch_size),
            "--epochs", str(args.train_epochs),
            "--lr", str(args.train_lr),
            "--image-dir", args.train_image_dir,
            "--scale-min", str(args.train_scale_min),
            "--scale-max", str(args.train_scale_max),
            "--phrase-dropout", str(args.train_phrase_dropout),
            "--save-every", str(args.train_save_every),
            "--save-epoch",
            "--freeze-gate",
            "--gate-value", str(args.train_gate_value),
        ]
        _run(train_cmd)
        final_ckpt = os.path.join(train_out, "spatial_adapter_final.pt")
        if not os.path.exists(final_ckpt):
            raise SystemExit(f"Expected trained checkpoint not found: {final_ckpt}")
        adapter_ckpt = final_ckpt
    if args.force_gate:
        import torch
        tmp_dir = args.output_root + "_debug"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_ckpt = os.path.join(tmp_dir, "spatial_adapter_gate.pt")
        sd = torch.load(adapter_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k in list(sd.keys()):
            if k.endswith("gate") and hasattr(sd[k], "numel") and sd[k].numel() == 1:
                sd[k].fill_(float(args.gate_value))
        torch.save(sd, tmp_ckpt)
        adapter_ckpt = tmp_ckpt

    if not args.skip_generate:
        base_cmd = [
            sys.executable, "scripts/run_full_chain_samples.py",
            "--base-model", args.base_model,
            "--layout-adapter", args.layout_adapter,
            "--layout-jsonl", args.layout_jsonl,
            "--spatial-adapter", adapter_ckpt,
            "--kolors-model", args.kolors_model,
            "--max-samples", str(args.max_samples),
            "--scheduled-sampling", str(args.scheduled_sampling),
            "--visual-encoder", args.visual_encoder,
            "--disable-verifier",
            "--llm-device", args.llm_device,
            "--sd-device", args.sd_device,
            "--gill-device", args.gill_device,
            "--seed", str(args.seed),
        ]
        if args.offline:
            base_cmd.append("--offline")
        if args.smart_expand:
            base_cmd.append("--smart-expand")

        cmd_no = base_cmd + [
            "--output-dir", out_no,
            "--results-jsonl", res_no,
            "--adapter-scale", str(args.adapter_scale_off),
        ]
        _run(cmd_no)

        cmd_on = base_cmd + [
            "--output-dir", out_on,
            "--results-jsonl", res_on,
            "--adapter-scale", str(args.adapter_scale_on),
        ]
        _run(cmd_on)

    eval_base = [
        sys.executable, "scripts/eval_spatial_adapter_small.py",
        "--max-samples", str(args.max_samples),
        "--seed", str(args.seed),
        "--dino-backend", args.dino_backend,
        "--dino-config", args.dino_config,
        "--dino-ckpt", args.dino_ckpt,
        "--dino-threshold", str(args.dino_threshold),
        "--dino-text-threshold", str(args.dino_text_threshold),
        "--iou-threshold", str(args.iou_threshold),
    ]
    if args.use_name_en:
        eval_base.append("--use-name-en")

    _run(eval_base + ["--results", res_no])
    _run(eval_base + ["--results", res_on])


if __name__ == "__main__":
    main()
