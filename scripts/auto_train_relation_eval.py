#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto train + relation-eval loop.

Workflow:
  1) Train for N epochs (incremental, resume from latest checkpoint).
  2) Run relation_eval_layout_planner.py to get relation_acc.
  3) Continue if relation_acc improves; allow small relation drop only when eval_loss improves.

This script is designed to stop early once relation_acc stops improving.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple


def _run(cmd, env=None, capture=False):
    print("\n>>", " ".join(cmd))
    if capture:
        return subprocess.run(cmd, check=True, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, check=True, env=env)


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not paths:
        return None
    def _step(p: str) -> int:
        try:
            return int(os.path.basename(p).split("-")[-1])
        except Exception:
            return -1
    paths = sorted(paths, key=_step)
    return paths[-1]


def _read_trainer_state(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_eval_loss(output_dir: str) -> Tuple[Optional[float], Optional[float]]:
    # Returns (eval_loss, epoch)
    ckpt = _latest_checkpoint(output_dir)
    state_path = os.path.join(ckpt, "trainer_state.json") if ckpt else os.path.join(output_dir, "trainer_state.json")
    state = _read_trainer_state(state_path)
    if not state:
        return None, None
    logs = [x for x in state.get("log_history", []) if "eval_loss" in x]
    if not logs:
        return None, None
    last = logs[-1]
    return last.get("eval_loss"), last.get("epoch")


def _current_epoch(output_dir: str) -> float:
    ckpt = _latest_checkpoint(output_dir)
    state_path = os.path.join(ckpt, "trainer_state.json") if ckpt else os.path.join(output_dir, "trainer_state.json")
    state = _read_trainer_state(state_path)
    if not state:
        return 0.0
    epochs = [x.get("epoch", 0.0) for x in state.get("log_history", []) if "epoch" in x]
    return max(epochs) if epochs else 0.0


def _parse_relation_acc(stdout: str) -> Optional[float]:
    # relation_eval prints JSON summary; parse relation_acc
    m = re.search(r"\"relation_acc\"\\s*:\\s*([0-9\\.]+)", stdout)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Try to extract JSON block from summary
    m = re.search(r"==== Summary ====.*?(\\{.*?\\})", stdout, re.S)
    if m:
        try:
            data = json.loads(m.group(1))
            if "relation_acc" in data:
                return float(data["relation_acc"])
        except Exception:
            pass
    # Try line-by-line JSON
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or "relation_acc" not in line:
            continue
        try:
            data = json.loads(line)
            if "relation_acc" in data:
                return float(data["relation_acc"])
        except Exception:
            continue
    return None


def _relation_acc_from_output(output_jsonl: str) -> Optional[float]:
    if not output_jsonl or not os.path.exists(output_jsonl):
        return None
    matched = 0
    ok = 0
    try:
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("pred_a") and r.get("pred_b"):
                    matched += 1
                    if r.get("relation_ok"):
                        ok += 1
    except Exception:
        return None
    if matched == 0:
        return None
    return ok / matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto train + relation eval loop")
    # Training
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto", choices=["auto", "none"])
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--train-nproc", type=int, default=1, help="Use torchrun with N processes if >1")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES")

    # Relation eval
    parser.add_argument("--relation-input", required=True)
    parser.add_argument("--relation-output", default="results/relation_eval_auto.jsonl")
    parser.add_argument("--relation-max-samples", type=int, default=500)
    parser.add_argument("--relation-max-per-image", type=int, default=2)
    parser.add_argument("--relation-min-sep", type=float, default=0.12)
    parser.add_argument("--gt-bbox-scale", default="1000", choices=["auto", "pixel", "1000"])
    parser.add_argument("--name-normalize", action="store_true")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--eval-dtype", default="fp16", choices=["bf16", "fp16"])

    # Early stop strategy
    parser.add_argument("--rel-min-delta", type=float, default=0.002, help="Min improvement in relation_acc")
    parser.add_argument("--rel-tolerance", type=float, default=0.003, help="Allowed relation_acc drop when eval_loss improves")
    parser.add_argument("--loss-min-delta", type=float, default=0.01, help="Min eval_loss improvement to offset small rel drop")

    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    python = sys.executable
    log_path = os.path.join(args.output_dir, "auto_train_eval_log.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    best_rel = None
    best_loss = None

    for round_idx in range(1, args.max_rounds + 1):
        current = _current_epoch(args.output_dir)
        current_int = int(current)
        target_epoch = current_int + args.epochs_per_round

        # build training command
        train_cmd = []
        if args.train_nproc and args.train_nproc > 1:
            train_cmd = ["torchrun", "--nproc_per_node", str(args.train_nproc), "scripts/train_layout_planner.py"]
        else:
            train_cmd = [python, "scripts/train_layout_planner.py"]

        train_cmd += [
            "--layout-json", args.layout_json,
            "--val-json", args.val_json,
            "--base-model", args.base_model,
            "--output-dir", args.output_dir,
            "--epochs", str(int(target_epoch)),
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--lr", str(args.lr),
            "--save-total-limit", str(args.save_total_limit),
            "--seed", str(args.seed),
            "--device-map", args.device_map,
        ]
        if args.use_lora:
            train_cmd.append("--use-lora")
        if args.max_memory_gb is not None:
            train_cmd += ["--max-memory-gb", str(args.max_memory_gb)]
        if args.gradient_checkpointing:
            train_cmd.append("--gradient-checkpointing")

        ckpt = _latest_checkpoint(args.output_dir)
        if ckpt:
            train_cmd += ["--resume-from-checkpoint", ckpt]
        elif args.adapter_path:
            train_cmd += ["--adapter-path", args.adapter_path]

        _run(train_cmd, env=env, capture=False)

        eval_loss, eval_epoch = _latest_eval_loss(args.output_dir)

        # run relation eval
        eval_cmd = [
            python, "scripts/relation_eval_layout_planner.py",
            "--input-jsonl", args.relation_input,
            "--base-model", args.base_model,
            "--layout-adapter", os.path.join(args.output_dir, "final"),
            "--device", args.eval_device,
            "--dtype", args.eval_dtype,
            "--gt-bbox-scale", args.gt_bbox_scale,
            "--max-samples", str(args.relation_max_samples),
            "--max-per-image", str(args.relation_max_per_image),
            "--min-sep", str(args.relation_min_sep),
            "--output-jsonl", args.relation_output,
        ]
        if args.name_normalize:
            eval_cmd.append("--name-normalize")

        res = _run(eval_cmd, env=env, capture=True)
        stdout = res.stdout or ""
        rel_acc = _parse_relation_acc(stdout)
        if rel_acc is None:
            rel_acc = _relation_acc_from_output(args.relation_output)
            if rel_acc is not None:
                print(f"Note: relation_acc parsed from output JSONL: {rel_acc:.6f}")

        record = {
            "round": round_idx,
            "target_epoch": target_epoch,
            "eval_loss": eval_loss,
            "eval_epoch": eval_epoch,
            "relation_acc": rel_acc,
            "checkpoint": _latest_checkpoint(args.output_dir),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if rel_acc is None:
            print("Warning: relation_acc not found; stopping.")
            break

        if best_rel is None or rel_acc >= best_rel + args.rel_min_delta:
            best_rel = rel_acc
            if eval_loss is not None:
                best_loss = eval_loss if best_loss is None else min(best_loss, eval_loss)
            continue

        # allow small relation drop if eval_loss improves enough
        if best_rel is not None and rel_acc >= best_rel - args.rel_tolerance:
            if eval_loss is not None and (best_loss is None or eval_loss <= best_loss - args.loss_min_delta):
                best_loss = eval_loss if best_loss is None else min(best_loss, eval_loss)
                continue

        print("Early stop: relation_acc did not improve and eval_loss not sufficiently better.")
        break


if __name__ == "__main__":
    main()
