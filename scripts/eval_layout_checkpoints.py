#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate layout format stability across multiple checkpoints.

Outputs a CSV summary with:
  - format_ok_rate
  - parsed_ok_rate (incl. <no_layout>)
  - parsed_objects_ok_rate
  - no_layout_rate

Usage:
  python scripts/eval_layout_checkpoints.py \
    --base-model ./model/qwen2.5-7B-Instruct \
    --checkpoint-dir ./checkpoints/layout_planner_v2 \
    --input-jsonl data/layout_planner_mixed_80_20.jsonl \
    --max-samples 200 \
    --device cuda:0 \
    --output-csv results/layout_ckpt_eval.csv
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from gill.layout_planner import format_layout_input, parse_layout_output


def _get_prompt(item: Dict) -> str:
    for key in ("caption", "prompt", "text"):
        val = item.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _reservoir_sample_prompts(path: str, k: int, seed: int) -> List[str]:
    import random
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            prompt = _get_prompt(item)
            if not prompt:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(prompt)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = prompt
    return reservoir


def _load_model_and_tokenizer(base_model: str, adapter_path: str, device: str):
    if os.path.exists(adapter_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]})

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _clean_text(text: str) -> str:
    text = text.strip()
    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        text = text.replace(token, "").strip()
    return text


def _eval_prompts(model, tokenizer, prompts: List[str]) -> Tuple[int, int, int, int]:
    total = 0
    format_ok = 0
    parsed_ok = 0  # includes <no_layout>
    parsed_obj_ok = 0
    no_layout = 0

    for prompt in prompts:
        formatted = format_layout_input(tokenizer, prompt, enable_cot=False, feedback=None)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = _clean_text(tokenizer.decode(gen_ids, skip_special_tokens=False))

        total += 1
        is_format_ok = ("<no_layout>" in text) or ("<obj>" in text and "<box>" in text)
        if is_format_ok:
            format_ok += 1

        if "<no_layout>" in text:
            no_layout += 1
            parsed_ok += 1
            continue

        objects = parse_layout_output(text)
        if objects:
            parsed_ok += 1
            parsed_obj_ok += 1

    return total, format_ok, parsed_ok, parsed_obj_ok, no_layout


def _list_checkpoints(checkpoint_dir: str, pattern: str, include_final: bool) -> List[str]:
    candidates = []
    if not os.path.isdir(checkpoint_dir):
        return candidates
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(path):
            continue
        if name == "final" and include_final:
            candidates.append(path)
        elif re.match(pattern, name):
            candidates.append(path)
    # sort by step if possible
    def _step(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else 10**9
    candidates.sort(key=_step)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate layout checkpoints")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default=r"checkpoint-\d+")
    parser.add_argument("--include-final", action="store_true")
    args = parser.parse_args()

    prompts = _reservoir_sample_prompts(args.input_jsonl, args.max_samples, args.seed)
    if not prompts:
        raise RuntimeError("No prompts sampled from input JSONL.")

    checkpoints = _list_checkpoints(args.checkpoint_dir, args.pattern, args.include_final)
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint",
            "total",
            "format_ok",
            "format_ok_rate",
            "parsed_ok",
            "parsed_ok_rate",
            "parsed_objects_ok",
            "parsed_objects_ok_rate",
            "no_layout",
            "no_layout_rate",
        ])

        for ckpt in checkpoints:
            print(f"\n=== Evaluating {ckpt} ===")
            model, tokenizer = _load_model_and_tokenizer(args.base_model, ckpt, args.device)
            total, format_ok, parsed_ok, parsed_obj_ok, no_layout = _eval_prompts(model, tokenizer, prompts)

            writer.writerow([
                os.path.basename(ckpt),
                total,
                format_ok,
                f"{format_ok/total:.4f}",
                parsed_ok,
                f"{parsed_ok/total:.4f}",
                parsed_obj_ok,
                f"{parsed_obj_ok/total:.4f}",
                no_layout,
                f"{no_layout/total:.4f}",
            ])
            print(f"Format OK: {format_ok}/{total} ({format_ok/total*100:.1f}%)")
            print(f"Parsed OK: {parsed_ok}/{total} ({parsed_ok/total*100:.1f}%)")
            print(f"No-layout: {no_layout}/{total} ({no_layout/total*100:.1f}%)")

            # free memory
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
