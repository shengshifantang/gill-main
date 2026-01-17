#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate COCO captions to Chinese using Qwen2.5-7B-Instruct.

Input: COCO captions JSON (annotations with image_id, caption)
Output: JSONL with fields:
  {"ann_id": 123, "image_id": 456, "caption": "...", "caption_zh": "..."}
"""

import argparse
import json
import os
import re
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYS_PROMPT = "你是一个翻译助手。请把英文图像描述翻译成简洁自然的中文，不要添加多余内容，只输出译文。"


def _build_prompt(tokenizer, text: str) -> str:
    msg = f"英文描述：{text}\n中文译文："
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": msg},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return SYS_PROMPT + "\n" + msg


def _clean_zh(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^(中文译文|译文|翻译)\s*[:：]\s*", "", t)
    t = t.strip().strip("\"'“”")
    if "\n" in t:
        t = t.split("\n")[0].strip()
    return t


def load_coco_captions(path: str) -> List[Dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    anns = data.get("annotations", []) if isinstance(data, dict) else []
    out = []
    for ann in anns:
        cap = ann.get("caption", "")
        if not cap:
            continue
        out.append({
            "ann_id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "caption": str(cap),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-en", required=True, help="COCO captions json")
    parser.add_argument("--output", required=True, help="Output jsonl")
    parser.add_argument("--qwen-model", required=True, help="Qwen2.5-7B-Instruct path")
    parser.add_argument("--device", default="cuda", help="cuda/cuda:0/cpu/auto")
    parser.add_argument("--local-files-only", action="store_true", help="Force local_files_only for HF loading")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Skip captions already translated in output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("ann_id") is not None:
                    done_ids.add(int(item["ann_id"]))

    print(f"[info] loading captions from {args.captions_en}")
    items = load_coco_captions(args.captions_en)
    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]
    if done_ids:
        items = [x for x in items if x.get("ann_id") not in done_ids]
        print(f"[info] resume enabled, remaining {len(items)} captions")

    local_only = args.local_files_only or os.path.isdir(args.qwen_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.qwen_model,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        # Decoder-only models require left padding for correct generation.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model,
            torch_dtype="auto",
            device_map="auto" if args.device == "auto" else None,
            trust_remote_code=True,
            local_files_only=local_only,
        )
    except Exception as e:
        print(f"[error] Failed to load Qwen model from: {args.qwen_model}")
        print(f"[error] local_files_only={local_only} (set --local-files-only if offline)")
        raise
    if args.device != "auto":
        model.to(args.device)
    model.eval()

    total = 0
    written = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            prompts = [_build_prompt(tokenizer, x["caption"]) for x in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            if args.device != "auto":
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for item, zh in zip(batch, decoded):
                zh = _clean_zh(zh)
                rec = {
                    "ann_id": item.get("ann_id"),
                    "image_id": item.get("image_id"),
                    "caption": item.get("caption"),
                    "caption_zh": zh,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            total += len(batch)
            if args.save_every and total % args.save_every == 0:
                f.flush()
                print(f"[info] translated {total}/{len(items)}")

    print(f"[ok] translated {written} captions -> {args.output}")


if __name__ == "__main__":
    main()
