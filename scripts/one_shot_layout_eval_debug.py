#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot layout evaluation + diagnostics.

Runs the Layout Planner on a JSONL dataset and reports:
  - format/parse stats
  - GT/pred bbox validity
  - IoU under different name matching modes
  - IoU with swapped xy (sanity check)

This is intended to quickly diagnose low mean_iou causes.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.layout_planner import format_layout_input, parse_layout_output


def _ensure_chat_template(tokenizer, adapter_path: Optional[str], base_model: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    candidates = []
    if adapter_path:
        candidates.append(os.path.join(adapter_path, "chat_template.jinja"))
    if base_model:
        candidates.append(os.path.join(base_model, "chat_template.jinja"))
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                return
            except Exception:
                pass
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )


def load_layout_planner(
    base_model: str,
    adapter_path: Optional[str],
    device: str,
    dtype: str = "bf16",
    max_memory_gb: Optional[float] = None,
    offload_dir: Optional[str] = None,
    lora_reserve_gb: float = 2.0,
    adapter_on_cpu: bool = False,
):
    if adapter_path and os.path.exists(adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _ensure_chat_template(tokenizer, adapter_path, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]})

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    device_map = device
    max_memory = None
    offload_folder = None
    if adapter_on_cpu:
        device_map = "cpu"
        max_memory = None
        offload_folder = None
        torch_dtype = torch.float32
    if max_memory_gb is not None:
        device_map = "auto"
        if torch.cuda.is_available():
            gpu_cap = max(max_memory_gb - lora_reserve_gb, 4)
            max_memory = {i: f"{gpu_cap}GiB" for i in range(torch.cuda.device_count())}
            max_memory["cpu"] = "48GiB"
        offload_folder = offload_dir
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _get_prompt(item: Dict) -> str:
    for key in ("caption", "prompt", "text"):
        val = item.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _names_match(pred_name: str, gt_name: str, mode: str = "contains") -> bool:
    pred_name = (pred_name or "").strip()
    gt_name = (gt_name or "").strip()
    if not gt_name:
        return mode == "any"
    if mode == "any":
        return True
    if mode == "exact":
        return pred_name == gt_name
    return (gt_name in pred_name) or (pred_name in gt_name)


def _normalize_gt_objects(item: Dict, scale_mode: str = "auto") -> List[Dict]:
    objs = item.get("objects", []) or []
    if not isinstance(objs, list):
        return []
    width = float(item.get("width", 0) or 0)
    height = float(item.get("height", 0) or 0)
    has_dim = width > 0 and height > 0

    normed = []
    for obj in objs:
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
        normed.append({"name": obj.get("name", ""), "bbox": b_norm})
    return normed


def _valid_box(b: List[float]) -> bool:
    if not isinstance(b, list) or len(b) != 4:
        return False
    x1, y1, x2, y2 = b
    return (x2 > x1) and (y2 > y1)


def _swap_xy(b: List[float]) -> List[float]:
    x1, y1, x2, y2 = b
    return [y1, x1, y2, x2]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    a_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    b_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _match_iou(pred: List[Dict], gt: List[Dict], name_match: str = "contains") -> Tuple[float, int]:
    if not gt:
        return 0.0, 0
    used = set()
    ious = []
    for g in gt:
        name = str(g.get("name", "")).strip()
        best = 0.0
        best_idx = -1
        for i, p in enumerate(pred):
            if i in used:
                continue
            if not _names_match(str(p.get("name", "")), name, mode=name_match):
                continue
            iou = _bbox_iou(p["bbox"], g["bbox"])
            if iou > best:
                best = iou
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            ious.append(best)
    if not ious:
        return 0.0, 0
    return sum(ious) / len(ious), len(ious)


def _reservoir_sample(path: str, k: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    reservoir: List[Dict] = []
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
                reservoir.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = item
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot layout eval + diagnostics")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--base-model", required=True, help="LLM base model path")
    parser.add_argument("--layout-adapter", required=True, help="Layout Planner LoRA adapter path")
    parser.add_argument("--device", default="cuda", help="Device, e.g., cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Use reservoir sampling")
    parser.add_argument("--gt-field", default="objects", choices=["objects", "annotations"], help="GT source field")
    parser.add_argument("--gt-bbox-scale", default="auto", choices=["auto", "pixel", "1000"])
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--offload-dir", type=str, default="./offload_eval")
    parser.add_argument("--lora-reserve-gb", type=float, default=2.0)
    parser.add_argument("--adapter-on-cpu", action="store_true")
    parser.add_argument("--print-examples", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer = load_layout_planner(
        args.base_model,
        args.layout_adapter,
        args.device,
        dtype=args.dtype,
        max_memory_gb=args.max_memory_gb,
        offload_dir=args.offload_dir,
        lora_reserve_gb=args.lora_reserve_gb,
        adapter_on_cpu=args.adapter_on_cpu,
    )

    if args.shuffle:
        items = _reservoir_sample(args.input_jsonl, args.max_samples, args.seed)
    else:
        items = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if args.max_samples > 0 and len(items) >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if _get_prompt(item):
                    items.append(item)

    total = 0
    format_ok = 0
    gt_obj = 0
    pred_obj = 0
    gt_valid = 0
    pred_valid = 0

    iou_stats = {
        "exact": {"sum": 0.0, "matched": 0},
        "contains": {"sum": 0.0, "matched": 0},
        "any": {"sum": 0.0, "matched": 0},
        "any_swapxy": {"sum": 0.0, "matched": 0},
    }

    bad_examples: List[Dict] = []

    for item in items:
        prompt = _get_prompt(item)
        if not prompt:
            continue

        if args.gt_field == "annotations":
            ann = item.get("annotations", {}) if isinstance(item.get("annotations", {}), dict) else {}
            gt_src = ann.get("objects", []) or []
        else:
            gt_src = item.get("objects", []) or []

        gt_item = {"objects": gt_src, "width": item.get("width", 0), "height": item.get("height", 0)}
        gt_objects = _normalize_gt_objects(gt_item, scale_mode=args.gt_bbox_scale)

        formatted = format_layout_input(tokenizer, prompt, enable_cot=False, feedback=None)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        layout_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            layout_text = layout_text.replace(tok, "").strip()
        pred_objects = parse_layout_output(layout_text)

        total += 1
        if "<no_layout>" in layout_text or ("<obj>" in layout_text and "<box>" in layout_text):
            format_ok += 1

        gt_obj += len(gt_objects)
        pred_obj += len(pred_objects)
        gt_valid += sum(1 for o in gt_objects if _valid_box(o["bbox"]))
        pred_valid += sum(1 for o in pred_objects if _valid_box(o["bbox"]))

        for mode in ("exact", "contains", "any"):
            mean_iou, matched = _match_iou(pred_objects, gt_objects, name_match=mode)
            if matched > 0:
                iou_stats[mode]["sum"] += mean_iou
                iou_stats[mode]["matched"] += 1

        # swap-xy diagnostic (ignore names)
        pred_swap = [{"name": p.get("name", ""), "bbox": _swap_xy(p["bbox"])} for p in pred_objects]
        mean_iou, matched = _match_iou(pred_swap, gt_objects, name_match="any")
        if matched > 0:
            iou_stats["any_swapxy"]["sum"] += mean_iou
            iou_stats["any_swapxy"]["matched"] += 1

        if len(bad_examples) < args.print_examples:
            if gt_objects and pred_objects:
                # capture worst-looking cases
                m_iou, m_cnt = _match_iou(pred_objects, gt_objects, name_match="contains")
                if m_cnt == 0 or m_iou < 0.1:
                    bad_examples.append({
                        "prompt": prompt,
                        "layout_text": layout_text,
                        "pred_objects": pred_objects,
                        "gt_objects": gt_objects,
                        "mean_iou": m_iou if m_cnt > 0 else None,
                        "matched": m_cnt,
                    })

    def _mean_iou(stat):
        return stat["sum"] / stat["matched"] if stat["matched"] > 0 else 0.0

    print("==== Summary ====")
    print(json.dumps({
        "total": total,
        "format_acc": float(format_ok) / total if total else 0.0,
        "gt_objects": gt_obj,
        "pred_objects": pred_obj,
        "gt_valid_ratio": float(gt_valid) / gt_obj if gt_obj else 0.0,
        "pred_valid_ratio": float(pred_valid) / pred_obj if pred_obj else 0.0,
        "mean_iou_exact": _mean_iou(iou_stats["exact"]),
        "mean_iou_contains": _mean_iou(iou_stats["contains"]),
        "mean_iou_any": _mean_iou(iou_stats["any"]),
        "mean_iou_any_swapxy": _mean_iou(iou_stats["any_swapxy"]),
        "matched_samples_exact": iou_stats["exact"]["matched"],
        "matched_samples_contains": iou_stats["contains"]["matched"],
        "matched_samples_any": iou_stats["any"]["matched"],
    }, ensure_ascii=False, indent=2))

    if bad_examples:
        print("\n==== Examples (low IoU) ====")
        for ex in bad_examples:
            print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
