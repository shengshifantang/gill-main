#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation-based evaluation for Layout Planner.

This script builds relation prompts from GT boxes and evaluates whether
the predicted layouts satisfy spatial relations (left/right/above/below).

Why: IoU on caption-only COCO-CN is unfair. Relation accuracy is a better
proxy for layout reasoning when prompts contain explicit spatial cues.
"""

import argparse
import json
import os
import random
import sys
import re
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


_NAME_NORMALIZE_MAP = {
    # People
    "人物": "人",
    "人群": "人",
    "行人": "人",
    "男人": "人",
    "女人": "人",
    "男孩": "人",
    "女孩": "人",
    "孩子": "人",
    "小孩": "人",
    "小伙子": "人",
    "老人": "人",
    "青年": "人",
    "少年": "人",
    "儿童": "人",
    "行走的人": "人",
    # Animals
    "小鸟": "鸟",
    "飞鸟": "鸟",
    "鸟类": "鸟",
    "小狗": "狗",
    "犬": "狗",
    "小猫": "猫",
    "猫咪": "猫",
    # Vehicles
    "车辆": "汽车",
    "小汽车": "汽车",
    "轿车": "汽车",
    "公交车": "公交车",
    "巴士": "公交车",
    "货车": "卡车",
    "摩托": "摩托车",
    "单车": "自行车",
    # Common scene objects
    "楼房": "建筑物",
    "房屋": "建筑物",
    "房子": "建筑物",
    "大楼": "建筑物",
    "公路": "道路",
    "马路": "道路",
    "街道": "道路",
    "草坪": "草地",
    "树林": "树",
    "树木": "树",
    "天空": "天空",
    "蓝天": "天空",
}

_NAME_STRIP_PREFIXES = (
    "一只", "一条", "一辆", "一个", "一位", "一名", "一群", "几只", "几条",
    "几辆", "几个人", "几名", "几位", "两只", "两条", "两辆", "两个", "两个",
    "三个", "四个", "五个", "六个", "七个", "八个", "九个", "十个",
)

_NAME_STRIP_SUFFIXES = (
    "左边", "右边", "左侧", "右侧", "上方", "下方", "前景", "背景", "中间",
)


def _normalize_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    # remove parentheses content
    n = re.sub(r"[（(].*?[）)]", "", n).strip()
    for p in _NAME_STRIP_PREFIXES:
        if n.startswith(p) and len(n) > len(p):
            n = n[len(p):].strip()
            break
    for s in _NAME_STRIP_SUFFIXES:
        if n.endswith(s) and len(n) > len(s):
            n = n[: -len(s)].strip()
            break
    # remove color prefixes (common adjectives)
    for c in ("红色", "蓝色", "白色", "黑色", "黄色", "绿色", "粉色", "金色", "银色", "棕色", "灰色", "橙色", "紫色"):
        if n.startswith(c) and len(n) > len(c):
            n = n[len(c):].strip()
            break
    return _NAME_NORMALIZE_MAP.get(n, n)


def _names_match(pred_name: str, gt_name: str, mode: str = "contains", normalize: bool = False) -> bool:
    pred_name = _normalize_name(pred_name) if normalize else (pred_name or "").strip()
    gt_name = _normalize_name(gt_name) if normalize else (gt_name or "").strip()
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
    for idx, obj in enumerate(objs):
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
        normed.append({
            "idx": idx,
            "name": obj.get("name", ""),
            "bbox": b_norm,
        })
    return normed


def _center(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _relation_from_gt(a: Dict, b: Dict, min_sep: float = 0.1) -> Optional[str]:
    ax, ay = _center(a["bbox"])
    bx, by = _center(b["bbox"])
    dx = ax - bx
    dy = ay - by
    if abs(dx) >= abs(dy) and abs(dx) >= min_sep:
        return "left" if dx < 0 else "right"
    if abs(dy) >= min_sep:
        return "above" if dy < 0 else "below"
    return None


def _relation_ok(rel: str, a: List[float], b: List[float], margin: float = 0.02) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    if rel == "left":
        return ax + margin < bx
    if rel == "right":
        return ax > bx + margin
    if rel == "above":
        return ay + margin < by
    if rel == "below":
        return ay > by + margin
    return False


def _select_pred_by_iou(
    preds: List[Dict],
    gt: Dict,
    name_mode: str = "contains",
    name_normalize: bool = False
) -> Optional[Dict]:
    best = None
    best_iou = -1.0
    gx1, gy1, gx2, gy2 = gt["bbox"]
    for p in preds:
        if not _names_match(
            str(p.get("name", "")),
            str(gt.get("name", "")),
            mode=name_mode,
            normalize=name_normalize
        ):
            continue
        px1, py1, px2, py2 = p["bbox"]
        ix1, iy1 = max(px1, gx1), max(py1, gy1)
        ix2, iy2 = min(px2, gx2), min(py2, gy2)
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        p_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
        g_area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)
        union = p_area + g_area - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best = p
    return best


def build_relation_samples(
    input_jsonl: str,
    max_samples: int,
    max_per_image: int,
    min_sep: float,
    seed: int,
    gt_field: str,
    gt_scale: str,
    no_same_class: bool = False,
) -> List[Dict]:
    rng = random.Random(seed)
    samples: List[Dict] = []

    def _iter_items():
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    for item in _iter_items():
        if max_samples > 0 and len(samples) >= max_samples:
            break
        ann = item.get("annotations", {}) if gt_field == "annotations" else {}
        objs = ann.get("objects", []) if gt_field == "annotations" else item.get("objects", [])
        gt_item = {
            "objects": objs,
            "width": item.get("width", 0),
            "height": item.get("height", 0),
        }
        gt_objects = _normalize_gt_objects(gt_item, scale_mode=gt_scale)
        if len(gt_objects) < 2:
            continue

        pairs = []
        for i in range(len(gt_objects)):
            for j in range(len(gt_objects)):
                if i == j:
                    continue
                rel = _relation_from_gt(gt_objects[i], gt_objects[j], min_sep=min_sep)
                if rel is None:
                    continue
                pairs.append((gt_objects[i], gt_objects[j], rel))

        if not pairs:
            continue
        rng.shuffle(pairs)
        pairs = pairs[:max_per_image]
        for a, b, rel in pairs:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            if no_same_class and str(a.get("name", "")) == str(b.get("name", "")):
                continue
            prompt = f"{a['name']}在{b['name']}{'左边' if rel=='left' else '右边' if rel=='right' else '上方' if rel=='above' else '下方'}。"
            samples.append({
                "prompt": prompt,
                "relation": rel,
                "gt_a": a,
                "gt_b": b,
            })

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Relation evaluation for Layout Planner")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--base-model", required=True, help="LLM base model path")
    parser.add_argument("--layout-adapter", required=True, help="Layout Planner LoRA adapter path")
    parser.add_argument("--output-jsonl", default=None, help="Optional output JSONL with per-sample results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-per-image", type=int, default=2)
    parser.add_argument("--min-sep", type=float, default=0.1, help="Minimum normalized center distance for relation")
    parser.add_argument("--margin", type=float, default=0.02, help="Margin for relation check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-field", default="objects", choices=["objects", "annotations"])
    parser.add_argument("--gt-bbox-scale", default="auto", choices=["auto", "pixel", "1000"])
    parser.add_argument("--name-match", default="contains", choices=["contains", "exact", "any"])
    parser.add_argument("--name-normalize", action="store_true", help="Normalize names before matching")
    parser.add_argument("--no-same-class", action="store_true", help="Drop pairs with identical class names")
    parser.add_argument("--enable-cot", action="store_true", help="Enable chain-of-thought layout planning")
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--offload-dir", type=str, default="./offload_eval")
    parser.add_argument("--lora-reserve-gb", type=float, default=2.0)
    parser.add_argument("--adapter-on-cpu", action="store_true")
    args = parser.parse_args()

    samples = build_relation_samples(
        args.input_jsonl,
        max_samples=args.max_samples,
        max_per_image=args.max_per_image,
        min_sep=args.min_sep,
        seed=args.seed,
        gt_field=args.gt_field,
        gt_scale=args.gt_bbox_scale,
        no_same_class=args.no_same_class,
    )

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

    total = 0
    format_ok = 0
    relation_ok = 0
    matched = 0
    results = []

    for s in samples:
        prompt = s["prompt"]
        gt_a = s["gt_a"]
        gt_b = s["gt_b"]
        rel = s["relation"]

        formatted = format_layout_input(tokenizer, prompt, enable_cot=args.enable_cot, feedback=None)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        layout_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            layout_text = layout_text.replace(tok, "").strip()
        pred_objects = parse_layout_output(layout_text)

        total += 1
        if "<no_layout>" in layout_text or ("<obj>" in layout_text and "<box>" in layout_text):
            format_ok += 1

        pred_a = _select_pred_by_iou(
            pred_objects,
            gt_a,
            name_mode=args.name_match,
            name_normalize=args.name_normalize
        )
        pred_b = _select_pred_by_iou(
            pred_objects,
            gt_b,
            name_mode=args.name_match,
            name_normalize=args.name_normalize
        )
        ok = False
        if pred_a and pred_b:
            matched += 1
            ok = _relation_ok(rel, pred_a["bbox"], pred_b["bbox"], margin=args.margin)
            if ok:
                relation_ok += 1

        if args.output_jsonl:
            results.append({
                "prompt": prompt,
                "relation": rel,
                "gt_a": gt_a,
                "gt_b": gt_b,
                "pred_a": pred_a,
                "pred_b": pred_b,
                "relation_ok": ok,
                "layout_text": layout_text,
            })

    summary = {
        "total": total,
        "format_acc": float(format_ok) / total if total else 0.0,
        "matched_ratio": float(matched) / total if total else 0.0,
        "relation_acc": float(relation_ok) / matched if matched else 0.0,
        "samples": len(samples),
    }
    print("==== Summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
