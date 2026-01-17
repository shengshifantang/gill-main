#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-loop evaluation template.

This script evaluates:
  - Layout format accuracy
  - Optional IoU (if GT boxes are provided in JSONL)
  - Open-loop vs Closed-loop success rate (using verifier)

Expected JSONL fields (per line):
  - "caption" or "prompt" or "text"
  - Optional: "objects": [{"name": "...", "bbox": [x1,y1,x2,y2]}]

Notes:
  - If --no-image is set, the script only evaluates layout outputs.
  - Closed-loop uses GILL.generate_with_layout with feedback_verifier.
"""

import argparse
import sys
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.layout_planner import format_layout_input, parse_layout_output
from gill.spatial_adapter import create_spatial_adapter_for_kolors, load_spatial_adapter_state_dict


class SimpleLayoutPlanner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def generate_layout(
        self,
        prompt: str,
        enable_cot: bool = False,
        feedback: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> Dict:
        formatted = format_layout_input(self.tokenizer, prompt, enable_cot=enable_cot, feedback=feedback)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        layout_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            layout_text = layout_text.replace(token, "").strip()
        objects = parse_layout_output(layout_text)
        return {"layout_text": layout_text, "objects": objects}


def load_layout_planner(
    base_model: str,
    adapter_path: Optional[str],
    device: str,
    max_memory_gb: Optional[float] = None,
    offload_dir: Optional[str] = None,
    dtype: str = "bf16",
    lora_reserve_gb: float = 2.0,
    adapter_on_cpu: bool = False,
) -> SimpleLayoutPlanner:
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
        # force CPU to avoid GPU OOM when loading LoRA
        device_map = "cpu"
        max_memory = None
        offload_folder = None
        torch_dtype = torch.float32
    elif max_memory_gb is not None:
        device_map = "auto"
        if torch.cuda.is_available():
            # leave some headroom for LoRA weights
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Try to pass device_map/max_memory for safer loading; fallback if unsupported.
        try:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map=device_map if max_memory_gb is not None else None,
                max_memory=max_memory if max_memory_gb is not None else None,
                offload_folder=offload_folder if max_memory_gb is not None else None,
            )
        except TypeError:
            model = PeftModel.from_pretrained(model, adapter_path)
    # If offload created meta tensors (old PEFT), fallback to CPU to avoid runtime crash.
    if _has_meta_params(model) and not adapter_on_cpu:
        print("⚠️ Detected meta parameters after LoRA load; falling back to CPU mode for evaluation.")
        return load_layout_planner(
            base_model=base_model,
            adapter_path=adapter_path,
            device="cpu",
            max_memory_gb=None,
            offload_dir=None,
            dtype="fp16",
            lora_reserve_gb=lora_reserve_gb,
            adapter_on_cpu=True,
        )

    model.eval()
    return SimpleLayoutPlanner(model, tokenizer)


def _ensure_chat_template(tokenizer, adapter_path: Optional[str], base_model: str) -> None:
    """Ensure tokenizer.chat_template exists for apply_chat_template()."""
    if getattr(tokenizer, "chat_template", None):
        return
    # Try load chat_template.jinja from adapter or base model directory
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
    # Fallback: basic ChatML-style template
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


def _has_meta_params(model) -> bool:
    try:
        for p in model.parameters():
            if getattr(p, "is_meta", False):
                return True
    except Exception:
        return False
    return False


def load_spatial_adapter(adapter_path: Optional[str], device: str):
    if adapter_path and os.path.exists(adapter_path):
        ckpt = torch.load(adapter_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        return load_spatial_adapter_state_dict(state_dict, device=device, dtype=torch.float32)
    return create_spatial_adapter_for_kolors()


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


def _names_match(pred_name: str, gt_name: str, mode: str = "contains") -> bool:
    pred_name = (pred_name or "").strip()
    gt_name = (gt_name or "").strip()
    if not gt_name:
        return mode == "any"
    if mode == "any":
        return True
    if mode == "exact":
        return pred_name == gt_name
    # contains (default): allow partial match in either direction
    return (gt_name in pred_name) or (pred_name in gt_name)


def _normalize_gt_objects(item: Dict, scale_mode: str = "auto") -> List[Dict]:
    """
    Normalize GT bboxes to [0,1] x1,y1,x2,y2.
    scale_mode:
      - "auto": try to infer pixel vs 0-1000 scale
      - "pixel": force divide by width/height
      - "1000": force divide by 1000
    Uses bbox_1000 if present; otherwise bbox.
    Assumes bbox format is already [x1,y1,x2,y2] (consistent with training).
    """
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
                # auto: prefer explicit bbox_1000; otherwise infer scale
                if isinstance(bbox_1000, list) and len(bbox_1000) == 4:
                    b_norm = [v / 1000.0 for v in b]
                elif max_val <= 1000:
                    if has_dim and max(width, height) > 0:
                        ratio = max_val / max(width, height)
                        if ratio > 1.2:
                            # bbox likely in 0-1000 scale
                            b_norm = [v / 1000.0 for v in b]
                        else:
                            b_norm = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
                    else:
                        b_norm = [v / 1000.0 for v in b]
                elif has_dim:
                    b_norm = [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
                else:
                    # cannot normalize reliably
                    continue

        # clamp to [0,1]
        b_norm = [max(0.0, min(1.0, v)) for v in b_norm]
        normed.append({"name": obj.get("name", ""), "bbox": b_norm})
    return normed


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


def _get_prompt(item: Dict) -> str:
    for key in ("caption", "prompt", "text"):
        val = item.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop evaluation template")
    parser.add_argument("--input-jsonl", required=True, help="Input prompts JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output results JSONL")
    parser.add_argument("--base-model", required=True, help="LLM base model path")
    parser.add_argument("--layout-adapter", default=None, help="Layout Planner LoRA adapter path")
    parser.add_argument("--kolors-model", default="./model/Kolors", help="Kolors model path")
    parser.add_argument("--spatial-adapter", default=None, help="Spatial Adapter checkpoint path")
    parser.add_argument("--verifier-type", default="hybrid", choices=["hybrid", "grounding_dino", "qwen2vl_7b", "qwen2vl"])
    parser.add_argument("--device", default="cuda", help="Device, e.g., cuda:0")
    parser.add_argument("--max-memory-gb", type=float, default=None, help="Per-GPU max memory for device_map=auto")
    parser.add_argument("--offload-dir", type=str, default="./offload_eval", help="Offload dir when max_memory_gb is set")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Model dtype")
    parser.add_argument("--lora-reserve-gb", type=float, default=2.0, help="Reserve GPU memory for LoRA weights")
    parser.add_argument("--adapter-on-cpu", action="store_true", help="Force loading LoRA on CPU to avoid GPU OOM")
    parser.add_argument(
        "--gt-bbox-scale",
        type=str,
        default="auto",
        choices=["auto", "pixel", "1000"],
        help="GT bbox scale for IoU: auto|pixel|1000",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--require-gt-objects", action="store_true", help="Skip samples with empty GT objects")
    parser.add_argument("--name-match", type=str, default="contains", choices=["contains", "exact", "any"], help="Name matching strategy for IoU")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-image", action="store_true", help="Skip image generation/verification")
    parser.add_argument("--enable-cot", action="store_true")
    parser.add_argument("--scheduled-sampling", type=float, default=0.4)
    args = parser.parse_args()

    planner = load_layout_planner(
        args.base_model,
        args.layout_adapter,
        args.device,
        max_memory_gb=args.max_memory_gb,
        offload_dir=args.offload_dir,
        dtype=args.dtype,
        lora_reserve_gb=args.lora_reserve_gb,
        adapter_on_cpu=args.adapter_on_cpu,
    )

    gill_model = None
    spatial_adapter = None
    verifier = None
    if not args.no_image:
        # Lazy import to avoid requiring diffusers in layout-only evaluation.
        from gill.models import GILL, GILLArgs
        from gill.feedback_verifier import create_feedback_verifier

        gill_model = GILL(tokenizer=None, model_args=GILLArgs(), load_sd=True, device_map=args.device)
        spatial_adapter = load_spatial_adapter(args.spatial_adapter, args.device)
        verifier = create_feedback_verifier(verifier_type=args.verifier_type, device=args.device, use_grounding=True)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    total = 0
    format_ok = 0
    iou_sum = 0.0
    iou_count = 0
    open_success = 0
    closed_success = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if args.max_samples > 0 and total >= args.max_samples:
                break
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

            gt_objects = _normalize_gt_objects(item, scale_mode=args.gt_bbox_scale)
            if args.require_gt_objects and not gt_objects:
                continue
            total += 1

            # Open-loop (layout only)
            layout_result = planner.generate_layout(prompt, enable_cot=args.enable_cot)
            layout_text = layout_result.get("layout_text", "")
            pred_objects = layout_result.get("objects", [])

            ok = ("<obj>" in layout_text and "<box>" in layout_text) or ("<no_layout>" in layout_text)
            if ok:
                format_ok += 1

            mean_iou, matched = _match_iou(pred_objects, gt_objects, name_match=args.name_match) if gt_objects else (0.0, 0)
            if matched > 0:
                iou_sum += mean_iou
                iou_count += 1

            record = {
                "prompt": prompt,
                "layout_text": layout_text,
                "pred_objects": pred_objects,
                "gt_objects": gt_objects,
                "format_ok": ok,
                "mean_iou": mean_iou if matched > 0 else None,
            }

            # Open-loop / Closed-loop generation (optional)
            if not args.no_image and gill_model is not None:
                open_res = gill_model.generate_with_layout(
                    prompt=prompt,
                    enable_layout=True,
                    enable_feedback=False,
                    layout_planner=planner,
                    spatial_adapter=spatial_adapter,
                    feedback_verifier=None,
                    num_inference_steps=30,
                    scheduled_sampling_ratio=args.scheduled_sampling,
                    max_retries=0,
                )
                record["open_status"] = open_res.get("status")
                if record["open_status"] == "success":
                    open_success += 1

                closed_res = gill_model.generate_with_layout(
                    prompt=prompt,
                    enable_layout=True,
                    enable_feedback=True,
                    layout_planner=planner,
                    spatial_adapter=spatial_adapter,
                    feedback_verifier=verifier,
                    num_inference_steps=30,
                    scheduled_sampling_ratio=args.scheduled_sampling,
                    max_retries=args.max_retries,
                )
                record["closed_status"] = closed_res.get("status")
                if record["closed_status"] == "success":
                    closed_success += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total": total,
        "format_acc": float(format_ok) / total if total else 0.0,
        "mean_iou": float(iou_sum) / iou_count if iou_count else 0.0,
    }
    if not args.no_image:
        summary.update({
            "open_loop_success": float(open_success) / total if total else 0.0,
            "closed_loop_success": float(closed_success) / total if total else 0.0,
        })

    print("==== Summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
