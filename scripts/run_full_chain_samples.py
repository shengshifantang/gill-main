#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full-chain inference (layout planner + spatial adapter + optional verifier)
on multiple prompts and save images + JSONL results.

Designed for offline environments and multi-GPU use by splitting LLM and SD.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure project root is on sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.layout_planner import format_layout_input, parse_layout_output
from gill.models import GILL, GILLArgs
from gill.spatial_adapter import load_spatial_adapter_state_dict, create_spatial_adapter_for_kolors
from gill.feedback_verifier import create_feedback_verifier


def _smart_expand_objects(objects: List[dict]) -> List[dict]:
    """Expand tiny boxes for hard-to-detect categories (inference-time heuristic)."""
    if not objects:
        return objects

    def _expand_bbox(bbox: List[float], name: str) -> List[float]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        target_w, target_h = 0.05, 0.05  # default minimum size
        if any(k in name for k in ["人", "男", "女", "童", "客"]):
            target_w, target_h = 0.08, 0.15
        elif any(k in name for k in ["灯", "球", "盘", "杯", "瓶", "鸟"]):
            target_w, target_h = 0.06, 0.06

        if w < target_w:
            half = target_w / 2.0
            x1 = max(0.0, cx - half)
            x2 = min(1.0, cx + half)
        if h < target_h:
            half = target_h / 2.0
            y1 = max(0.0, cy - half)
            y2 = min(1.0, cy + half)
        return [x1, y1, x2, y2]

    for obj in objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        name = str(obj.get("name", ""))
        obj["bbox"] = _expand_bbox([float(v) for v in bbox], name)
    return objects


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
    # Fallback ChatML template
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
        strict_entities: bool = False,
        **kwargs,
    ) -> dict:
        formatted = format_layout_input(
            self.tokenizer,
            prompt,
            enable_cot=enable_cot,
            feedback=feedback,
            strict_entities=strict_entities,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        layout_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            layout_text = layout_text.replace(token, "").strip()
        objects = parse_layout_output(layout_text)
        if getattr(self, "smart_expand", False):
            objects = _smart_expand_objects(objects)
        return {"layout_text": layout_text, "objects": objects}


def _load_layout_planner(base_model: str, adapter_path: Optional[str], device: str, dtype: str) -> SimpleLayoutPlanner:
    if adapter_path and os.path.exists(adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _ensure_chat_template(tokenizer, adapter_path, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]})

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return SimpleLayoutPlanner(model, tokenizer)


def _load_spatial_adapter(path: Optional[str], device: str):
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        return load_spatial_adapter_state_dict(state_dict, device=device, dtype=torch.float32)
    return create_spatial_adapter_for_kolors()


def _slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    text = re.sub(r"[-\s]+", "_", text)
    return text[:max_len] or "sample"


def _read_prompts(args) -> List[str]:
    prompts = []
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if args.prompts:
        prompts.extend(args.prompts)
    if not prompts:
        prompts = [
            "一只猫在桌子左边",
            "右边有一辆车，左边有一棵树",
            "上方是蓝天，下方是草地",
        ]
    return prompts[: args.max_samples]


def _objects_to_layout_text(objects: List[dict]) -> str:
    if not objects:
        return "<no_layout>"
    chunks = []
    for obj in objects:
        name = str(obj.get("name", "")).strip()
        bbox = obj.get("bbox", [])
        if not name or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        chunks.append(f"<obj>{name}</obj><box>[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]</box>")
    return "".join(chunks) if chunks else "<no_layout>"


def _load_layout_jsonl(path: str) -> List[dict]:
    items = []
    if not path or not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            prompt = (item.get("prompt") or item.get("caption") or "").strip()
            if not prompt:
                continue
            layout = item.get("layout") or {}
            layout_text = item.get("layout_text") or layout.get("layout_text")
            objects = item.get("objects") or layout.get("objects") or []
            if not objects and layout_text:
                objects = parse_layout_output(layout_text)
            if not layout_text:
                layout_text = _objects_to_layout_text(objects)
            items.append({"prompt": prompt, "objects": objects, "layout_text": layout_text})
    return items


class FixedLayoutPlanner:
    def __init__(self, tokenizer, items: List[dict]):
        self.tokenizer = tokenizer
        self._items = items or []
        self._index = 0
        self._by_prompt = {}
        for it in self._items:
            self._by_prompt.setdefault(it["prompt"], []).append(it)

    @torch.no_grad()
    def generate_layout(self, prompt: str, **kwargs) -> dict:
        item = None
        if self._index < len(self._items):
            candidate = self._items[self._index]
            if candidate.get("prompt") == prompt:
                item = candidate
                self._index += 1
        if item is None:
            bucket = self._by_prompt.get(prompt)
            if bucket:
                item = bucket.pop(0)
        if item is None:
            return {"layout_text": "<no_layout>", "objects": []}
        objects = item.get("objects") or []
        layout_text = item.get("layout_text") or _objects_to_layout_text(objects)
        if getattr(self, "smart_expand", False):
            objects = _smart_expand_objects([dict(obj) for obj in objects])
        return {"layout_text": layout_text, "objects": objects}


def _resolve_visual_encoder(value: str) -> str:
    if value is None:
        return "none"
    v = value.strip().lower()
    if v in {"auto", "local"}:
        candidates = [
            os.path.join("model", "chinese-clip-vit-base-patch16"),
            os.path.join("model", "clip-vit-large-patch14"),
            os.path.join("model", "openai-clip-vit-large-patch14"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
        return "none"
    return value


def main():
    parser = argparse.ArgumentParser(description="Full-chain multi-sample inference")
    parser.add_argument("--base-model", required=True, help="LLM base model path")
    parser.add_argument("--layout-adapter", required=True, help="Layout Planner LoRA adapter path")
    parser.add_argument("--spatial-adapter", required=True, help="Spatial Adapter checkpoint (.pt)")
    parser.add_argument("--kolors-model", default="./model/Kolors", help="Kolors model path")
    parser.add_argument("--output-dir", default="outputs/full_chain_samples", help="Output directory")
    parser.add_argument("--results-jsonl", default="outputs/full_chain_samples/results.jsonl")
    parser.add_argument("--prompts-file", default=None, help="Text file with one prompt per line")
    parser.add_argument("--prompts", nargs="*", default=None, help="Prompts passed via CLI")
    parser.add_argument("--layout-jsonl", default=None, help="JSONL with prompt + objects/layout_text (bypass planner)")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for reproducible generation")
    parser.add_argument("--llm-device", default="cuda:1", help="Device for LLM (e.g., cuda:1)")
    parser.add_argument("--sd-device", default="cuda:0", help="Device for Kolors/SD (e.g., cuda:0)")
    parser.add_argument("--gill-device", default="cpu", help="Device for GILL LM/vision (default: cpu to save GPU)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--enable-cot", action="store_true")
    parser.add_argument("--strict-entities", action="store_true", help="Force layout to cover all entities in prompt")
    parser.add_argument("--strict-cot", action="store_true",
                        help="Enable CoT and strict entity coverage for layout planner")
    parser.add_argument("--scheduled-sampling", type=float, default=0.4)
    parser.add_argument("--adapter-scale", type=float, default=1.0, help="Fixed adapter injection scale")
    parser.add_argument("--force-gate", action="store_true", help="Force adapter gate value (override learned gate)")
    parser.add_argument("--gate-value", type=float, default=1.0, help="Gate value when --force-gate is set")
    parser.add_argument("--disable-phrase-emb", action="store_true", help="Disable phrase embeddings for spatial adapter")
    parser.add_argument("--smart-expand", action="store_true", help="Expand tiny boxes for hard-to-detect objects")
    parser.add_argument("--use-gill-prompt", action="store_true",
                        help="Use GILL to generate a semantic prompt before Kolors generation")
    parser.add_argument("--disable-verifier", action="store_true")
    parser.add_argument("--verifier-type", default="hybrid", choices=["hybrid", "grounding_dino", "qwen2vl_7b", "qwen2vl", "manager_surveyor"])
    parser.add_argument("--vlm-model", default=None, help="VLM model path/name for verifier (Qwen2/3-VL)")
    parser.add_argument("--dino-backend", choices=["auto", "hf", "official"], default="auto",
                        help="GroundingDINO backend: hf or official (Swin-B .pth+.py)")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base",
                        help="HF GroundingDINO model id or local path")
    parser.add_argument("--dino-config", default=None, help="Official GroundingDINO config .py path")
    parser.add_argument("--dino-ckpt", default=None, help="Official GroundingDINO checkpoint .pth path")
    parser.add_argument("--dino-box-thr", type=float, default=0.35, help="DINO box threshold")
    parser.add_argument("--dino-text-thr", type=float, default=0.25, help="DINO text threshold")
    parser.add_argument("--visual-encoder", default="auto", help="Vision encoder path, 'auto' to prefer local, or 'none'")
    parser.add_argument("--offline", action="store_true", help="Force HF offline mode")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Ensure GILL uses the intended SD device (models.py reads SD_DEVICE env)
    if args.sd_device:
        os.environ["SD_DEVICE"] = args.sd_device

    if args.strict_cot:
        args.enable_cot = True
        args.strict_entities = True

    if args.seed is not None:
        try:
            import random
            random.seed(args.seed)
        except Exception:
            pass
        torch.manual_seed(args.seed)

    layout_items = _load_layout_jsonl(args.layout_jsonl) if args.layout_jsonl else []
    if layout_items:
        layout_items = layout_items[: args.max_samples]
    if layout_items:
        prompts = [it["prompt"] for it in layout_items][: args.max_samples]
    else:
        prompts = _read_prompts(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_jsonl) or ".", exist_ok=True)

    if layout_items:
        if args.layout_adapter and os.path.exists(args.layout_adapter):
            tokenizer = AutoTokenizer.from_pretrained(args.layout_adapter, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        _ensure_chat_template(tokenizer, args.layout_adapter, args.base_model)
        planner = FixedLayoutPlanner(tokenizer, layout_items)
    else:
        planner = _load_layout_planner(args.base_model, args.layout_adapter, args.llm_device, args.dtype)
    planner.smart_expand = bool(args.smart_expand)
    spatial_adapter = _load_spatial_adapter(args.spatial_adapter, args.sd_device)

    model_args = GILLArgs()
    model_args.visual_encoder = _resolve_visual_encoder(args.visual_encoder)
    # Prefer local base model path to avoid any hub access in offline mode
    if args.base_model and os.path.exists(args.base_model):
        model_args.opt_version = args.base_model
    # Load GILL LM on llm_device to avoid consuming SD GPU memory
    gill = GILL(tokenizer=planner.tokenizer, model_args=model_args, load_sd=True, device_map=args.gill_device)
    if hasattr(gill, "sd_pipe") and gill.sd_pipe is not None:
        gill.sd_pipe.safety_checker = None
        gill.sd_pipe.feature_extractor = None
        gill.sd_pipe.requires_safety_checker = False

    verifier = None
    if not args.disable_verifier:
        verifier = create_feedback_verifier(
            verifier_type=args.verifier_type,
            vlm_model_name=args.vlm_model,
            device=args.sd_device,
            use_grounding=True,
            dino_model_id=args.dino_model,
            dino_backend=args.dino_backend,
            dino_config_path=args.dino_config,
            dino_checkpoint_path=args.dino_ckpt,
            dino_box_threshold=args.dino_box_thr,
            dino_text_threshold=args.dino_text_thr,
        )

    def _make_generator(sample_idx: int):
        if args.seed is None:
            return None
        device = args.sd_device or "cpu"
        try:
            gen = torch.Generator(device=device)
        except Exception:
            gen = torch.Generator()
        gen.manual_seed(int(args.seed) + int(sample_idx))
        return gen

    with open(args.results_jsonl, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            gen = _make_generator(i)
            result = gill.generate_with_layout(
                prompt=prompt,
                enable_layout=True,
                enable_feedback=not args.disable_verifier,
                layout_planner=planner,
                spatial_adapter=spatial_adapter,
                feedback_verifier=verifier,
                num_inference_steps=30,
                scheduled_sampling_ratio=args.scheduled_sampling,
                adapter_scale=args.adapter_scale,
                force_gate=args.force_gate,
                gate_value=args.gate_value,
                disable_phrase_emb=args.disable_phrase_emb,
                use_gill_prompt=args.use_gill_prompt,
                max_retries=args.max_retries,
                strict_entities=args.strict_entities,
                enable_cot=args.enable_cot,
                generator=gen,
            )

            image = result.get("image")
            slug = _slugify(prompt)
            img_path = os.path.join(args.output_dir, f"{i:02d}_{slug}.png")
            if image is not None:
                image.save(img_path)

            record = {
                "prompt": prompt,
                "status": result.get("status"),
                "layout": result.get("layout"),
                "semantic_prompt": result.get("semantic_prompt"),
                "image_path": img_path if image is not None else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] {prompt} -> {record['status']}")


if __name__ == "__main__":
    main()
