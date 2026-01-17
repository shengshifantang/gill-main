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
    def generate_layout(self, prompt: str, enable_cot: bool = False, feedback: Optional[str] = None, **kwargs) -> dict:
        formatted = format_layout_input(self.tokenizer, prompt, enable_cot=enable_cot, feedback=feedback)
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
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--llm-device", default="cuda:1", help="Device for LLM (e.g., cuda:1)")
    parser.add_argument("--sd-device", default="cuda:0", help="Device for Kolors/SD (e.g., cuda:0)")
    parser.add_argument("--gill-device", default="cpu", help="Device for GILL LM/vision (default: cpu to save GPU)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--enable-cot", action="store_true")
    parser.add_argument("--scheduled-sampling", type=float, default=0.4)
    parser.add_argument("--adapter-scale", type=float, default=1.0, help="Fixed adapter injection scale")
    parser.add_argument("--disable-phrase-emb", action="store_true", help="Disable phrase embeddings for spatial adapter")
    parser.add_argument("--use-gill-prompt", action="store_true",
                        help="Use GILL to generate a semantic prompt before Kolors generation")
    parser.add_argument("--disable-verifier", action="store_true")
    parser.add_argument("--verifier-type", default="hybrid", choices=["hybrid", "grounding_dino", "qwen2vl_7b", "qwen2vl"])
    parser.add_argument("--visual-encoder", default="auto", help="Vision encoder path, 'auto' to prefer local, or 'none'")
    parser.add_argument("--offline", action="store_true", help="Force HF offline mode")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Ensure GILL uses the intended SD device (models.py reads SD_DEVICE env)
    if args.sd_device:
        os.environ["SD_DEVICE"] = args.sd_device

    prompts = _read_prompts(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_jsonl) or ".", exist_ok=True)

    planner = _load_layout_planner(args.base_model, args.layout_adapter, args.llm_device, args.dtype)
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
        verifier = create_feedback_verifier(verifier_type=args.verifier_type, device=args.sd_device, use_grounding=True)

    with open(args.results_jsonl, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
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
                disable_phrase_emb=args.disable_phrase_emb,
                use_gill_prompt=args.use_gill_prompt,
                max_retries=args.max_retries,
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
