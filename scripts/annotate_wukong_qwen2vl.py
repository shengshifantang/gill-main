#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小可用的本地标注脚本（Qwen2-VL-7B-Instruct）
功能：
 1. 读取未标注的悟空 JSONL（每行至少包含 image_path, caption）
 2. 过滤坏图（不存在/过小）
 3. 调用本地 Qwen2-VL-7B-Instruct 生成 bbox，要求输出 JSON 列表
 4. 将结果逐行写入新的 JSONL（保留原字段，新增 objects）

用法示例：
python scripts/annotate_wukong_qwen2vl.py \
  --input /mnt/disk/lxh/Project/gill-data/wukong_raw.jsonl \
  --image-root /mnt/disk/lxh/Project/gill-data/images \
  --output /mnt/disk/lxh/Project/gill-data/wukong_labeled.jsonl \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --device cuda \
  --max-samples 10000
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional, Set

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration


def is_valid_image(path: str, min_size: int = 256) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= min_size and h >= min_size
    except Exception:
        return False


def build_prompt(caption: str) -> str:
    # 仅关注描述中提到的实体，减少背景噪声
    return (
        "检测图像中与描述最相关的物体，只输出描述里出现的实体。"
        "严格返回 JSON 数组，每个元素形如："
        '{"name": "类别", "bbox": [x1, y1, x2, y2]}，坐标需 0-1 归一化；'
        "不要添加额外解释或文本。描述："
        f"{caption}"
    )


def parse_bboxes(text: str) -> List[Dict[str, Any]]:
    """
    尝试从模型输出中解析 JSON 数组；若失败则返回空列表。
    - 支持包含多段内容、markdown 代码块、多个数组等情况
    """
    # 去掉 markdown 代码块标记
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    # 可能存在多个数组：依次尝试非贪婪匹配的 [ ... ]
    for m in re.finditer(r"\[.*?\]", text, re.S):
        candidate = m.group(0)
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        # 只接受 list 且其中至少有一个带 bbox 的 dict
        if isinstance(data, list):
            valid: List[Dict[str, Any]] = []
            for obj in data:
                if (
                    isinstance(obj, dict)
                    and "bbox" in obj
                    and isinstance(obj["bbox"], list)
                    and len(obj["bbox"]) == 4
                ):
                    try:
                        bbox = [float(x) for x in obj["bbox"]]
                    except Exception:
                        continue
                    valid.append(
                        {
                            "name": obj.get("name", "object"),
                            "bbox": bbox,
                        }
                    )
            if valid:
                return valid

    # 兜底：有时模型直接输出单个 dict
    try:
        data = json.loads(text)
        if (
            isinstance(data, dict)
            and "bbox" in data
            and isinstance(data["bbox"], list)
            and len(data["bbox"]) == 4
        ):
            return [
                {
                    "name": data.get("name", "object"),
                    "bbox": [float(x) for x in data["bbox"]],
                }
            ]
    except Exception:
        pass

    return []


@torch.inference_mode()
def annotate_batch(
    model,
    processor,
    image_paths: List[str],
    captions: List[str],
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> List[Optional[List[Dict[str, Any]]]]:
    batch_images = []
    texts = []
    for path, cap in zip(image_paths, captions):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            batch_images.append(None)
            texts.append(None)
            continue
        prompt = build_prompt(cap)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch_images.append(img)
        texts.append(text)

    # 过滤掉损坏的样本
    valid_idx = [i for i, t in enumerate(texts) if t is not None]
    if not valid_idx:
        return [None] * len(image_paths)

    inputs = processor(
        text=[texts[i] for i in valid_idx],
        images=[batch_images[i] for i in valid_idx],
        return_tensors="pt",
        padding=True,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    generated_ids = [
        out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    text_outputs = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    # 简单调试：记录少量原始输出，便于检查格式
    try:
        if not hasattr(annotate_batch, "_log_count"):
            annotate_batch._log_count = 0
        if annotate_batch._log_count < 3:
            with open("qwen_vl_debug_outputs.txt", "a", encoding="utf-8") as f:
                for t in text_outputs:
                    f.write(t)
                    f.write("\n" + "-" * 60 + "\n")
            annotate_batch._log_count += 1
    except Exception:
        pass

    parsed_results = [None] * len(image_paths)
    for idx, out in zip(valid_idx, text_outputs):
        parsed_results[idx] = parse_bboxes(out)
    return parsed_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="未标注的 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--image-root", required=True, help="图片根目录")
    parser.add_argument(
        "--model", default="Qwen/Qwen2-VL-7B-Instruct", help="模型名称或路径"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = args.device
    print(f"🚀 加载模型 {args.model} 到 {device} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # 断点续传：读取已有输出的 image_path 集合
    processed: Set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as fexist:
            for line in fexist:
                try:
                    obj = json.loads(line)
                    p = obj.get("image_path") or obj.get("image")
                    if p:
                        processed.add(p)
                except Exception:
                    continue
        fout_mode = "a"
        print(f"🧩 检测到已存在输出，跳过 {len(processed)} 条，继续追加写入。")
    else:
        fout_mode = "w"

    total, success = 0, 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    buffer_items = []

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, fout_mode, encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            rel_path = item.get("image_path") or item.get("image")
            caption = item.get("caption") or ""
            if not rel_path:
                continue
            if rel_path in processed:
                continue

            full_path = rel_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(args.image_root, rel_path)
            if not is_valid_image(full_path, min_size=args.min_size):
                continue

            buffer_items.append((item, full_path, rel_path, caption))
            # 满批处理
            if len(buffer_items) >= args.batch_size:
                total += len(buffer_items)
                imgs = [x[1] for x in buffer_items]
                caps = [x[3] for x in buffer_items]
                results = annotate_batch(
                    model, processor, imgs, caps, device=device
                )
                for (itm, _, relp, _), bboxes in zip(buffer_items, results):
                    if bboxes:
                        itm["objects"] = bboxes
                        fout.write(json.dumps(itm, ensure_ascii=False) + "\n")
                        processed.add(relp)
                        success += 1
                buffer_items = []
                if args.max_samples and success >= args.max_samples:
                    break
                if success % 50 == 0:
                    print(f"✓ 已标注 {success}")

        # 处理残留不足一批的样本
        if buffer_items and (not args.max_samples or success < args.max_samples):
            total += len(buffer_items)
            imgs = [x[1] for x in buffer_items]
            caps = [x[3] for x in buffer_items]
            results = annotate_batch(
                model, processor, imgs, caps, device=device
            )
            for (itm, _, relp, _), bboxes in zip(buffer_items, results):
                if not bboxes:
                    continue
                # 过滤明显无效或“全图”框
                filtered = []
                for obj in bboxes:
                    name = str(obj.get("name", "")).lower()
                    bbox = obj.get("bbox", [])
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    x1, y1, x2, y2 = bbox
                    # 判定是否为 0-1 或 0-1000 坐标系下的全图框
                    is_full_01 = (
                        0.0 <= x1 <= 1.0
                        and 0.0 <= y1 <= 1.0
                        and 0.0 <= x2 <= 1.0
                        and 0.0 <= y2 <= 1.0
                        and x1 == 0.0
                        and y1 == 0.0
                        and x2 == 1.0
                        and y2 == 1.0
                    )
                    is_full_1000 = (
                        0.0 <= x1 <= 1000.0
                        and 0.0 <= y1 <= 1000.0
                        and 0.0 <= x2 <= 1000.0
                        and 0.0 <= y2 <= 1000.0
                        and x1 == 0.0
                        and y1 == 0.0
                        and x2 == 1000.0
                        and y2 == 1000.0
                    )
                    # 一些明显缺乏语义定位的类别名，也直接丢弃
                    if name in ["全图", "图片", "文字", "物体", "人名", "检测报告"]:
                        continue
                    if is_full_01 or is_full_1000:
                        continue
                    filtered.append({"name": obj.get("name", "object"), "bbox": bbox})

                if not filtered:
                    continue
                itm["objects"] = filtered
                fout.write(json.dumps(itm, ensure_ascii=False) + "\n")
                processed.add(relp)
                success += 1
            for (itm, _, relp, _), bboxes in zip(buffer_items, results):
                if not bboxes:
                    continue
                filtered = []
                for obj in bboxes:
                    name = str(obj.get("name", "")).lower()
                    bbox = obj.get("bbox", [])
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    x1, y1, x2, y2 = bbox
                    is_full_01 = (
                        0.0 <= x1 <= 1.0
                        and 0.0 <= y1 <= 1.0
                        and 0.0 <= x2 <= 1.0
                        and 0.0 <= y2 <= 1.0
                        and x1 == 0.0
                        and y1 == 0.0
                        and x2 == 1.0
                        and y2 == 1.0
                    )
                    is_full_1000 = (
                        0.0 <= x1 <= 1000.0
                        and 0.0 <= y1 <= 1000.0
                        and 0.0 <= x2 <= 1000.0
                        and 0.0 <= y2 <= 1000.0
                        and x1 == 0.0
                        and y1 == 0.0
                        and x2 == 1000.0
                        and y2 == 1000.0
                    )
                    if name in ["全图", "图片", "文字", "物体", "人名", "检测报告"]:
                        continue
                    if is_full_01 or is_full_1000:
                        continue
                    filtered.append({"name": obj.get("name", "object"), "bbox": bbox})

                if not filtered:
                    continue
                itm["objects"] = filtered
                fout.write(json.dumps(itm, ensure_ascii=False) + "\n")
                processed.add(relp)
                success += 1

    print(f"完成。成功 {success} 条。输出 -> {args.output}")


if __name__ == "__main__":
    main()

