#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多GPU并行标注脚本（Qwen2-VL-7B-Instruct）
功能：
 1. 读取未标注的悟空 JSONL（每行至少包含 image_path, caption）
 2. 过滤坏图（不存在/过小）
 3. 调用本地 Qwen2-VL-7B-Instruct 生成 bbox，要求输出 JSON 列表
 4. 将结果逐行写入新的 JSONL（保留原字段，新增 objects）
 5. 支持多GPU并行（3张卡）
 6. 详细的进度显示（已标注、剩余、速度、预计时间）

用法示例（单卡）：
python scripts/annotate_wukong_qwen2vl.py \
  --input /mnt/disk/lxh/Project/gill-data/wukong_raw.jsonl \
  --image-root /mnt/disk/lxh/Project/gill-data/images \
  --output /mnt/disk/lxh/Project/gill-data/wukong_labeled.jsonl \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --device cuda \
  --batch-size 4 \
  --max-samples 10000

用法示例（多GPU，3张卡）：
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/annotate_wukong_qwen2vl.py \
  --input /mnt/disk/lxh/Project/gill-data/wukong_raw.jsonl \
  --image-root /mnt/disk/lxh/Project/gill-data/images \
  --output /mnt/disk/lxh/Project/gill-data/wukong_labeled.jsonl \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --device cuda \
  --batch-size 4 \
  --num-gpus 3
"""

import argparse
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Set
from collections import deque

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from tqdm import tqdm


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


def filter_bboxes(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤明显无效或"全图"框"""
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
    return filtered


@torch.inference_mode()
def annotate_batch(
    model,
    processor,
    image_paths: List[str],
    captions: List[str],
    device: str = "cuda",
    max_new_tokens: int = 256,
    max_image_size: int = 1024,  # 限制图片最大尺寸，减少显存占用
) -> List[Optional[List[Dict[str, Any]]]]:
    batch_images = []
    texts = []
    for path, cap in zip(image_paths, captions):
        try:
            img = Image.open(path).convert("RGB")
            # 限制图片尺寸，减少显存占用
            w, h = img.size
            if w > max_image_size or h > max_image_size:
                ratio = min(max_image_size / w, max_image_size / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
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

    # 清理显存缓存
    torch.cuda.empty_cache()
    
    output_ids = None
    generated_ids = None
    text_outputs = None
    
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        generated_ids = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        text_outputs = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
    except torch.cuda.OutOfMemoryError as e:
        # 清理显存后重新抛出异常
        if output_ids is not None:
            del output_ids
        if generated_ids is not None:
            del generated_ids
        if text_outputs is not None:
            del text_outputs
        del inputs
        torch.cuda.empty_cache()
        raise e
    finally:
        # 清理输入和中间变量（如果存在）
        if 'inputs' in locals():
            del inputs
        if output_ids is not None:
            del output_ids
        if generated_ids is not None:
            del generated_ids
        torch.cuda.empty_cache()

    if text_outputs is None:
        # 如果生成失败，返回空结果
        return [None] * len(image_paths)

    parsed_results = [None] * len(image_paths)
    for idx, out in zip(valid_idx, text_outputs):
        parsed_results[idx] = parse_bboxes(out)
    
    # 清理文本输出
    del text_outputs
    torch.cuda.empty_cache()
    
    return parsed_results


def load_models_multi_gpu(model_path: str, num_gpus: int = 3):
    """在多GPU上加载模型"""
    models = []
    processors = []
    devices = []
    
    # 检查是否为本地路径
    is_local = os.path.exists(model_path) and os.path.isdir(model_path)
    
    print(f"🚀 在 {num_gpus} 张 GPU 上加载模型 {model_path} ...")
    if is_local:
        print(f"  📦 检测到本地模型路径，使用 local_files_only=True")
    else:
        print(f"  📦 从 HuggingFace 加载模型")
    
    for i in range(num_gpus):
        device = f"cuda:{i}"
        print(f"  📦 加载到 {device} ...")
        
        # 加载模型
        if is_local:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
                local_files_only=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        
        models.append(model)
        processors.append(processor)
        devices.append(device)
        print(f"  ✅ {device} 加载完成")
    
    return models, processors, devices


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
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小（多GPU时建议1-2，单GPU可更大）")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的GPU数量（默认1，支持多GPU并行）")
    parser.add_argument("--save-interval", type=int, default=50, help="每处理多少条保存一次进度信息")
    parser.add_argument("--max-image-size", type=int, default=1024, help="图片最大尺寸（像素），超过会缩放，减少显存占用")
    args = parser.parse_args()

    # 检查是否为本地路径
    is_local = os.path.exists(args.model) and os.path.isdir(args.model)
    
    # 多GPU支持
    if args.num_gpus > 1:
        models, processors, devices = load_models_multi_gpu(args.model, args.num_gpus)
        current_gpu = 0  # 轮询使用GPU
    else:
        device = args.device
        print(f"🚀 加载模型 {args.model} 到 {device} ...")
        if is_local:
            print(f"  📦 检测到本地模型路径，使用 local_files_only=True")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model,
                device_map=device,
                torch_dtype="auto",
                local_files_only=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                args.model,
                local_files_only=True,
                trust_remote_code=True,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                args.model,
                trust_remote_code=True,
            )
        models = [model]
        processors = [processor]
        devices = [device]

    # 断点续传：读取已有输出的 image_path 集合
    processed: Set[str] = set()
    initial_success = 0
    if os.path.exists(args.output):
        print(f"📖 读取已有输出文件: {args.output}")
        with open(args.output, "r", encoding="utf-8") as fexist:
            for line in fexist:
                try:
                    obj = json.loads(line)
                    p = obj.get("image_path") or obj.get("image")
                    if p:
                        processed.add(p)
                        initial_success += 1
                except Exception:
                    continue
        fout_mode = "a"
        print(f"🧩 检测到已存在输出，已标注 {initial_success} 条，继续追加写入。")
    else:
        fout_mode = "w"
        print(f"📝 创建新输出文件: {args.output}")

    # 统计总行数（用于进度条）
    print("📊 统计输入文件总行数...")
    total_lines = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for _ in fin:
            total_lines += 1
    print(f"   总行数: {total_lines}")

    total, success = 0, initial_success
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    buffer_items = []
    
    # 进度统计
    start_time = time.time()
    speed_history = deque(maxlen=100)  # 记录最近100个批次的速度
    
    # 打开文件
    fin = open(args.input, "r", encoding="utf-8")
    fout = open(args.output, fout_mode, encoding="utf-8")
    
    try:
        # 使用 tqdm 显示进度
        pbar = tqdm(total=total_lines, desc="标注进度", unit="行", initial=len(processed))
        
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                pbar.update(1)
                continue
            try:
                item = json.loads(line)
            except Exception:
                pbar.update(1)
                continue

            rel_path = item.get("image_path") or item.get("image")
            caption = item.get("caption") or ""
            if not rel_path:
                pbar.update(1)
                continue
            if rel_path in processed:
                pbar.update(1)
                continue

            full_path = rel_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(args.image_root, rel_path)
            if not is_valid_image(full_path, min_size=args.min_size):
                pbar.update(1)
                continue

            buffer_items.append((item, full_path, rel_path, caption))
            
            # 满批处理
            if len(buffer_items) >= args.batch_size:
                batch_start_time = time.time()
                total += len(buffer_items)
                
                # 选择GPU（多GPU时轮询）
                if args.num_gpus > 1:
                    gpu_idx = current_gpu % args.num_gpus
                    current_gpu += 1
                    model = models[gpu_idx]
                    processor = processors[gpu_idx]
                    device = devices[gpu_idx]
                else:
                    model = models[0]
                    processor = processors[0]
                    device = devices[0]
                
                imgs = [x[1] for x in buffer_items]
                caps = [x[3] for x in buffer_items]
                
                try:
                    results = annotate_batch(
                        model, processor, imgs, caps, device=device,
                        max_image_size=args.max_image_size
                    )
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n❌ GPU显存不足！建议：")
                    print(f"   1. 减少批次大小：--batch-size 1")
                    print(f"   2. 使用更少的GPU：--num-gpus 2 或 --num-gpus 1")
                    print(f"   3. 减小图片尺寸：--max-image-size 512")
                    print(f"   4. 清理其他占用显存的进程")
                    raise e
                
                batch_count = 0
                for (itm, _, relp, _), bboxes in zip(buffer_items, results):
                    if not bboxes:
                        continue
                    
                    filtered = filter_bboxes(bboxes)
                    if not filtered:
                        continue
                    
                    itm["objects"] = filtered
                    fout.write(json.dumps(itm, ensure_ascii=False) + "\n")
                    fout.flush()  # 立即刷新到磁盘
                    processed.add(relp)
                    success += 1
                    batch_count += 1
                
                # 清理批次数据
                del results
                del imgs
                del caps
                torch.cuda.empty_cache()
                
                # 更新进度统计
                batch_time = time.time() - batch_start_time
                batch_speed = len(buffer_items) / batch_time if batch_time > 0 else 0
                speed_history.append(batch_speed)
                avg_speed = sum(speed_history) / len(speed_history) if speed_history else 0
                
                buffer_items = []
                pbar.update(1)
                
                # 定期输出详细进度
                if success % args.save_interval == 0:
                    elapsed = time.time() - start_time
                    remaining = (total_lines - len(processed)) / avg_speed if avg_speed > 0 else 0
                    
                    progress_info = (
                        f"✅ 已标注: {success} | "
                        f"剩余: {total_lines - len(processed)} | "
                        f"速度: {avg_speed:.2f} 样本/秒 | "
                        f"已用: {elapsed/3600:.1f}h | "
                        f"预计剩余: {remaining/3600:.1f}h"
                    )
                    if args.num_gpus > 1:
                        progress_info += f" | GPU: {args.num_gpus}张并行"
                    print(f"\n{progress_info}")
                    pbar.set_postfix({
                        "已标注": success,
                        "速度": f"{avg_speed:.2f}/s",
                        "剩余": f"{(remaining/3600):.1f}h"
                    })
                
                if args.max_samples and success >= args.max_samples:
                    break

        # 处理残留不足一批的样本
        if buffer_items and (not args.max_samples or success < args.max_samples):
            batch_start_time = time.time()
            total += len(buffer_items)
            
            if args.num_gpus > 1:
                gpu_idx = current_gpu % args.num_gpus
                model = models[gpu_idx]
                processor = processors[gpu_idx]
                device = devices[gpu_idx]
            else:
                model = models[0]
                processor = processors[0]
                device = devices[0]
            
            imgs = [x[1] for x in buffer_items]
            caps = [x[3] for x in buffer_items]
            
            try:
                results = annotate_batch(
                    model, processor, imgs, caps, device=device,
                    max_image_size=args.max_image_size
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n❌ GPU显存不足！建议：")
                print(f"   1. 减少批次大小：--batch-size 1")
                print(f"   2. 使用更少的GPU：--num-gpus 2 或 --num-gpus 1")
                print(f"   3. 减小图片尺寸：--max-image-size 512")
                print(f"   4. 清理其他占用显存的进程")
                raise e
            
            for (itm, _, relp, _), bboxes in zip(buffer_items, results):
                if not bboxes:
                    continue
                
                filtered = filter_bboxes(bboxes)
                if not filtered:
                    continue
                
                itm["objects"] = filtered
                fout.write(json.dumps(itm, ensure_ascii=False) + "\n")
                fout.flush()
                processed.add(relp)
                success += 1
            
            pbar.update(1)
        
        pbar.close()
        
    finally:
        fin.close()
        fout.close()

    elapsed = time.time() - start_time
    print(f"\n✅ 标注完成！")
    print(f"📊 统计:")
    print(f"   成功标注: {success} 条")
    print(f"   总处理: {total} 条")
    print(f"   成功率: {success/total*100:.1f}%" if total > 0 else "   成功率: N/A")
    print(f"   总耗时: {elapsed/3600:.2f} 小时")
    print(f"   平均速度: {success/elapsed:.2f} 样本/秒" if elapsed > 0 else "   平均速度: N/A")
    print(f"💾 输出文件: {args.output}")


if __name__ == "__main__":
    main()
