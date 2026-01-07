#!/usr/bin/env python3
"""
Layout Planner 训练脚本

使用 layout_dataset_qwen2vl_heuristic_4k.jsonl 对布局规划器进行指令微调。

数据格式（JSONL 每行）：
{
  "caption": "上方是蓝天，下方是草地",
  "image_path": "xxxxx.jpg",
  "objects": [
    {"name": "天空", "bbox": [0.0, 0.0, 1.0, 0.4]},
    {"name": "草地", "bbox": [0.0, 0.6, 1.0, 1.0]}
  ]
}

会被转成训练样本：
input : caption
output: <obj>天空</obj><box>[0.00,0.00,1.00,0.40]</box><obj>草地</obj><box>[0.00,0.60,1.00,1.00]</box>

然后使用 gill.layout_planner.train_layout_planner 进行 CAUSAL LM 训练。
"""

import os
import sys
import json
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# 保证可以 import gill
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.layout_planner import LayoutPlanner, train_layout_planner  # type: ignore


class LayoutJsonlDataset(Dataset):
    """从 JSONL 布局数据集中构造 Layout Planner 指令样本。"""

    def __init__(self, jsonl_path: str, max_samples: int = -1):
        self.samples: List[Dict] = []
        assert os.path.exists(jsonl_path), f"JSONL 不存在: {jsonl_path}"

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                # 支持两种格式：
                # 1. CoT 格式：直接有 input/output 字段（来自 prepare_cot_training_data.py）
                # 2. 传统格式：从 caption 和 objects 构造
                if "input" in item and "output" in item:
                    # CoT 格式：直接使用
                    input_text = str(item.get("input", "")).strip()
                    output_text = str(item.get("output", "")).strip()
                    if input_text and output_text:
                        self.samples.append({"input": input_text, "output": output_text})
                        if max_samples > 0 and len(self.samples) >= max_samples:
                            break
                        continue
                
                # 传统格式：从 caption 和 objects 构造
                caption = str(item.get("caption", "")).strip()
                objects = item.get("objects", []) or []
                if not caption or not objects:
                    continue

                parts: List[str] = []
                for obj in objects:
                    name = str(obj.get("name", "")).strip() or "物体"
                    # 支持两种 bbox 格式：0-1 浮点数 或 0-1000 整数
                    bbox = obj.get("bbox", [])
                    bbox_1000 = obj.get("bbox_1000", [])
                    
                    if bbox_1000 and len(bbox_1000) == 4:
                        # 使用 0-1000 整数格式
                        bbox_str = ",".join(f"{int(v)}" for v in bbox_1000)
                    elif bbox and len(bbox) == 4:
                        # 使用 0-1 浮点数格式
                        try:
                            bbox_f = [float(v) for v in bbox]
                            bbox_str = ",".join(f"{v:.2f}" for v in bbox_f)
                        except Exception:
                            continue
                    else:
                        continue
                    
                    parts.append(f"<obj>{name}</obj><box>[{bbox_str}]</box>")

                if not parts:
                    continue

                output_text = "".join(parts)
                self.samples.append({"input": caption, "output": output_text})

                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

        print(f"✓ 从 {jsonl_path} 读取到 {len(self.samples)} 条训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """保持 batch 为 list[dict]，方便 train_layout_planner 直接使用。"""
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Planner 训练脚本")
    parser.add_argument(
        "--layout-json",
        type=str,
        default="data/layout_dataset_qwen2vl_heuristic_4k.jsonl",
        help="布局数据集 JSONL 路径",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="./model/deepseek-llm-7b-base",
        help="Layout Planner 基座模型路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda/cpu/cuda:0,1 等多 GPU，或 'auto' 自动分配)",
    )
    parser.add_argument(
        "--use-lora",
        nargs='?',
        const=True,
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="是否使用 LoRA（默认 True；传 --use-lora False 关闭，若环境未安装 peft 会自动退回全参数训练）",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="最多使用多少条样本（-1 表示使用全部）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/layout_planner",
        help="保存 LoRA/模型权重的目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 如果指定了多 GPU，设置 CUDA_VISIBLE_DEVICES
    if "," in args.device or args.device == "auto":
        # 提取 GPU 编号（如 "cuda:0,1" -> ["0", "1"]）
        if "," in args.device:
            gpu_ids = args.device.replace("cuda:", "").split(",")
        else:
            # 默认使用 GPU 0 和 1
            gpu_ids = ["0", "1"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"✓ 设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        # 重置 device 为 "auto"，让 transformers 自动分配
        actual_device = "auto"
    else:
        actual_device = args.device

    print("=" * 60)
    print("🚀 Layout Planner 训练")
    print("=" * 60)
    print(f"数据集: {args.layout_json}")
    print(f"基座模型: {args.base_model}")
    print(f"设备: {args.device}")
    print(f"batch size: {args.batch_size}, epochs: {args.epochs}, lr: {args.lr}")

    # 1) 构造数据集与 DataLoader
    dataset = LayoutJsonlDataset(args.layout_json, max_samples=args.max_samples)
    if len(dataset) == 0:
        print("❌ 数据集为空，退出")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 2) 创建 LayoutPlanner（多 GPU 支持）
    planner = LayoutPlanner(args.base_model, device=actual_device, use_lora=args.use_lora)

    # 3) 优化器（全参数微调时优先使用 8-bit 优化器节省显存）
    if not args.use_lora:
        try:
            import bitsandbytes as bnb  # type: ignore
            print("✓ 使用 8-bit AdamW 优化器（节省显存）")
            optimizer = bnb.optim.AdamW8bit(planner.model.parameters(), lr=args.lr)
        except ImportError:
            print("⚠️ bitsandbytes 未安装，回退到标准 AdamW（显存占用较高）")
            optimizer = torch.optim.AdamW(planner.model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(planner.model.parameters(), lr=args.lr)

    # 4) 训练
    planner = train_layout_planner(
        planner,
        train_loader=loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=actual_device,
    )

    # 5) 保存权重
    try:
        if hasattr(planner.model, "save_pretrained"):
            out_dir = os.path.join(args.save_dir, "final")
            os.makedirs(out_dir, exist_ok=True)
            planner.model.save_pretrained(out_dir)
            if hasattr(planner, "tokenizer") and hasattr(planner.tokenizer, "save_pretrained"):
                planner.tokenizer.save_pretrained(out_dir)
            print(f"✓ 模型已保存到: {out_dir}")
        else:
            # 对于 LoRA，可以考虑使用 peft 的 save_pretrained，这里先简单保存 state_dict
            out_path = os.path.join(args.save_dir, "planner_model.pt")
            torch.save(planner.model.state_dict(), out_path)
            print(f"✓ 模型 state_dict 已保存到: {out_path}")
    except Exception as e:
        print(f"⚠️ 保存模型时出错: {e}")


if __name__ == "__main__":
    main()
