#!/usr/bin/env python3
"""
Layout Planner 训练脚本 (Refactored)

使用 Hugging Face Trainer 和 DataCollator，实现正确的 Label Masking。
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict
from dataclasses import dataclass
import inspect

import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    set_seed
)

# 保证可以 import gill
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COCO 80 类别 ID 到中文名称的映射
COCO_CATEGORY_ID_TO_CHINESE = {
    1: "人", 2: "自行车", 3: "汽车", 4: "摩托车", 5: "飞机", 6: "公交车", 7: "火车", 8: "卡车",
    9: "船", 10: "交通灯", 11: "消防栓", 12: "停止标志", 13: "停车计时器", 14: "长椅", 15: "鸟",
    16: "猫", 17: "狗", 18: "马", 19: "羊", 20: "牛", 21: "大象", 22: "熊", 23: "斑马",
    24: "长颈鹿", 25: "背包", 26: "雨伞", 27: "手提包", 28: "领带", 29: "行李箱", 30: "飞盘",
    31: "滑雪板", 32: "滑雪板", 33: "运动球", 34: "风筝", 35: "棒球棒", 36: "棒球手套",
    37: "滑板", 38: "冲浪板", 39: "网球拍", 40: "瓶子", 41: "酒杯", 42: "杯子", 43: "叉子",
    44: "刀", 45: "勺子", 46: "碗", 47: "香蕉", 48: "苹果", 49: "三明治", 50: "橙子",
    51: "西兰花", 52: "胡萝卜", 53: "热狗", 54: "披萨", 55: "甜甜圈", 56: "蛋糕", 57: "椅子",
    58: "沙发", 59: "盆栽", 60: "床", 61: "餐桌", 62: "厕所", 63: "电视", 64: "笔记本电脑",
    65: "鼠标", 66: "遥控器", 67: "键盘", 68: "手机", 69: "微波炉", 70: "烤箱", 71: "烤面包机",
    72: "水槽", 73: "冰箱", 74: "书", 75: "时钟", 76: "花瓶", 77: "剪刀", 78: "泰迪熊",
    79: "吹风机", 80: "牙刷"
}

FORMAT_INSTRUCTION = (
    "请严格按照以下格式输出：<obj>名称</obj><box>[x1,y1,x2,y2]</box>... "
    "如果无法给出布局，请输出 <no_layout>。只输出格式，不要解释。"
)

class LayoutJsonlDataset(Dataset):
    """从 JSONL 布局数据集中构造 Layout Planner 指令样本。"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_samples: int = -1):
        self.samples = []
        if not os.path.exists(jsonl_path):
            print(f"❌ 错误: 文件不存在 {jsonl_path}")
            return

        print(f"📖 正在加载: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    
                    # 1. 优先处理 CoT 格式 (直接有 input/output)
                    if "input" in item and "output" in item:
                        inp, out = str(item["input"]).strip(), str(item["output"]).strip()
                        if inp and out:
                            self.samples.append({"input": inp, "output": out})
                        continue

                    # 2. 处理标准 caption + objects 格式
                    inp = str(item.get("caption", "")).strip()
                    objs = item.get("objects", [])
                    if not inp:
                        continue
                    if not isinstance(objs, list):
                        objs = []
                    if not objs:
                        self.samples.append({"input": inp, "output": "<no_layout>"})
                        continue

                    # 获取图像尺寸
                    width = float(item.get("width", 0))
                    height = float(item.get("height", 0))
                    has_dim = width > 0 and height > 0

                    out_parts = []
                    for obj in objs:
                        # --- 名称处理 ---
                        name = str(obj.get("name", "")).strip()
                        if not name:
                            category_id = obj.get("category_id")
                            if category_id and category_id in COCO_CATEGORY_ID_TO_CHINESE:
                                name = COCO_CATEGORY_ID_TO_CHINESE[category_id]
                            else:
                                name = "物体"

                        # --- 坐标处理 ---
                        bbox = obj.get("bbox", [])
                        bbox_1000 = obj.get("bbox_1000", [])
                        bbox_final = None 

                        # 优先级 1: 明确的 0-1000 格式（最可靠）
                        if bbox_1000 and len(bbox_1000) == 4:
                            bbox_final = [float(v) / 1000.0 for v in bbox_1000]

                        # 优先级 2: 通用 bbox 处理
                        elif bbox and len(bbox) == 4:
                            bbox_raw = [float(v) for v in bbox]
                            max_val = max(bbox_raw)

                            # 情况 A: 已经是 0-1 格式（最优先判断，避免误判）
                            if max_val <= 1.05:
                                bbox_final = bbox_raw
                            
                            # 情况 B: 有宽高信息 -> 强制按像素归一化（针对 COCO-CN 等像素坐标）
                            elif has_dim:
                                bbox_final = [
                                    bbox_raw[0] / width, 
                                    bbox_raw[1] / height,
                                    bbox_raw[2] / width, 
                                    bbox_raw[3] / height
                                ]
                            
                            # 情况 C: 无宽高信息，但值 <= 1000 -> 只能假设是 0-1000 格式（兜底方案）
                            elif max_val <= 1000:
                                bbox_final = [v / 1000.0 for v in bbox_raw]
                            
                            # 情况 D: 无法处理，跳过
                            else:
                                continue

                        if bbox_final:
                            # 截断到 0-1 范围
                            bbox_final = [max(0.0, min(1.0, v)) for v in bbox_final]
                            bbox_str = ",".join(f"{v:.2f}" for v in bbox_final)
                            out_parts.append(f"<obj>{name}</obj><box>[{bbox_str}]</box>")
                    
                    if out_parts:
                        out = "".join(out_parts)
                        self.samples.append({"input": inp, "output": out})

                    if max_samples > 0 and len(self.samples) >= max_samples:
                        break

                except Exception:
                    continue
        
        print(f"✓ 加载 {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

@dataclass
class DataCollatorForLayoutPlanner:
    """
    关键组件：正确处理 Chat Template 和 Label Masking
    """
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        
        for example in examples:
            user_input = example["input"]
            if "<obj>" not in user_input and "只输出格式" not in user_input:
                user_input = f"{user_input}\n\n{FORMAT_INSTRUCTION}"
            # 1. 构建完整对话
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": example["output"]}
            ]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # 2. 构建 Prompt 部分
            user_msg = [{"role": "user", "content": user_input}]
            prompt_text = self.tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
            
            # 3. Tokenize
            full_ids = self.tokenizer(full_text, add_special_tokens=False, max_length=self.max_length, truncation=True).input_ids
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, max_length=self.max_length, truncation=True).input_ids
            
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = input_ids.clone()
            
            # 4. Masking
            prompt_len = len(prompt_ids)
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                labels[:] = -100
                
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
        # 5. Padding
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        input_ids_padded = []
        labels_padded = []
        attention_mask_list = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids_padded.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
                labels_padded.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))
                attention_mask_list.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
            else:
                input_ids_padded.append(input_ids[:max_len])
                labels_padded.append(labels[:max_len])
                attention_mask_list.append(torch.ones(max_len, dtype=torch.long))
        
        return {
            "input_ids": torch.stack(input_ids_padded),
            "labels": torch.stack(labels_padded),
            "attention_mask": torch.stack(attention_mask_list)
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Planner 训练脚本")
    parser.add_argument("--layout-json", type=str, default="data/layout_planner_train.jsonl")
    parser.add_argument("--base-model", type=str, default="./model/qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/layout_planner")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--val-json", type=str, default="data/coco-cn/coco-cn_val.jsonl")
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--load-best-model-at-end", action="store_true", default=False,
                       help="是否加载验证集上最佳模型（False=使用最后一个epoch的模型，推荐用于格式要求高的任务）")
    parser.add_argument("--save-total-limit", type=int, default=3,
                       help="保存的checkpoint数量限制")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（默认 42）")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory (e.g., output_dir/checkpoint-xxxx).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path to warm-start training.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Trainer optimizer name (e.g., adamw_torch, adafactor, adamw_hf)"
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="Per-GPU max memory in GiB for device_map auto (e.g., 23)."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Use model sharding with device_map=auto, or 'none' for DDP (torchrun)."
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader worker processes (0 disables multiprocessing)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    
    # ... (初始化 Tokenizer 和 Model 部分保持不变，此处省略以节省篇幅，请保留原代码中的加载逻辑)
    # 简写如下：
    print("\n📦 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<obj>", "</obj>", "<box>", "</box>", "<no_layout>"]})
    
    print("\n📦 加载模型...")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    use_device_map = args.device_map != "none"
    if local_rank >= 0 and use_device_map:
        print("DDP detected; forcing device_map=none for torchrun.")
        use_device_map = False

    max_memory = None
    device_map = "auto" if use_device_map else None
    if use_device_map and args.max_memory_gb is not None and torch.cuda.is_available():
        max_memory = {i: f"{args.max_memory_gb}GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    if not use_device_map and torch.cuda.is_available():
        target_device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
        model.to(target_device)
    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # Keep embeddings and lm_head trainable so new special tokens can be learned.
            modules_to_save=["embed_tokens", "lm_head"],
        )
        if args.adapter_path and os.path.exists(args.adapter_path):
            print("Loading LoRA adapter from: " + str(args.adapter_path))
            model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        else:
            model = get_peft_model(model, peft_config)
    
    # 准备数据集
    print("\n📖 准备数据集...")
    train_dataset = LayoutJsonlDataset(args.layout_json, tokenizer, max_samples=args.max_samples)
    
    # 准备验证集
    if args.val_json and os.path.exists(args.val_json):
        print(f"📊 使用独立验证集: {args.val_json}")
        val_dataset = LayoutJsonlDataset(args.val_json, tokenizer, max_samples=-1)
    else:
        # 回退逻辑
        val_size = int(len(train_dataset) * args.val_split_ratio)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-val_size, val_size])
    
    collator = DataCollatorForLayoutPlanner(tokenizer=tokenizer, max_length=512)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,  # loss 越小越好
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        optim=args.optim
    )
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    
    print("\n🚀 开始训练...")
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint: " + str(args.resume_from_checkpoint))
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 完成: {save_path}")

if __name__ == "__main__":
    main()
