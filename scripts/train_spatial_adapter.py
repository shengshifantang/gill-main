#!/usr/bin/env python3
"""
训练 Spatial Adapter（适配 Kolors/SDXL）- 防 NaN 增强版（Mixed Precision Training）

核心功能：
1. 加载 Kolors Pipeline（冻结 UNet/VAE/TextEncoder）
2. 加载混合布局数据集（JSONL）
3. 动态注入 Spatial Adapter 到 UNet 的所有 Attention 层
4. 混合精度训练：FP32 权重 + FP16 计算 + GradScaler

Usage:
    python scripts/train_spatial_adapter.py \
        --mixed-data data/mixed_training_65k.jsonl \
        --kolors-model ./model/Kolors \
        --output-dir ./checkpoints/spatial_adapter_mixed \
        --batch-size 4 \
        --epochs 5
"""

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

# 尝试导入 diffusers 组件
try:
    from diffusers import KolorsPipeline, DDPMScheduler
    from diffusers.optimization import get_scheduler
except ImportError:
    print("❌ 未安装 diffusers，请运行: pip install diffusers accelerate")
    sys.exit(1)

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.spatial_adapter import (
    inject_spatial_control_to_unet, 
    remove_spatial_control_from_unet, 
    create_spatial_adapter_for_kolors,
    SpatialAdapterModuleDict
)


class MixedLayoutDataset(Dataset):
    """
    混合布局数据集
    
    支持格式：
    {
        "image_path": "path/to/img.jpg",
        "caption": "描述文本",
        "objects": [{"name": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}, ...],
        "has_layout": true/false
    }
    """
    def __init__(self, jsonl_path: str, image_dir: str = None, resolution: int = 1024):
        self.samples = []
        self.image_dir = image_dir
        self.resolution = resolution
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
            
        print(f"📖 读取数据: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if 'caption' in item:
                            self.samples.append(item)
                    except:
                        continue
        print(f"✓ 加载 {len(self.samples)} 条数据")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 加载图像
        image_path = item.get('image_path', '')
        if self.image_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_dir, image_path)
            
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.resolution, self.resolution))
                pixel_values = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
                pixel_values = pixel_values.permute(2, 0, 1) # [3, H, W]
                # 强制 clamp 到 [-1, 1]，防止极端值导致 VAE NaN
                pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            else:
                # Dummy image for testing
                pixel_values = torch.randn(3, self.resolution, self.resolution)
                pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
        except Exception as e:
            # 图片加载失败，使用安全的 dummy 数据
            pixel_values = torch.randn(3, self.resolution, self.resolution)
            pixel_values = torch.clamp(pixel_values, -1.0, 1.0)

        # 2. 处理 BBox（统一归一化到 0-1，并过滤几何极端样本），同时收集对象名称
        objects = item.get('objects', [])
        bboxes = []
        obj_names = []
        for obj in objects:
            bbox = obj.get('bbox', [])
            if len(bbox) == 4:
                # 兼容 0-1000 和 0-1
                x1, y1, x2, y2 = bbox
                if max(x1, y1, x2, y2) > 1.5:
                    bbox = [x / 1000.0 for x in bbox]
                    x1, y1, x2, y2 = bbox
                
                # 几何过滤：过滤极小框和几乎全图的框
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h
                if 0.02 < area < 0.9 and w > 0.03 and h > 0.03:
                    bboxes.append([x1, y1, x2, y2])
                    obj_names.append(obj.get("name", ""))
        
        return {
            'pixel_values': pixel_values,
            'caption': item.get('caption', ''),
            'bboxes': bboxes,
            'obj_names': obj_names,
            'has_layout': len(bboxes) > 0
        }


def collate_mixed_batch(batch):
    # 过滤掉 None 样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    max_boxes = max(len(item['bboxes']) for item in batch)
    max_boxes = max(max_boxes, 1)
    
    bboxes_padded = []
    masks = []
    obj_names_batched = []
    
    for item in batch:
        boxes = item['bboxes']
        names = item.get('obj_names', [])
        num_boxes = len(boxes)
        padded = boxes + [[0.0]*4] * (max_boxes - num_boxes)
        bboxes_padded.append(padded)
        masks.append([1]*num_boxes + [0]*(max_boxes - num_boxes))
        # 名称按相同长度 padding，空字符串表示无对象
        padded_names = names + [""] * (max_boxes - num_boxes)
        obj_names_batched.append(padded_names)
        
    bboxes_tensor = torch.tensor(bboxes_padded, dtype=torch.float32)
    masks_tensor = torch.tensor(masks, dtype=torch.float32)
    
    return {
        'pixel_values': pixel_values,
        'captions': captions,
        'bboxes': bboxes_tensor,
        'masks': masks_tensor,
        'obj_names': obj_names_batched
    }


def _get_add_time_ids(bs, device, original_size=(1024, 1024), target_size=(1024, 1024), crops_coords_top_left=(0, 0)):
    # SDXL/Kolors 需要的 time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.long, device=device)
    return add_time_ids.repeat(bs, 1)


def train_spatial_adapter(
    mixed_data_path: str, 
    kolors_model_path: str, 
    output_dir: str, 
    batch_size: int = 4, 
    epochs: int = 5,
    lr: float = 1e-4, 
    device: str = "cuda:0",
    image_dir: str = None
):
    print(f"🚀 初始化 Kolors Spatial Adapter 训练 (Mixed Precision)...")
    print(f"   Model: {kolors_model_path}")
    print(f"   Data: {mixed_data_path}")
    
    # 1. 加载组件 (FP16)
    try:
        # 注意：不传 variant="fp16"，只传 torch_dtype，避免 IndexError
        pipeline = KolorsPipeline.from_pretrained(
            kolors_model_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)

        # 修复 Kolors Tokenizer 不支持 padding_side 参数的问题
        if hasattr(pipeline, "tokenizer") and pipeline.tokenizer is not None:
            if hasattr(pipeline.tokenizer, "_pad"):
                original_pad = pipeline.tokenizer._pad
                def compatible_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
                    kwargs.pop("padding_side", None)
                    return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask, **kwargs)
                pipeline.tokenizer._pad = compatible_pad
                print("✓ 已修复 Kolors Tokenizer padding 兼容性")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 提取组件并冻结
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # 【修复】强制 VAE 使用 FP32，防止 NaN（SDXL/Kolors VAE 在 FP16 下不稳定）
    vae.to(dtype=torch.float32)
    print("✓ VAE 已切换到 FP32 精度（防止数值溢出）")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # 【显存优化1】启用 UNet 的 gradient checkpointing（必须启用，否则 OOM）
    # 注意：checkpointing 必须在 train 模式下才能工作
    unet.train()
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        print("✓ 已启用 UNet gradient checkpointing（节省显存）")
    
    # 【显存优化2】VAE CPU Offload；Text Encoder 常驻 GPU（此前反复 .to 过慢并曾触发中断）
    # UNet 必须一直在 GPU（因为训练循环中多次调用）
    vae.to("cpu", dtype=torch.float32)
    text_encoder.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float16)
    print("✓ 已启用 VAE CPU Offload；Text Encoder/UNet 常驻 GPU")
    
    # 2. 初始化 Adapter (保持 FP32 以稳定训练)
    print("📦 初始化 Adapter 容器 (FP32)...")
    adapter_container = create_spatial_adapter_for_kolors() 
    adapter_container.to(device, dtype=torch.float32)  # 明确指定 FP32
    
    # 3. 优化器（不使用 GradScaler，因为 Adapter 是 FP32，不需要混合精度）
    # 注意：UNet 是 FP16 但被冻结，只有 Adapter (FP32) 需要梯度
    optimizer = None 
    
    # 4. 数据加载
    dataset = MixedLayoutDataset(mixed_data_path, image_dir=image_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_mixed_batch,
        num_workers=0
    )
    
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    
    for epoch in range(epochs):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            if batch is None:
                continue
                
            # --- A. 准备 Latents (VAE CPU Offload) ---
            with torch.no_grad():
                # 【显存优化】临时将 VAE 移到 GPU
                vae.to(device)
                torch.cuda.empty_cache()  # 清理碎片
                
                # 【修复】图像转为 FP32 进入 VAE（VAE 必须用 FP32）
                pixel_values = batch['pixel_values'].to(device, dtype=torch.float32)
                # 再次 clamp，确保输入 VAE 的值在合法范围
                pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
                
                try:
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # 【重要】编码完后再转回 FP16 给 UNet 用（节省显存）
                    latents = latents.to(dtype=torch.float16)
                    
                    # 数据检查：防止坏数据导致的 NaN
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        print(f"⚠️ 警告: 检测到 VAE 输出 NaN/Inf，跳过此 Batch (step {global_step})")
                        vae.to("cpu")  # 出错也要移回 CPU
                        continue
                except Exception as e:
                    print(f"⚠️ 警告: VAE 编码失败: {e}，跳过此 Batch (step {global_step})")
                    vae.to("cpu")  # 出错也要移回 CPU
                    continue
                finally:
                    # 【显存优化】VAE 用完立即移回 CPU，释放显存
                    vae.to("cpu")
                    torch.cuda.empty_cache()

                noise = torch.randn_like(latents)
                bs = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # --- B. 准备 Text Embeddings (Text Encoder 常驻 GPU) ---
                try:
                    encoded = pipeline.encode_prompt(
                        prompt=batch['captions'], 
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False 
                    )
                    
                    if isinstance(encoded, tuple) and len(encoded) >= 3:
                        prompt_embeds = encoded[0]
                        pooled_embeds = encoded[2]
                    else:
                        continue
                except Exception as e:
                    print(f"⚠️ 警告: Text Encoder 编码失败: {e}，跳过此 Batch (step {global_step})")
                    continue
                
                # === 额外：编码物体名称为 phrase_embeddings (用于语义绑定) ===
                obj_names_batch = batch.get('obj_names', [])
                max_boxes = batch['bboxes'].shape[1]
                text_hidden = getattr(text_encoder.config, "hidden_size", 4096)
                
                # 展平名称列表，空字符串保留为占位
                flat_names = [name for names in obj_names_batch for name in names]
                phrase_emb_batch = torch.zeros((len(flat_names), text_hidden), device=device, dtype=torch.float32)
                
                valid_indices = [i for i, n in enumerate(flat_names) if isinstance(n, str) and len(n.strip()) > 0]
                if len(valid_indices) > 0:
                    valid_names = [flat_names[i] for i in valid_indices]
                    try:
                        tok_inputs = pipeline.tokenizer(
                            valid_names,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        with torch.no_grad():
                            tok_outputs = text_encoder(**tok_inputs)
                            attn_mask = tok_inputs.attention_mask.unsqueeze(-1)
                            # mean pooling
                            embs = (tok_outputs.last_hidden_state * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-6)
                            phrase_emb_batch[valid_indices] = embs.to(dtype=torch.float32)
                    except Exception as e:
                        print(f"⚠️ 警告: Phrase embedding 编码失败: {e}，将使用零向量 (step {global_step})")
                
                phrase_embeddings = phrase_emb_batch.view(bs, max_boxes, text_hidden).to(dtype=torch.float32)

            # --- C. 注入 Spatial Control ---
            bboxes = batch['bboxes'].to(device, dtype=torch.float32) # Adapter 期望 FP32 计算
            
            # 动态注入 (Adapter 保持 FP32)
            orig_procs, spatial_procs, adapter_container = inject_spatial_control_to_unet(
                unet, 
                adapter_dict=adapter_container, 
                bboxes=bboxes,
                phrase_embeddings=phrase_embeddings
            )
            
            # --- 延迟初始化优化器 ---
            if optimizer is None:
                params_to_optimize = [p for p in adapter_container.parameters() if p.requires_grad]
                if len(params_to_optimize) == 0:
                    for p in adapter_container.parameters(): 
                        p.requires_grad = True
                    params_to_optimize = adapter_container.parameters()
                
                optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, eps=1e-4, weight_decay=0.0)
                print(f"✓ 优化器初始化完成，参数量: {sum(p.numel() for p in params_to_optimize)}")

            # --- D. Forward (Autocast for UNet only) ---
            added_cond_kwargs = {
                "text_embeds": pooled_embeds, 
                "time_ids": _get_add_time_ids(bs, device)
            }
            
            # 开启 Autocast：UNet 使用 FP16，但 Adapter 保持 FP32
            # 注意：Adapter 在 UNet 内部被调用，但参数是 FP32，计算也会保持 FP32
            with torch.amp.autocast('cuda', dtype=torch.float16):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # 计算 Loss（转换为 FP32 以确保精度）
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # loss 数值检查，防止 NaN/Inf 进入 backward
            if not torch.isfinite(loss):
                print(f"⚠️ 警告: loss 非有限 (step {global_step})，跳过此 Batch")
                optimizer.zero_grad()
                remove_spatial_control_from_unet(unet, orig_procs)
                continue
            
            # --- E. Backward (直接 backward，不使用 Scaler) ---
            # 因为 Adapter 是 FP32，不需要混合精度训练
            # UNet 被冻结，只有 Adapter 需要梯度
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）并检查梯度是否有效
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter_container.parameters(), 0.5)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️ 警告: 检测到 NaN/Inf 梯度，跳过此 Batch (step {global_step})")
                optimizer.zero_grad()
                continue
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # --- F. 清理 ---
            remove_spatial_control_from_unet(unet, orig_procs)
            
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % 500 == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                torch.save(adapter_container.state_dict(), save_path)

    # 保存最终模型
    final_path = os.path.join(output_dir, "spatial_adapter_final.pt")
    torch.save(adapter_container.state_dict(), final_path)
    print(f"✅ 训练完成！模型已保存至: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-data", type=str, required=True, help="混合数据集 JSONL 路径")
    parser.add_argument("--kolors-model", type=str, default="./model/Kolors")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/spatial_adapter_mixed")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定 GPU 设备 (例如: cuda:0, cuda:1, cuda:2)。默认自动选择第一个可用 GPU"
    )
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"  # 默认使用第一个 GPU
    else:
        device = "cpu"
    
    print(f"🔧 使用设备: {device}")
    
    train_spatial_adapter(
        args.mixed_data,
        args.kolors_model,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.lr,
        device=device,
        image_dir=args.image_dir
    )
