#!/usr/bin/env python3
"""
训练 Spatial Adapter（适配 Kolors/SDXL）- 稳健修复版

核心修复：
1. Phrase Embedding 对齐：优先使用 attention_mask 进行 masked mean pooling，避免 padding 污染。
2. 空数据过滤：增强 Dataset 和 DataLoader 的鲁棒性。
3. 显存优化：保留 VAE FP32 + CPU Offload。
"""

import argparse
import os
import sys
import json
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import deque, Counter

# 尝试导入 diffusers 组件
try:
    from diffusers import KolorsPipeline, DDPMScheduler
    from diffusers.optimization import get_scheduler  # noqa: F401
except ImportError:
    print("❌ 未安装 diffusers，请运行: pip install diffusers accelerate")
    sys.exit(1)

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gill.spatial_adapter import (
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    create_spatial_adapter_for_kolors,
    SpatialAdapterModuleDict,
)

_MEASURE_RE = re.compile(r"^(一|二|三|四|五|六|七|八|九|十|两|几|多|每)?(个|只|条|张|把|台|部|辆|块|片|件|根|位|名|对|双|群)")
_NOISE_RE = re.compile(r"(正在|位于|看着|站在|坐在|躺在|趴在|穿着|拿着|走在|骑着)")


def clean_object_name(name: str, max_len: int = 10, min_len: int = 1) -> str:
    """在线清洗物体名：保留核心名词，过滤明显噪声。"""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""

    # 去标签/符号/标点
    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"[\"'“”‘’（）()《》【】\[\]{}<>]", "", name)
    name = re.sub(r"[，,。\.、;；:：!?！？~`·•]", "", name)
    name = re.sub(r"\s+", "", name)
    if not name:
        return ""

    # 去数量词前缀：一只/两张/三辆…
    name = _MEASURE_RE.sub("", name)
    if not name:
        return ""

    # 去“的”修饰，只保留中心语
    if "的" in name:
        parts = [p for p in name.split("的") if p]
        if parts:
            name = parts[-1]

    # 基本长度与噪声词过滤
    if len(name) < min_len or len(name) > max_len:
        return ""
    if _NOISE_RE.search(name):
        return ""
    return name


class MixedLayoutDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_dir: str = None,
        resolution: int = 1024,
        name_max_len: int = 10,
        name_min_len: int = 1,
        min_name_freq: int = 1,
        enable_name_clean: bool = True,
    ):
        self.samples = []
        self.image_dir = image_dir
        self.resolution = resolution
        self.name_max_len = int(name_max_len)
        self.name_min_len = int(name_min_len)
        self.min_name_freq = int(min_name_freq)
        self.enable_name_clean = bool(enable_name_clean)
        self.name_freq = Counter()

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")

        print(f"📖 读取数据: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # 必须包含 caption
                        if "caption" in item:
                            self.samples.append(item)
                    except Exception:
                        continue
        # 统计清洗后词频（可选）
        if self.enable_name_clean and self.min_name_freq > 1:
            for item in self.samples:
                for obj in item.get("objects", []) or []:
                    raw = str(obj.get("name", ""))
                    cleaned = clean_object_name(raw, self.name_max_len, self.name_min_len)
                    if cleaned:
                        self.name_freq[cleaned] += 1
        print(f"✓ 加载 {len(self.samples)} 条数据")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. 加载图像
        image_path = item.get("image_path", "")
        if self.image_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_dir, image_path)

        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image = image.resize((self.resolution, self.resolution))
                pixel_values = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
                pixel_values = pixel_values.permute(2, 0, 1)  # [3, H, W]
                pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            else:
                # Dummy image
                pixel_values = torch.zeros(3, self.resolution, self.resolution)
        except Exception:
            pixel_values = torch.zeros(3, self.resolution, self.resolution)

        # 2. 处理 BBox
        objects = item.get("objects", [])
        bboxes = []
        obj_names = []
        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # 归一化检查
            if max(x1, y1, x2, y2) > 1.5:
                bbox = [x / 1000.0 for x in bbox]
                x1, y1, x2, y2 = bbox

            # 几何过滤
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            # 与过滤脚本保持一致（min-area=0.02, min-side=0.03）
            if 0.02 < area < 0.95 and w > 0.03 and h > 0.03:
                raw_name = str(obj.get("name", ""))
                if self.enable_name_clean:
                    clean_name = clean_object_name(raw_name, self.name_max_len, self.name_min_len)
                    if self.min_name_freq > 1 and clean_name:
                        if self.name_freq.get(clean_name, 0) < self.min_name_freq:
                            clean_name = ""
                else:
                    clean_name = raw_name.strip()
                bboxes.append([x1, y1, x2, y2])
                obj_names.append(clean_name)

        return {
            "pixel_values": pixel_values,
            "caption": item.get("caption", ""),
            "bboxes": bboxes,
            "obj_names": obj_names,
            "has_layout": len(bboxes) > 0,
        }


def collate_mixed_batch(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]

    # 找出 batch 中最多的物体数量
    max_boxes = max([len(item["bboxes"]) for item in batch] + [0])
    max_boxes = max(max_boxes, 1)  # 至少留 1 个位置防止 shape 错误

    bboxes_padded = []
    masks = []
    obj_names_batched = []

    for item in batch:
        boxes = item["bboxes"]
        names = item.get("obj_names", [])
        num_boxes = len(boxes)

        # Padding
        padded_boxes = boxes + [[0.0] * 4] * (max_boxes - num_boxes)
        padded_masks = [1] * num_boxes + [0] * (max_boxes - num_boxes)
        padded_names = names + [""] * (max_boxes - num_boxes)

        bboxes_padded.append(padded_boxes)
        masks.append(padded_masks)
        obj_names_batched.append(padded_names)

    bboxes_tensor = torch.tensor(bboxes_padded, dtype=torch.float32)
    masks_tensor = torch.tensor(masks, dtype=torch.float32)

    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "bboxes": bboxes_tensor,
        "masks": masks_tensor,
        "obj_names": obj_names_batched,
    }




def _pool_hidden_states(hidden, attention_mask=None):
    # Pool hidden states into a single vector per sample.
    # Returns shape [H] for 2D input or [B, H] for 3D input.
    if hidden is None:
        return None
    if hidden.dim() == 2:
        # [L, H]
        if attention_mask is not None and attention_mask.numel() == hidden.shape[0]:
            mask = attention_mask.reshape(-1, 1).float()
            return (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1e-9)
        return hidden.mean(dim=0)
    if hidden.dim() == 3:
        # [B, L, H]
        if attention_mask is not None and attention_mask.shape[:2] == hidden.shape[:2]:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden.mean(dim=1)
    # Fallback: flatten
    return hidden.reshape(-1, hidden.shape[-1]).mean(dim=0)


def _maybe_transpose_hidden(hidden, expected_bs: int):
    # ChatGLM-style output can be (seq_len, batch, hidden). Align to (batch, seq_len, hidden).
    if hidden is None or hidden.dim() != 3:
        return hidden
    if hidden.shape[0] != expected_bs and hidden.shape[1] == expected_bs:
        return hidden.transpose(0, 1)
    return hidden
def _get_add_time_ids(bs, device, original_size=(1024, 1024), target_size=(1024, 1024), crops_coords_top_left=(0, 0)):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.long, device=device)
    return add_time_ids.repeat(bs, 1)


def _sync_module_params(module: torch.nn.Module):
    """多卡时同步 Rank0 参数，避免初始化不一致。"""
    if not dist.is_initialized():
        return
    dist.barrier()
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buf in module.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()


def train_spatial_adapter(
    mixed_data_path: str,
    kolors_model_path: str,
    output_dir: str,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda:0",
    image_dir: str = None,
    scale_min: float = 0.0,
    scale_max: float = 1.0,
    phrase_dropout: float = 0.0,
    save_every: int = 2000,
    name_max_len: int = 10,
    name_min_len: int = 1,
    min_name_freq: int = 1,
    enable_name_clean: bool = True,
):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = local_rank >= 0
    rank = 0
    world_size = 1
    if distributed:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    is_main = rank == 0
    if is_main:
        print("Init Kolors Spatial Adapter training (Mixed Precision)...")


    try:
        pipeline = KolorsPipeline.from_pretrained(
            kolors_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        # Tokenizer Padding Fix
        if hasattr(pipeline, "tokenizer") and pipeline.tokenizer is not None:
            if hasattr(pipeline.tokenizer, "_pad"):
                original_pad = pipeline.tokenizer._pad

                def compatible_pad(encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None, **kwargs):
                    kwargs.pop("padding_side", None)
                    return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask, **kwargs)

                pipeline.tokenizer._pad = compatible_pad
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # FP32 VAE & Freeze
    vae.to(dtype=torch.float32)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Checkpointing
    unet.train()
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # CPU Offload
    vae.to("cpu")
    text_encoder.to(device)
    unet.to(device)

    # 2. Adapter
    adapter_container = create_spatial_adapter_for_kolors()
    adapter_container.to(device, dtype=torch.float32)

    # 2.1 预构建 Adapter（避免参数为空导致 optimizer 失败）
    try:
        dummy_bboxes = torch.zeros((1, 1, 4), device=device, dtype=torch.float32)
        dummy_masks = torch.zeros((1, 1), device=device, dtype=torch.float32)
        orig_procs, _, adapter_container = inject_spatial_control_to_unet(
            unet,
            adapter_dict=adapter_container,
            bboxes=dummy_bboxes,
            phrase_embeddings=None,
            masks=dummy_masks,
            adapter_dtype=torch.float32,
        )
        remove_spatial_control_from_unet(unet, orig_procs)
    except Exception as e:
        print(f"⚠️ Adapter warm-up failed: {e}")

    # 多卡：同步初始化权重，确保各 rank 起点一致
    if distributed and dist.is_initialized():
        _sync_module_params(adapter_container)
        if is_main:
            print("✅ Synced adapter params from rank0.")

    # 多卡：用 DDP 包装 UNet，并把 adapter 挂到 UNet 上让 DDP 跟踪参数
    if distributed:
        unet.adapter_container = adapter_container
        unet = DDP(
            unet,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if is_main:
            print("✅ Wrapped UNet with DDP (tracking adapter params).")

    # 3. Optimizer
    params_to_optimize = [p for p in adapter_container.parameters() if p.requires_grad]
    if not params_to_optimize:
        for p in adapter_container.parameters():
            p.requires_grad = True
        params_to_optimize = list(adapter_container.parameters())
    if not params_to_optimize:
        raise RuntimeError("Adapter params still empty after warm-up; check adapter injection.")

    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, eps=1e-4, weight_decay=0.0)

    # 4. Data
    dataset = MixedLayoutDataset(
        mixed_data_path,
        image_dir=image_dir,
        name_max_len=name_max_len,
        name_min_len=name_min_len,
        min_name_freq=min_name_freq,
        enable_name_clean=enable_name_clean,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_mixed_batch,
        num_workers=0,
    )

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    loss_window = deque(maxlen=100)
    phrase_total = 0
    phrase_fail = 0

    scale_min = float(scale_min)
    scale_max = float(scale_max)
    phrase_dropout = float(phrase_dropout)
    save_every = max(int(save_every), 1)

    for epoch in range(epochs):
        unet.train()
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not is_main)

        for batch in progress_bar:
            if batch is None:
                continue

            # --- A/B/C. Latents + Text + Phrase Embeds ---
            skip_step = False
            with torch.no_grad():
                # VAE to GPU for encoding
                vae.to(device)
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                try:
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.to(dtype=torch.float16)
                except Exception:
                    skip_step = True
                finally:
                    vae.to("cpu")  # Back to CPU
                    torch.cuda.empty_cache()

                if not skip_step:
                    noise = torch.randn_like(latents)
                    bs = latents.shape[0]
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                    # --- B. Text Embeds ---
                    try:
                        encoded = pipeline.encode_prompt(
                            prompt=batch["captions"],
                            device=device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        prompt_embeds, pooled_embeds = encoded[0], encoded[2]
                    except Exception:
                        skip_step = True

                if not skip_step:
                    # --- C. Phrase Embeds (Robust) ---
                    obj_names_batch = batch.get("obj_names", [])
                    max_boxes = batch["bboxes"].shape[1]
                    text_hidden = getattr(text_encoder.config, "hidden_size", 4096)

                    # Initialize with Zeros
                    phrase_emb_batch = torch.zeros((bs * max_boxes, text_hidden), device=device, dtype=torch.float32)

                    # Flatten
                    flat_names = [name for names in obj_names_batch for name in names]
                    # Only valid non-empty strings
                    valid_indices = [i for i, n in enumerate(flat_names) if isinstance(n, str) and n.strip()]

                    phrase_total += 1
                    if valid_indices:
                        valid_names = [flat_names[i] for i in valid_indices]
                        try:
                            # Tokenize
                            tok_inputs = pipeline.tokenizer(
                                valid_names,
                                padding=True,
                                truncation=True,
                                max_length=32,
                                return_tensors="pt",
                            ).to(device)

                            # Encode without attention_mask to avoid internal mismatch
                            tok_outputs = text_encoder(input_ids=tok_inputs.input_ids, output_hidden_states=True)
                            last_hidden = tok_outputs.last_hidden_state
                            last_hidden = _maybe_transpose_hidden(last_hidden, tok_inputs.input_ids.shape[0])

                            # If batch mismatch, fallback to per-name encoding
                            if last_hidden.shape[0] != tok_inputs.input_ids.shape[0]:
                                # Per-name fallback with direct assignment (avoid stack shape mismatch).
                                for idx, name in zip(valid_indices, valid_names):
                                    single = pipeline.tokenizer(
                                        name,
                                        padding=True,
                                        truncation=True,
                                        max_length=32,
                                        return_tensors="pt",
                                    ).to(device)
                                    out = text_encoder(input_ids=single.input_ids, output_hidden_states=True)
                                    lh = out.last_hidden_state
                                    lh = _maybe_transpose_hidden(lh, single.input_ids.shape[0])
                                    emb = _pool_hidden_states(
                                        lh,
                                        single.attention_mask if hasattr(single, "attention_mask") else None,
                                    )
                                    if emb is not None:
                                        if emb.dim() == 2:
                                            emb = emb.mean(dim=0) if emb.shape[0] > 1 else emb.squeeze(0)
                                        elif emb.dim() > 2:
                                            emb = emb.reshape(-1, emb.shape[-1]).mean(dim=0)
                                        phrase_emb_batch[idx] = emb.to(dtype=torch.float32)
                                embs = None
                            else:
                                # Pooling with mask if safe
                                mask = tok_inputs.attention_mask if hasattr(tok_inputs, "attention_mask") else None
                                if mask is not None and mask.shape[:2] != last_hidden.shape[:2]:
                                    mask = None
                                embs = _pool_hidden_states(last_hidden, mask)

                            # Ensure [B, H]
                            if embs is not None and embs.dim() == 1:
                                embs = embs.unsqueeze(0)

                            # Safe assignment to avoid size mismatch
                            if embs is not None:
                                safe_count = min(embs.shape[0], len(valid_indices))
                                if safe_count > 0:
                                    phrase_emb_batch[valid_indices[:safe_count]] = embs[:safe_count].to(dtype=torch.float32)

                        except Exception as e:
                            phrase_fail += 1
                            if global_step % 100 == 0:
                                print(f"WARN: Phrase encode failed: {e}")

                    phrase_embeddings = phrase_emb_batch.view(bs, max_boxes, text_hidden)

                    # Phrase embedding dropout: encourage bbox-only robustness
                    if phrase_dropout > 0.0 and phrase_embeddings is not None:
                        drop_mask = (torch.rand((bs, max_boxes), device=device) > phrase_dropout).float()
                        phrase_embeddings = phrase_embeddings * drop_mask.unsqueeze(-1)

            # 多卡同步跳过标志，避免 rank 间步数不一致导致 DDP 卡死
            if distributed and dist.is_initialized():
                skip_tensor = torch.tensor(1 if skip_step else 0, device=device)
                dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                if skip_tensor.item() > 0:
                    continue
            else:
                if skip_step:
                    continue

            # --- D. Injection ---
            bboxes = batch["bboxes"].to(device, dtype=torch.float32)
            masks = batch.get("masks").to(device, dtype=torch.float32) if batch.get("masks") is not None else None

            orig_procs, spatial_procs, adapter_container = inject_spatial_control_to_unet(
                unet,
                adapter_dict=adapter_container,
                bboxes=bboxes,
                phrase_embeddings=phrase_embeddings,
                masks=masks,
                adapter_dtype=torch.float32,
            )
            if hasattr(adapter_container, "set_scale"):
                if scale_max < scale_min:
                    scale_max, scale_min = scale_min, scale_max
                scale = float(torch.empty(1, device=device).uniform_(scale_min, scale_max).item())
                adapter_container.set_scale(scale)

            # --- E. Forward & Backward ---
            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": _get_add_time_ids(bs, device)}

            with torch.amp.autocast("cuda", dtype=torch.float16):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_container.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    loss_window.append(loss.item())
                    avg_loss = sum(loss_window) / max(len(loss_window), 1)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg100": f"{avg_loss:.4f}"})

                    if is_main and global_step % 100 == 0:
                        fail_rate = (phrase_fail / phrase_total) * 100 if phrase_total > 0 else 0.0
                        if fail_rate > 10.0:
                            print(f"WARN: Phrase Embedding fail rate high ({fail_rate:.2f}%)")
                        else:
                            print(f"[Stats] Step {global_step}: Phrase Fail Rate = {fail_rate:.2f}%")
                        print(f"[Stats] Step {global_step}: Avg Loss (last 100) = {avg_loss:.4f}")

                if is_main and global_step % save_every == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(adapter_container.state_dict(), save_path)

            # Clean up hooks
            remove_spatial_control_from_unet(unet, orig_procs)

    # Save final
    if distributed and dist.is_initialized():
        dist.barrier()
    if is_main:
        final_path = os.path.join(output_dir, "spatial_adapter_final.pt")
        torch.save(adapter_container.state_dict(), final_path)
        print(f"Training done. Model saved to: {final_path}")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-data", type=str, required=True)
    parser.add_argument("--kolors-model", type=str, default="./model/Kolors")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/spatial_adapter_mixed")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scale-min", type=float, default=0.0, help="Min adapter scale during training")
    parser.add_argument("--scale-max", type=float, default=1.0, help="Max adapter scale during training")
    parser.add_argument("--phrase-dropout", type=float, default=0.1, help="Dropout prob for phrase embeddings")
    parser.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--name-max-len", type=int, default=10, help="Max length of cleaned object name")
    parser.add_argument("--name-min-len", type=int, default=1, help="Min length of cleaned object name")
    parser.add_argument("--min-name-freq", type=int, default=1, help="Min frequency to keep object name (after cleaning)")
    parser.add_argument("--disable-name-clean", action="store_true", help="Disable on-the-fly object name cleaning")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    train_spatial_adapter(
        args.mixed_data,
        args.kolors_model,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.lr,
        device=device,
        image_dir=args.image_dir,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        phrase_dropout=args.phrase_dropout,
        save_every=args.save_every,
        name_max_len=args.name_max_len,
        name_min_len=args.name_min_len,
        min_name_freq=args.min_name_freq,
        enable_name_clean=not args.disable_name_clean,
    )
