import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gill.gill_dataset import GILLDataset, collate_fn
from gill.spatial_adapter_fixed import (
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    get_trainable_parameters
)

logger = get_logger(__name__)

def build_text_projection_if_needed(unet, text_encoder_dim, device):
    """构建文本特征投影层，用于对齐 Text Encoder 和 UNet 的维度"""
    # 1. 检测 UNet 期望的 Token Embedding 维度 (Cross-Attention 输入)
    # 优先检查 encoder_hid_proj (SDXL/Kolors 特征)
    desired_token_dim = None
    if hasattr(unet, 'encoder_hid_proj'):
        desired_token_dim = unet.encoder_hid_proj.in_features
    
    # 2. 检测 UNet 期望的 Pooled Embedding 维度 (Add Embedding 输入)
    # Kolors/SDXL 使用 add_embedding 加入 time_ids 和 pooled_text_embeds
    desired_pooled_dim = None
    add_time_dim = getattr(unet.config, 'addition_time_embed_dim', 0)
    if hasattr(unet, 'add_embedding') and hasattr(unet.add_embedding, 'linear_1'):
        total_add_in = unet.add_embedding.linear_1.in_features

        # 修复逻辑：SDXL/Kolors 的 time embedding 部分通常是 6 个分量
        # 因此应扣除 6 * add_time_dim
        if add_time_dim > 0:
            expected_time_part = add_time_dim * 6
            desired_pooled_dim = total_add_in - expected_time_part
            logger.info(f"🔍 Detected SDXL-like config: total_add_in={total_add_in}, time_part={expected_time_part} (6x{add_time_dim}), expected_pooled={desired_pooled_dim}")
        else:
            desired_pooled_dim = total_add_in

    projections = nn.ModuleDict()
    
    # 构建 Token 投影 (Seq -> Seq)
    if desired_token_dim is not None and desired_token_dim != text_encoder_dim:
        logger.info(f"⚠️ Mismatch: Token dim {text_encoder_dim} != UNet expected {desired_token_dim}. Creating projection.")
        projections['token_proj'] = nn.Linear(text_encoder_dim, desired_token_dim)
    
    # 构建 Pooled 投影 (Vec -> Vec)
    if desired_pooled_dim is not None and desired_pooled_dim != text_encoder_dim:
         logger.info(f"⚠️ Mismatch: Pooled dim {text_encoder_dim} != UNet expected {desired_pooled_dim}. Creating projection.")
         projections['pooled_proj'] = nn.Linear(text_encoder_dim, desired_pooled_dim)

    return projections.to(device)

def train(args):
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(42)

    device = accelerator.device

    # 2. 加载 UNet
    logger.info(f"📦 Loading UNet from {args.model_path}...")
    try:
        unet = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder='unet',
            local_files_only=True
        )
        # 冻结 UNet
        unet.requires_grad_(False)
    except Exception as e:
        logger.error(f"❌ Failed to load UNet: {e}")
        return

    # 启用 Gradient Checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 3. 加载 Text Encoder (仅主进程加载以节省内存，或全部加载)
    tokenizer = None
    text_encoder = None
    if not args.dry_run:
        logger.info(f"📦 Loading Text Encoder from {args.model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder='tokenizer', trust_remote_code=True)
            text_encoder = AutoModel.from_pretrained(
                args.model_path, 
                subfolder='text_encoder', 
                trust_remote_code=True
            )
            text_encoder.requires_grad_(False)
        except Exception as e:
            logger.error(f"❌ Failed to load Text Encoder: {e}")
            return

    # 4. 注入 Spatial Adapter
    logger.info("💉 Injecting Spatial Adapter...")
    # 移动到 device 以便正确初始化
    unet.to(device)
    dummy_bboxes = torch.zeros((1, 1, 4), device=device)
    
    # 注入 Adapter
    orig_proc, spatial_procs, adapter_dict = inject_spatial_control_to_unet(unet, bboxes=dummy_bboxes)
    
    # 激活 Adapter 梯度
    trainable_params = list(get_trainable_parameters(adapter_dict))
    for p in trainable_params:
        p.requires_grad = True
    
    # 5. 构建投影层 (如果需要)
    projections = build_text_projection_if_needed(unet, args.text_encoder_dim, device)
    if len(projections) > 0:
        projections.train()
        for p in projections.parameters():
            p.requires_grad = True
        trainable_params.extend(list(projections.parameters()))
        
    logger.info(f"🔥 Total trainable parameters: {len(trainable_params)} tensors")

    # 6. 优化器与数据准备
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    if args.dry_run:
        logger.info("⚠️ DRY RUN MODE")
        # Fake Tokenizer/Encoder for dry run
        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                batch = len(texts)
                seq = 77
                return {
                    'input_ids': torch.zeros((batch, seq), dtype=torch.long),
                    'attention_mask': torch.ones((batch, seq), dtype=torch.long),
                    'position_ids': torch.arange(seq).unsqueeze(0).repeat(batch, 1)
                }
        import types
        class FakeTextEncoder:
            def __call__(self, **kwargs):
                batch = kwargs.get('input_ids').shape[0]
                seq = 77
                # 返回 (seq, batch, dim) 格式
                h_seq = torch.randn(seq, batch, 2816).to(device)
                h_pool = torch.randn(batch, 4096).to(device)
                return types.SimpleNamespace(hidden_states=[h_seq, h_pool])

        tokenizer = FakeTokenizer()
        text_encoder = FakeTextEncoder()
        
        # Fake Batch
        sample_batch = []
        for i in range(args.batch_size):
            pv = torch.randn(3, 512, 512)
            bboxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
            sample_batch.append({'pixel_values': pv, 'caption': "dry run", 'bboxes': bboxes})
        
        # 手动 collate
        batch = collate_fn(sample_batch, tokenizer, text_encoder, device=device, kolors=True)
        dataloader = [batch] # List pretending to be DataLoader
        
        # Dry run 不使用 accelerate prepare dataloader，直接循环
        adapter_dict, optimizer = accelerator.prepare(adapter_dict, optimizer)
    else:
        dataset = GILLDataset(args.manifest, args.image_root)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda b: collate_fn(b, tokenizer, text_encoder, device=device, kolors=True)
        )
        # Prepare everything
        adapter_dict, projections, optimizer, dataloader = accelerator.prepare(
            adapter_dict, projections, optimizer, dataloader
        )

    # 7. 训练循环
    unet.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        
        for batch in dataloader:
            with accelerator.accumulate(adapter_dict): # 梯度累积上下文
                
                # --- 数据转换 ---
                # Accelerate 会自动处理 device，但 collate_fn 可能已经 .to(device) 了
                # 只需要确保 dtype 正确
                pixel_values = batch['pixel_values'].to(dtype=unet.dtype)
                
                # 获取 Text Embeddings
                encoder_hidden_states = batch.get('encoder_hidden_states') # (B, Seq, Dim)
                pooled_embeds = batch.get('pooled_embeds')       # (B, Dim)
                
                if encoder_hidden_states is not None: 
                    encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype)
                if pooled_embeds is not None:
                    pooled_embeds = pooled_embeds.to(dtype=unet.dtype)

                # --- 应用投影 (如果存在) ---
                if 'token_proj' in projections:
                    encoder_hidden_states = projections['token_proj'](encoder_hidden_states)
                else:
                    # dynamic token projection: if UNet expects a different token dim, create temporary proj
                    try:
                        desired_tok = getattr(unet, 'encoder_hid_proj').in_features
                    except Exception:
                        desired_tok = None
                    if desired_tok is not None and encoder_hidden_states is not None and encoder_hidden_states.shape[-1] != desired_tok:
                        in_dim = encoder_hidden_states.shape[-1]
                        tmp_tok = nn.Linear(in_dim, desired_tok).to(encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                        # apply across sequence: Linear will broadcast over leading dims
                        encoder_hidden_states = tmp_tok(encoder_hidden_states)
                        logger.warning(f"Dynamic token_proj created: {in_dim} -> {desired_tok}")
                
                if 'pooled_proj' in projections:
                    try:
                        pooled_embeds = projections['pooled_proj'](pooled_embeds)
                    except RuntimeError:
                        # 动态适配：如果 pooled_embeds 的维度与 projections 不匹配，临时创建一个投影层
                        in_dim = pooled_embeds.shape[-1]
                        out_dim = projections['pooled_proj'].out_features
                        tmp_proj = nn.Linear(in_dim, out_dim).to(pooled_embeds.device, dtype=pooled_embeds.dtype)
                        pooled_embeds = tmp_proj(pooled_embeds)
                        logger.warning(f"Dynamic pooled_proj created: {in_dim} -> {out_dim}")

                # --- 构建 Condition ---
                time_ids = torch.zeros((pixel_values.shape[0], 6), device=pixel_values.device, dtype=torch.long)
                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}
                
                latents = torch.randn((pixel_values.shape[0], 4, 64, 64), device=pixel_values.device, dtype=unet.dtype)
                timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=pixel_values.device).long()
                
                # --- 处理 BBoxes ---
                bboxes_raw = batch['bboxes']
                if isinstance(bboxes_raw, list):
                    max_obj = max([b.shape[0] for b in bboxes_raw]) if bboxes_raw else 0
                    if max_obj == 0: max_obj = 1
                    padded = []
                    for b in bboxes_raw:
                        b = b.to(device)
                        pad_len = max_obj - b.shape[0]
                        if pad_len > 0:
                            pad = torch.zeros((pad_len, 4), device=device, dtype=b.dtype)
                            padded.append(torch.cat([b, pad]))
                        else:
                            padded.append(b)
                    bboxes = torch.stack(padded).to(dtype=unet.dtype)
                else:
                    bboxes = bboxes_raw.to(dtype=unet.dtype)

                # --- 注入 Spatial Control ---
                for proc in spatial_procs.values():
                    if hasattr(proc, 'set_spatial_control'):
                        proc.set_spatial_control(bboxes)
                
                # --- Forward ---
                noise = torch.randn_like(latents)
                try:
                    model_pred = unet(
                        latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                except Exception as e:
                    # 输出诊断信息以便调试维度不匹配问题
                    try:
                        logger.error(f"UNet add_embedding.in_features={getattr(unet.add_embedding, 'linear_1').in_features}")
                    except Exception:
                        pass
                    logger.error(f"encoder_hidden_states shape: {None if encoder_hidden_states is None else tuple(encoder_hidden_states.shape)}")
                    logger.error(f"pooled_embeds shape: {None if pooled_embeds is None else tuple(pooled_embeds.shape)}")
                    logger.error(f"time_ids shape: {time_ids.shape}")
                    raise
                
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                
                # --- Backward ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

        # --- 保存 ---
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f'adapter_epoch_{epoch}.pt')
            try:
                unwrapped_adapter = accelerator.unwrap_model(adapter_dict)
                torch.save(unwrapped_adapter.state_dict(), save_path)
                logger.info(f"✅ Saved adapter to {save_path}")
                if len(projections) > 0:
                    proj_path = os.path.join(args.output_dir, f'projections_epoch_{epoch}.pt')
                    torch.save(accelerator.unwrap_model(projections).state_dict(), proj_path)
            except Exception as e:
                logger.error(f"⚠️ Failed saving: {e}")
                
    # 恢复
    if accelerator.is_main_process:
        remove_spatial_control_from_unet(unet, orig_proc)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str)
    p.add_argument('--image_root', type=str)
    p.add_argument('--model_path', type=str, default='./model/Kolors')
    p.add_argument('--output_dir', type=str, default='./runs')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--gradient_checkpointing', action='store_true', default=True)
    p.add_argument('--text_encoder_dim', type=int, default=4096)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--mixed_precision', type=str, default="fp16", choices=["no", "fp16", "bf16"])
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
