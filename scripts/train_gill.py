import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from gill.gill_dataset import GILLDataset, collate_fn
from gill.spatial_adapter_fixed import (
    inject_spatial_control_to_unet,
    remove_spatial_control_from_unet,
    get_trainable_parameters
)
from diffusers import UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel
import argparse


def build_text_projection_if_needed(unet, text_encoder_dim, device='cuda'):
    """如果 UNet 需要特定 pooled text_dim，但文本编码器输出不匹配，返回一个线性投影模块。
    
    对于 SDXL/Kolors：time_ids 是 6 个分量，每个分量是 add_time_dim 维。
    因此 pooled text_dim = add_in - add_time_dim * 6
    """
    proj = None
    add_in = None
    if hasattr(unet, 'add_embedding') and hasattr(unet.add_embedding, 'linear_1'):
        add_in = unet.add_embedding.linear_1.in_features
    add_time_dim = getattr(unet.config, 'addition_time_embed_dim', None)
    if add_in is None or add_time_dim is None:
        return None, None
    
    # 修复：SDXL/Kolors 的 time_ids 是 6 个分量
    text_expected = add_in - add_time_dim * 6
    if text_expected != text_encoder_dim:
        # create projection
        proj = torch.nn.Linear(text_encoder_dim, text_expected).to(device)
        print(f"⚠️ Built text projection: {text_encoder_dim} -> {text_expected} (add_in={add_in}, time_dim={add_time_dim}, time_components=6)")
    return proj, text_expected


def train(args):
    device = 'cuda'
    # 加载本地 UNet
    unet = None
    model_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    for name in os.listdir(model_root):
        candidate = os.path.join(model_root, name)
        try:
            unet = UNet2DConditionModel.from_pretrained(candidate, subfolder='unet', torch_dtype=torch.float16, local_files_only=True).to(device)
            print('Loaded UNet from', candidate)
            break
        except Exception:
            continue
    if unet is None:
        raise RuntimeError('No local UNet found under model/')

    # prepare dataset (user must provide manifest path)
    manifest = getattr(args, 'manifest', None)
    dataset = None
    tokenizer = None
    text_encoder = None

    # build projection if needed (assume user knows text_encoder output dim)
    # for template, assume 2048
    text_encoder_dim = args.text_encoder_dim
    text_projection, text_expected = build_text_projection_if_needed(unet, text_encoder_dim)
    if text_projection is not None:
        print(f"Built text projection to map {text_encoder_dim} -> {text_expected}")

    dataloader = None
    if args.dry_run:
        # create fake tokenizer / encoder for dry-run
        class FakeTokenizer:
            def __init__(self, seq_len=77):
                self.seq_len = seq_len
            def __call__(self, texts, padding="max_length", max_length=256, truncation=True, return_tensors='pt'):
                batch = len(texts)
                input_ids = torch.zeros((batch, self.seq_len), dtype=torch.long)
                attention_mask = torch.ones((batch, self.seq_len), dtype=torch.long)
                position_ids = torch.arange(0, self.seq_len).unsqueeze(0).repeat(batch, 1)
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        class FakeTextEncoder:
            def __init__(self, seq_len=77, token_dim=2816, pooled_dim=4096):
                self.seq_len = seq_len
                self.token_dim = token_dim
                self.pooled_dim = pooled_dim
            def __call__(self, input_ids=None, attention_mask=None, position_ids=None, output_hidden_states=True):
                batch = input_ids.shape[0]
                h_seq = torch.randn(self.seq_len, batch, self.token_dim).to(device)
                h_pool = torch.randn(1, batch, self.pooled_dim).to(device)
                class Out: pass
                out = Out()
                out.hidden_states = [h_seq, h_pool]
                return out

        tokenizer = FakeTokenizer(seq_len=77)
        text_encoder = FakeTextEncoder(seq_len=77, token_dim=args.token_dim, pooled_dim=args.text_encoder_dim)

        # build projection if needed
        text_projection, text_expected = build_text_projection_if_needed(unet, args.text_encoder_dim, device=device)

        # build one fake batch (no file IO)
        bs = args.batch_size
        sample_batch = []
        for i in range(bs):
            pv = torch.randn(3, 512, 512).half().to(device)
            caption = f"dry run sample {i}"
            bboxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32)
            sample_batch.append({'pixel_values': pv, 'caption': caption, 'bboxes': bboxes})

        # collate into one batched dict
        batch = collate_fn(sample_batch, tokenizer=tokenizer, text_encoder=text_encoder, text_projection=text_projection, device=device, kolors=True)
        # ensure float16 where UNet expects fp16
        for k in ['encoder_hidden_states', 'pooled_embeds', 'text_embeds_proj']:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].half().to(device)
        # create an iterator that yields this batch once per epoch
        dataloader = [batch]
    else:
        if manifest is None:
            raise RuntimeError('manifest is required unless --dry_run is set')
        
        # 加载 tokenizer 和 text_encoder（从 model 目录中查找）
        print("📦 Loading tokenizer and text_encoder...")
        for name in os.listdir(model_root):
            candidate = os.path.join(model_root, name)
            try:
                # 尝试加载 tokenizer
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(candidate, subfolder='tokenizer', trust_remote_code=True, local_files_only=True)
                    print(f"  ✓ Loaded tokenizer from {candidate}")
                
                # 尝试加载 text_encoder
                if text_encoder is None:
                    text_encoder = AutoModel.from_pretrained(
                        candidate, 
                        subfolder='text_encoder', 
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=torch.float16
                    ).to(device)
                    text_encoder.requires_grad_(False)
                    print(f"  ✓ Loaded text_encoder from {candidate}")
                
                if tokenizer is not None and text_encoder is not None:
                    break
            except Exception as e:
                continue
        
        if tokenizer is None or text_encoder is None:
            raise RuntimeError('Failed to load tokenizer and text_encoder. Please ensure they exist in model/ directory.')
        
        # 重新构建 text_projection（使用实际的 text_encoder 维度）
        # 获取 text_encoder 的实际输出维度
        if hasattr(text_encoder, 'config'):
            # 尝试从 config 获取维度
            actual_dim = getattr(text_encoder.config, 'hidden_size', None) or getattr(text_encoder.config, 'd_model', None)
            if actual_dim is not None and actual_dim != text_encoder_dim:
                print(f"⚠️ Text encoder actual dim ({actual_dim}) != specified dim ({text_encoder_dim}). Using actual dim.")
                text_encoder_dim = actual_dim
                text_projection, text_expected = build_text_projection_if_needed(unet, text_encoder_dim, device=device)
        
        dataset = GILLDataset(manifest, args.image_root)
        # 注意：如果使用 Kolors 模型，kolors 应该设为 True
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer, text_encoder=text_encoder, text_projection=text_projection, device=device, kolors=True))

    # inject adapters
    # sample bboxes for injection: here use a single placeholder; production should compute per-sample
    bboxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]]).half().to(device)
    orig_proc, spatial_procs, adapter_dict = inject_spatial_control_to_unet(unet, bboxes=bboxes)

    # freeze unet
    for p in unet.parameters():
        p.requires_grad = False
    trainable = list(get_trainable_parameters(adapter_dict))
    for p in trainable:
        p.requires_grad = True
    
    # 如果创建了 text_projection，也需要加入训练
    if text_projection is not None:
        text_projection.train()
        for p in text_projection.parameters():
            p.requires_grad = True
        trainable.extend(list(text_projection.parameters()))
        print(f"✅ Added text_projection to trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"🔥 Total trainable parameters: {sum(p.numel() for p in trainable if p.requires_grad)}")

    # training loop
    unet.train()
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            # 确保所有数据在正确的设备上
            pixel_values = batch['pixel_values'].to(device)
            
            # compute text embeddings using tokenizer+encoder (collate produced either 'text_embeds_proj' or 'pooled_embeds')
            if 'text_embeds_proj' in batch:
                text_embeds = batch['text_embeds_proj'].to(device)
            elif 'pooled_embeds' in batch:
                pooled = batch['pooled_embeds'].to(device)
                if text_projection is not None:
                    text_embeds = text_projection(pooled)
                else:
                    text_embeds = pooled
            else:
                raise RuntimeError('Text embeddings not provided by collate_fn; integrate your tokenizer+text_encoder')

            # construct added_cond_kwargs
            # time_ids: SDXL/Kolors 需要 (batch, 6) 形状，6 个分量分别是：
            # [original_size_height, original_size_width, crop_coords_top, crop_coords_left, target_size_height, target_size_width]
            # 这里暂时设为 0，实际训练时应该根据图像尺寸设置
            batch_size = text_embeds.shape[0]
            time_ids = torch.zeros((batch_size, 6), dtype=torch.long).to(device)
            added_cond = {'text_embeds': text_embeds, 'time_ids': time_ids}

            # 更新 bboxes 到 spatial processors（每个 batch 使用真实的 bboxes）
            batch_bboxes = batch.get('bboxes', None)
            if batch_bboxes is not None:
                # 处理 bboxes：如果是 list，需要 padding 到相同长度
                if isinstance(batch_bboxes, list):
                    max_obj = max([b.shape[0] for b in batch_bboxes]) if batch_bboxes else 1
                    if max_obj == 0:
                        max_obj = 1
                    padded_bboxes = []
                    for b in batch_bboxes:
                        b = b.to(device)
                        pad_len = max_obj - b.shape[0]
                        if pad_len > 0:
                            pad = torch.zeros((pad_len, 4), device=device, dtype=b.dtype)
                            padded_bboxes.append(torch.cat([b, pad]))
                        else:
                            padded_bboxes.append(b)
                    batch_bboxes = torch.stack(padded_bboxes).to(dtype=torch.float16)  # (B, N, 4)
                else:
                    batch_bboxes = batch_bboxes.to(device).to(dtype=torch.float16)
                
                # 更新到所有 spatial processors
                for proc in spatial_procs.values():
                    if hasattr(proc, 'set_spatial_control'):
                        proc.set_spatial_control(batch_bboxes)
            
            # sample noise and timestep for denoising step
            # 标准扩散模型训练：预测添加到 latents 的噪声
            noise = torch.randn(pixel_values.shape[0], 4, 64, 64, device=device, dtype=torch.float16)
            latents = noise  # 简化：直接使用噪声作为输入（实际应该用 VAE 编码的 latents）
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device).long()

            # provide encoder_hidden_states if collate produced them
            encoder_hidden_states = batch.get('encoder_hidden_states', None)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(device)
            out = unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond)
            pred = getattr(out, 'sample', out)
            
            # 标准扩散 loss：预测噪声与真实噪声的 MSE
            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            
            optimizer.step()
            
            # 记录 loss
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 每 10 个 batch 打印一次
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Step {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")

        # checkpoint (save adapter state dicts and text_projection if exists)
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"✅ Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss': avg_epoch_loss,
            'adapters': {k: v.state_dict() if hasattr(v, 'state_dict') else v.cpu() for k, v in adapter_dict.items()}
        }
        if text_projection is not None:
            checkpoint['text_projection'] = text_projection.state_dict()
        
        save_path = os.path.join(args.output_dir, f'adapters_epoch_{epoch+1}.pt')
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint to {save_path}")

    # restore
    remove_spatial_control_from_unet(unet, orig_proc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=False, help='Path to manifest jsonl (not required with --dry_run)')
    p.add_argument('--image_root', required=False, help='Image root folder (not required with --dry_run)')
    p.add_argument('--dry_run', action='store_true', help='Run a single-batch dry run with fake tokenizer/encoder')
    p.add_argument('--token_dim', type=int, default=2816, help='Token dimension for fake text encoder (dry run)')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--output_dir', default='runs')
    p.add_argument('--text_encoder_dim', type=int, default=4096, help='Text encoder output dimension (default: 4096 for ChatGLM/Kolors)')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--model_path', type=str, default=None, help='Explicit model path (optional, will auto-detect from model/ if not provided)')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
