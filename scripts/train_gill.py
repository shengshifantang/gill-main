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
import argparse


def build_text_projection_if_needed(unet, text_encoder_dim, device='cuda'):
    """如果 UNet 需要特定 pooled text_dim，但文本编码器输出不匹配，返回一个线性投影模块。"""
    proj = None
    add_in = None
    if hasattr(unet, 'add_embedding') and hasattr(unet.add_embedding, 'linear_1'):
        add_in = unet.add_embedding.linear_1.in_features
    add_time_dim = getattr(unet.config, 'addition_time_embed_dim', None)
    if add_in is None or add_time_dim is None:
        return None, None
    text_expected = add_in - add_time_dim
    if text_expected != text_encoder_dim:
        # create projection
        proj = torch.nn.Linear(text_encoder_dim, text_expected).to(device)
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
    # placeholders: user should supply tokenizer and text_encoder
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
        dataset = GILLDataset(manifest, args.image_root)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer, text_encoder=text_encoder, text_projection=text_projection, device=device, kolors=False))

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

    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # training loop (skeleton)
    unet.train()
    for epoch in range(args.epochs):
        for batch in dataloader:
            # when dry_run, batch is already a collated dict on device
            pixel_values = batch['pixel_values'] if not args.dry_run else batch['pixel_values'].to(device)
            # compute text embeddings using tokenizer+encoder (collate produced either 'text_embeds_proj' or 'pooled_embeds')
            if 'text_embeds_proj' in batch:
                text_embeds = batch['text_embeds_proj'] if not args.dry_run else batch['text_embeds_proj'].to(device)
            elif 'pooled_embeds' in batch:
                pooled = batch['pooled_embeds'] if not args.dry_run else batch['pooled_embeds'].to(device)
                if text_projection is not None:
                    text_embeds = text_projection(pooled)
                else:
                    text_embeds = pooled
            else:
                raise RuntimeError('Text embeddings not provided by collate_fn; integrate your tokenizer+text_encoder')

            # construct added_cond_kwargs
            # time_ids currently set to zeros; replace with your deterministic time id logic
            time_ids = torch.zeros(text_embeds.shape[0], dtype=torch.long).to(device)
            added_cond = {'text_embeds': text_embeds, 'time_ids': time_ids}

            # sample noise and timestep for denoising step (user should adapt to diffusion loss)
            latents = torch.randn(pixel_values.shape[0], 4, 64, 64).half().to(device)
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)

            # provide encoder_hidden_states if collate produced them
            encoder_hidden_states = batch.get('encoder_hidden_states', None) if not args.dry_run else batch.get('encoder_hidden_states', None)
            out = unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond)
            pred = getattr(out, 'sample', out)
            loss = pred.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # checkpoint (only save adapter state dicts)
        torch.save({'epoch': epoch, 'adapters': {k: v.cpu() for k, v in adapter_dict.items()}}, os.path.join(args.output_dir, f'adapters_epoch_{epoch}.pt'))

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
    p.add_argument('--text_encoder_dim', type=int, default=2048)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
