import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GILLDataset(Dataset):
    """Minimal dataset template for GILL experiments.

    Expects a folder with images and a JSON/TSV manifest that provides:
      - image_path
      - caption (text)
      - bboxes: list of normalized boxes [[x0,y0,x1,y1], ...]

    The dataset returns:
      - pixel_values: torch.FloatTensor, shape (C,H,W)
      - caption: raw text (or token ids if you integrate tokenizer)
      - bboxes: torch.FloatTensor, shape (N,4) (normalized 0..1)

    This template also supports an optional `text_projection` function that
    maps encoder outputs to the UNet-expected `text_dim` discovered from
    `unet.add_embedding` configuration.
    """

    def __init__(self, manifest, image_root, transform=None):
        # manifest: list of dicts or path to jsonl/tsv. For simplicity accept list.
        if isinstance(manifest, str):
            # load a jsonl file
            data = []
            with open(manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    import json
                    data.append(json.loads(line))
            self.items = data
        else:
            self.items = manifest
        self.image_root = image_root
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec.get('image_path')
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        img = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(img)

        caption = rec.get('caption', "")
        bboxes = rec.get('bboxes', [])
        # ensure tensor shape (N,4)
        if isinstance(bboxes, (list, tuple)):
            if len(bboxes) == 0:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                bboxes = torch.tensor(bboxes, dtype=torch.float32)
        elif isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.float()

        return {
            'pixel_values': pixel_values,
            'caption': caption,
            'bboxes': bboxes
        }


def collate_fn(batch, tokenizer=None, text_encoder=None, text_projection=None, device='cuda', kolors=False):
    """Collate that optionally encodes text and projects embeddings.

    - If `tokenizer`+`text_encoder` provided, returns `input_ids` and
      `encoder_hidden_states` (token-level) and `pooled_embeds` (pooled).
    - If `text_projection` provided (nn.Module), it will project pooled
      embeddings to required `text_dim` for UNet.
    """
    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0).to(device)
    captions = [b['caption'] for b in batch]
    bboxes = [b['bboxes'] for b in batch]

    out = {'pixel_values': pixel_values, 'captions': captions, 'bboxes': bboxes}

    if tokenizer is not None and text_encoder is not None:
        # tokenization
        if kolors:
            # Kolors / ChatGLM style encoding: need position_ids and output_hidden_states
            tokens = tokenizer(captions, padding="max_length", max_length=256, truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            # position_ids may be provided by tokenizer; otherwise create
            position_ids = tokens.get('position_ids', None)
            if position_ids is not None:
                position_ids = position_ids.to(device)

            # encode (we expect ChatGLMModel-like return with hidden_states)
            with torch.no_grad():
                enc_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True)
            # follow Kolors pipeline extraction: hidden_states[-2] -> token-level, hidden_states[-1][-1,:,:] -> pooled
            hidden_states = getattr(enc_out, 'hidden_states', None)
            # Robust extraction:
            # - token-level features are typically 3D tensors (seq, batch, dim) in Kolors' returns
            # - pooled features may appear as a 2D tensor (batch, dim) or as a last-token slice
            encoder_hidden_states = None
            pooled = None
            if hidden_states is None:
                encoder_hidden_states = getattr(enc_out, 'last_hidden_state', None)
                if encoder_hidden_states is not None and encoder_hidden_states.dim() == 3:
                    # to (batch, seq, dim)
                    encoder_hidden_states = encoder_hidden_states.permute(0, 1, 2).clone()
                pooled = (encoder_hidden_states.mean(dim=1) if encoder_hidden_states is not None else None)
            else:
                # find candidate tensors by dimensionality
                token_cands = [h for h in hidden_states if isinstance(h, torch.Tensor) and h.dim() == 3]
                pooled_cands = [h for h in hidden_states if isinstance(h, torch.Tensor) and h.dim() == 2]

                if len(token_cands) > 0:
                    # prefer the last 3D tensor as token-level
                    tok = token_cands[-1]
                    # expected tok shape: (seq, batch, dim) -> convert to (batch, seq, dim)
                    try:
                        encoder_hidden_states = tok.permute(1, 0, 2).clone()
                    except Exception:
                        encoder_hidden_states = tok.clone()

                if len(pooled_cands) > 0:
                    # prefer last 2D tensor as pooled
                    pooled = pooled_cands[-1].clone()
                else:
                    # fallback: derive pooled by pooling token-level features if available
                    if encoder_hidden_states is not None:
                        pooled = encoder_hidden_states.mean(dim=1)

            out.update({'input_ids': input_ids, 'encoder_hidden_states': encoder_hidden_states, 'pooled_embeds': pooled})

            if text_projection is not None and pooled is not None:
                out['text_embeds_proj'] = text_projection(pooled)
        else:
            # generic transformers path
            tokens = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            # encode (user must ensure text_encoder returns token-level hidden states and pooled)
            with torch.no_grad():
                enc_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # user must adapt based on encoder's return signature
            encoder_hidden_states = getattr(enc_out, 'last_hidden_state', None) or getattr(enc_out, 'hidden_states', None)
            pooled = getattr(enc_out, 'pooler_output', None) or (encoder_hidden_states.mean(dim=1) if encoder_hidden_states is not None else None)
            out.update({'input_ids': input_ids, 'encoder_hidden_states': encoder_hidden_states, 'pooled_embeds': pooled})

            if text_projection is not None and pooled is not None:
                # project pooled to required UNet text_dim
                out['text_embeds_proj'] = text_projection(pooled)

    return out
