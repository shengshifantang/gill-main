"""Dry-run to validate `collate_fn` with Kolors (ChatGLM) encoding path.
This script uses a fake tokenizer and fake text encoder to simulate
ChatGLM behavior and verifies shapes (on CPU) without loading heavy models.
"""
import sys
import os
import torch

# ensure project root is on sys.path so `gill` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gill.gill_dataset import GILLDataset, collate_fn


class FakeTokenizer:
    def __init__(self, seq_len=77):
        self.seq_len = seq_len

    def __call__(self, texts, padding="max_length", max_length=256, truncation=True, return_tensors='pt'):
        batch = len(texts)
        # produce deterministic input_ids and attention_mask
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
        # emulate hidden_states list where
        # hidden_states[-2] shape = (seq_len, batch, token_dim)
        # hidden_states[-1] shape = (1, batch, pooled_dim)
        batch = input_ids.shape[0]
        h_seq = torch.randn(self.seq_len, batch, self.token_dim)
        h_pool = torch.randn(1, batch, self.pooled_dim)
        class Out:
            pass
        out = Out()
        out.hidden_states = [h_seq, h_pool]
        return out


def main():
    # mock manifest with two items (no image files are read in this dry-run path)
    manifest = [
        {'image_path': 'img1.png', 'caption': '一只在月球上骑马的宇航员', 'bboxes': [[0.1, 0.1, 0.5, 0.5]]},
        {'image_path': 'img2.png', 'caption': '城市夜景中的霓虹灯招牌', 'bboxes': [[0.2, 0.2, 0.6, 0.6]]},
    ]

    # We avoid calling ds[0] to prevent filesystem access; construct batch dicts directly
    tokenizer = FakeTokenizer(seq_len=77)
    text_encoder = FakeTextEncoder(seq_len=77, token_dim=2816, pooled_dim=4096)

    # create a projection that maps pooled_dim (4096) -> expected 5376 (example)
    proj = torch.nn.Linear(4096, 5376)

    # create dummy pixel_values for two samples (C,H,W) normalized like dataset transform
    pv1 = torch.randn(3, 512, 512)
    pv2 = torch.randn(3, 512, 512)
    sample1 = {'pixel_values': pv1, 'caption': manifest[0]['caption'], 'bboxes': torch.tensor(manifest[0]['bboxes'], dtype=torch.float32)}
    sample2 = {'pixel_values': pv2, 'caption': manifest[1]['caption'], 'bboxes': torch.tensor(manifest[1]['bboxes'], dtype=torch.float32)}
    batch = [sample1, sample2]
    out = collate_fn(batch, tokenizer=tokenizer, text_encoder=text_encoder, text_projection=proj, device='cpu', kolors=True)

    print('input_ids.shape ->', out['input_ids'].shape)
    print('encoder_hidden_states.shape ->', None if out['encoder_hidden_states'] is None else out['encoder_hidden_states'].shape)
    print('pooled_embeds.shape ->', None if out['pooled_embeds'] is None else out['pooled_embeds'].shape)
    print('text_embeds_proj.shape ->', None if out.get('text_embeds_proj') is None else out['text_embeds_proj'].shape)


if __name__ == '__main__':
    main()
