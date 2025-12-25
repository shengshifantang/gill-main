"""Checks a manifest and optionally runs collate dry-run using Fake tokenizer/encoder.

Usage:
    python3 scripts/check_manifest_and_dryrun.py --manifest data/sample_manifest.jsonl --dry-run
"""
import os
import json
import argparse
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gill.gill_dataset import GILLDataset, collate_fn


def load_manifest(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


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
        h_seq = torch.randn(self.seq_len, batch, self.token_dim)
        h_pool = torch.randn(1, batch, self.pooled_dim)
        class Out: pass
        out = Out()
        out.hidden_states = [h_seq, h_pool]
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    items = load_manifest(args.manifest)
    print(f'Loaded {len(items)} manifest items. First two:')
    for i, it in enumerate(items[:2]):
        print(i, it)

    # check files
    missing = []
    for it in items[:10]:
        path = it['image_path']
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(args.manifest), path)
        if not os.path.exists(path):
            missing.append(path)
    if len(missing) > 0:
        print('\nWarning: some files are missing (paths shown). If files are on a mounted path, run this script on that machine.')
        for m in missing[:10]:
            print(' MISSING:', m)
    else:
        print('\nAll checked files exist (within sample subset).')

    if args.dry_run:
        print('\nRunning collate dry-run using Fake tokenizer/encoder...')
        tokenizer = FakeTokenizer(seq_len=77)
        text_encoder = FakeTextEncoder(seq_len=77, token_dim=2816, pooled_dim=4096)
        proj = torch.nn.Linear(4096, 5376)
        # construct two sample dicts without reading images
        sample_batch = []
        for it in items[:2]:
            pv = torch.randn(3, 512, 512)
            bboxes = torch.tensor(it.get('bboxes', []), dtype=torch.float32)
            sample_batch.append({'pixel_values': pv, 'caption': it.get('caption', ''), 'bboxes': bboxes})
        out = collate_fn(sample_batch, tokenizer=tokenizer, text_encoder=text_encoder, text_projection=proj, device='cpu', kolors=True)
        print('input_ids.shape ->', out['input_ids'].shape)
        print('encoder_hidden_states.shape ->', None if out['encoder_hidden_states'] is None else out['encoder_hidden_states'].shape)
        print('pooled_embeds.shape ->', None if out['pooled_embeds'] is None else out['pooled_embeds'].shape)
        print('text_embeds_proj.shape ->', None if out.get('text_embeds_proj') is None else out['text_embeds_proj'].shape)


if __name__ == '__main__':
    main()
