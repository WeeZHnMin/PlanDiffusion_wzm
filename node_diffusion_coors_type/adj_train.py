"""
Train AdjGenerationModel: Chinese prompt -> adjacency token sequence.

Usage:
    python -m node_diffusion.adj_train \
        --npz data/processed/nodes_train_6k_norm.npz \
        --bert_path models/bert-base-chinese \
        --save_dir checkpoints/adj_model
"""

import os
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .adj_tokenize_exp import adj_to_tokens
from .adj_model import AdjGenerationModel


class AdjDataset(Dataset):
    def __init__(self, npz_path, bert_path, max_len=128):
        d = np.load(npz_path, allow_pickle=True)
        self.adj_matrix = d['adj_matrix']          # [N, 40, 40]
        self.node_mask  = d['node_mask']            # [N, 40]
        self.prompts    = d['prompts']              # [N] str
        self.tokenizer  = BertTokenizer.from_pretrained(bert_path)
        self.max_len    = max_len
        print(f"AdjDataset: {len(self.prompts)} samples")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        tokens = adj_to_tokens(self.adj_matrix[idx])           # [98] int

        enc = self.tokenizer(
            str(self.prompts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),       # [max_len]
            'attention_mask': enc['attention_mask'].squeeze(0),  # [max_len]
            'tgt_tokens':     torch.from_numpy(tokens).long(),   # [98]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',          default='data/processed/nodes_train_6k_norm.npz')
    parser.add_argument('--bert_path',    default='models/bert-base-chinese')
    parser.add_argument('--save_dir',     default='checkpoints/adj_model')
    parser.add_argument('--resume',       default='')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--total_steps',  type=int,   default=50000)
    parser.add_argument('--log_interval', type=int,   default=100)
    parser.add_argument('--save_interval',type=int,   default=5000)
    parser.add_argument('--d_model',      type=int,   default=256)
    parser.add_argument('--num_layers',   type=int,   default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = AdjDataset(args.npz, args.bert_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2, drop_last=True)

    model = AdjGenerationModel(
        bert_path=args.bert_path,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    opt = AdamW(model.decoder.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.decoder.load_state_dict(ckpt['decoder'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    def inf_loader():
        while True:
            yield from loader

    data = inf_loader()
    model.train()
    running_loss = 0.0

    for step in range(start_step, args.total_steps):
        batch = next(data)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tgt_tokens     = batch['tgt_tokens'].to(device)

        loss = model(input_ids, attention_mask, tgt_tokens)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval if step > 0 else running_loss
            running_loss = 0.0
            # token accuracy on this batch (greedy)
            with torch.no_grad():
                model.eval()
                pred = model.generate(input_ids[:4], attention_mask[:4])  # [4, 98]
                acc  = (pred == tgt_tokens[:4]).float().mean().item()
                model.train()
            print(f"step {step:6d} | loss {avg:.4f} | token_acc {acc:.3f}")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f'adj_model_{step:06d}.pt')
            torch.save({'decoder': model.decoder.state_dict(),
                        'opt':     opt.state_dict(),
                        'step':    step}, path)
            print(f"  saved -> {path}")

    path = os.path.join(args.save_dir, f'adj_model_{args.total_steps:06d}.pt')
    torch.save({'decoder': model.decoder.state_dict(),
                'opt':     opt.state_dict(),
                'step':    args.total_steps}, path)
    print(f"done. saved -> {path}")


if __name__ == '__main__':
    main()
