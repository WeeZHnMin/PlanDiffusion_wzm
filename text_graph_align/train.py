"""
Text-Graph alignment pretraining (InfoNCE).

Usage:
    python -m text_graph_align.train \
        --data  data/jsonl/final_graph_dataset_v3.jsonl \
        --save  checkpoints/align \
        --gpu   0
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from .dataset import AlignDataset
from .model   import AlignModel

from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',     default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--bert',     default='models/bert-base-uncased')
    p.add_argument('--save',     default='checkpoints/align')
    p.add_argument('--gpu',      type=int, default=0)
    p.add_argument('--batch',    type=int, default=256)
    p.add_argument('--lr',       type=float, default=1e-4)
    p.add_argument('--warmup',   type=int, default=500)
    p.add_argument('--steps',    type=int, default=30000)
    p.add_argument('--log_every',type=int, default=100)
    p.add_argument('--tau',      type=float, default=0.07)
    p.add_argument('--out_dim',  type=int, default=384)
    p.add_argument('--unfreeze', type=int, default=2)
    p.add_argument('--workers',  type=int, default=4)
    return p.parse_args()


def cosine_lr_with_warmup(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (1 + math.cos(math.pi * t)) / 2


import math


def main():
    args = parse_args()
    os.makedirs(args.save, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds     = AlignDataset(args.data, bert_name=args.bert)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)

    model = AlignModel(bert_name=args.bert, out_dim=args.out_dim,
                       unfreeze_last_n=args.unfreeze, tau=args.tau).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    opt    = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler()

    log_path = os.path.join(args.save, 'train_log.jsonl')
    log_f    = open(log_path, 'a')

    step      = 0
    t0        = time.time()
    loss_accum = 0.0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            # lr schedule
            lr = cosine_lr_with_warmup(step, args.steps, args.warmup, args.lr)
            for pg in opt.param_groups:
                pg['lr'] = lr

            ids   = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            coords= batch['coords'].to(device)
            adj   = batch['adj'].to(device)
            mask  = batch['mask'].to(device)

            opt.zero_grad()
            with autocast():
                loss, _, _ = model(ids, amask, coords, adj, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()

            loss_accum += loss.item()
            step       += 1

            if step % args.log_every == 0:
                avg_loss = loss_accum / args.log_every
                elapsed  = time.time() - t0
                rec = {'step': step, 'loss': avg_loss, 'lr': lr,
                       'elapsed_s': round(elapsed, 1)}
                print(f"step {step:6d}  loss {avg_loss:.4f}  lr {lr:.2e}  "
                      f"elapsed {elapsed/60:.1f}m")
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                loss_accum = 0.0

            if step % 5000 == 0 or step == args.steps:
                ckpt = os.path.join(args.save, f'align_step{step:06d}.pt')
                torch.save({
                    'step':       step,
                    'model':      model.state_dict(),
                    'opt':        opt.state_dict(),
                }, ckpt)
                # always keep a "latest" too
                torch.save({
                    'step':  step,
                    'model': model.state_dict(),
                }, os.path.join(args.save, 'align_latest.pt'))
                print(f"Saved {ckpt}")

    log_f.close()
    print("Done.")


if __name__ == '__main__':
    main()
