"""
训练 TextGraphAlign (CLIP 对比预训练，文本编码器从零训练)。

先用 room_type_clf/build_npz.py 构建 npz，再训练：
  python -m room_type_clf.build_npz \\
      --jsonl     data/jsonl/final_graph_dataset_v3.jsonl \\
      --val_jsonl data/jsonl/val_graph_dataset_18k5.jsonl \\
      --augment 3 --output data/processed/room_type_clf/train.npz

  python -m text_graph_align.train \\
      --train data/processed/room_type_clf/train.npz \\
      --val   data/processed/room_type_clf/val.npz \\
      --save  checkpoints/align --gpu 3
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from .dataset import load_align_data
from .model   import TextGraphAlign

VAL_SAMPLES = 4096


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train',        required=True)
    p.add_argument('--val',          default='')
    p.add_argument('--save',         default='checkpoints/align')
    p.add_argument('--gpu',          type=int,   default=0)
    p.add_argument('--batch',        type=int,   default=256)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--warmup',       type=int,   default=500)
    p.add_argument('--steps',        type=int,   default=30000)
    p.add_argument('--log_every',    type=int,   default=100)
    p.add_argument('--save_every',   type=int,   default=2000)
    p.add_argument('--val_every',    type=int,   default=2000)
    p.add_argument('--d_model',      type=int,   default=256)
    p.add_argument('--num_layers',   type=int,   default=4)
    p.add_argument('--num_heads',    type=int,   default=4)
    p.add_argument('--d_embed',      type=int,   default=256)
    p.add_argument('--vocab_size',   type=int,   default=10000)
    p.add_argument('--max_len',      type=int,   default=192)
    p.add_argument('--workers',      type=int,   default=4)
    p.add_argument('--resume',       default='')
    return p


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (1 + math.cos(math.pi * t)) / 2


def run_val(model, val_loader, device):
    model.eval()
    total_loss = total_g2t = total_t2g = n_samples = n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            node_mask  = batch['node_mask'].to(device)
            adj        = batch['adj_matrix'].to(device)
            membership = batch['room_membership'].to(device)
            input_ids  = batch['input_ids'].to(device)
            attn_mask  = batch['attn_mask'].to(device)
            loss, acc_g2t, acc_t2g = model(
                node_mask, adj, membership, input_ids, attn_mask)
            total_loss += loss.item()
            total_g2t  += acc_g2t
            total_t2g  += acc_t2g
            n_samples  += node_mask.shape[0]
            n_batches  += 1
            if n_samples >= VAL_SAMPLES:
                break
    model.train()
    nb = max(n_batches, 1)
    return total_loss / nb, total_g2t / nb, total_t2g / nb


def main():
    args = build_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader = load_align_data(args.train, args.batch, shuffle=True,
                                      num_workers=args.workers)
    val_loader = None
    if args.val:
        _, val_loader = load_align_data(args.val, batch_size=64,
                                        shuffle=False, num_workers=args.workers)

    model = TextGraphAlign(
        vocab_size = args.vocab_size,
        d_model    = args.d_model,
        num_layers = args.num_layers,
        num_heads  = args.num_heads,
        max_len    = args.max_len,
        d_embed    = args.d_embed,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    log_f = open(save_dir / 'log.jsonl', 'a', encoding='utf-8', buffering=1)

    def inf_loader():
        while True:
            yield from train_loader

    model.train()
    data_iter = inf_loader()
    loss_acc = g2t_acc = t2g_acc = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg['lr'] = lr

        batch      = next(data_iter)
        node_mask  = batch['node_mask'].to(device)
        adj        = batch['adj_matrix'].to(device)
        membership = batch['room_membership'].to(device)
        input_ids  = batch['input_ids'].to(device)
        attn_mask  = batch['attn_mask'].to(device)

        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            loss, acc_g2t, acc_t2g = model(
                node_mask, adj, membership, input_ids, attn_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        loss_acc += loss.item()
        g2t_acc  += acc_g2t
        t2g_acc  += acc_t2g

        if (step + 1) % args.log_every == 0:
            n        = args.log_every
            avg_loss = loss_acc / n
            avg_g2t  = g2t_acc  / n
            avg_t2g  = t2g_acc  / n
            elapsed  = time.perf_counter() - t0
            tau      = 1.0 / model.logit_scale.exp().item()
            print(f"step {step+1:6d} | loss {avg_loss:.4f} | "
                  f"g2t {avg_g2t:.2%} | t2g {avg_t2g:.2%} | "
                  f"tau {tau:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            log_f.write(json.dumps({
                'step': step+1, 'loss': round(avg_loss, 4),
                'acc_g2t': round(avg_g2t, 4), 'acc_t2g': round(avg_t2g, 4),
                'tau': round(tau, 4), 'lr': lr, 'elapsed': round(elapsed, 1),
            }) + '\n')
            loss_acc = g2t_acc = t2g_acc = 0.0
            t0 = time.perf_counter()

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ckpt_path = save_dir / f'align_step{step+1:06d}.pt'
            torch.save({'step': step, 'model': model.state_dict(),
                        'opt': opt.state_dict()}, ckpt_path)
            torch.save({'step': step, 'model': model.state_dict()},
                       save_dir / 'align_latest.pt')
            print(f"saved -> {ckpt_path}")

        if val_loader and (step + 1) % args.val_every == 0:
            v_loss, v_g2t, v_t2g = run_val(model, val_loader, device)
            print(f"  [val] loss {v_loss:.4f} | g2t {v_g2t:.2%} | t2g {v_t2g:.2%}")
            log_f.write(json.dumps({
                'step': step+1,
                'val_loss': round(v_loss, 4),
                'val_acc_g2t': round(v_g2t, 4),
                'val_acc_t2g': round(v_t2g, 4),
            }) + '\n')

    log_f.close()
    print("done.")


if __name__ == '__main__':
    main()
