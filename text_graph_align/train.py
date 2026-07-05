"""
训练 TextGraphAlign (CLIP 对比预训练，BERT 文本编码器 + 布局图编码器)。

使用 node_diffusion_room_tri/build_graph_npz.py 生成的 npz：
  python -m text_graph_align.train \
      --train data/processed/node_diffusion_room_tri/graph_dataset.npz \
      --val   data/processed/node_diffusion_room_tri/graph_dataset_val.npz \
      --save  checkpoints/text_graph_align \
      --bert  models/bert-base-uncased \
      --gpu   0

训练完毕后，bert 权重保存在 save_dir/bert_aligned.pt，
可直接加载进 node_diffusion_room_proj 的扩散模型。
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
    p.add_argument('--train',           required=True)
    p.add_argument('--val',             default='')
    p.add_argument('--save',            default='checkpoints/text_graph_align')
    p.add_argument('--gpus',            default='0',
                   help='使用的 GPU，单卡: "0"，双卡: "0,1"')
    p.add_argument('--batch',           type=int,   default=256)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--bert_lr',         type=float, default=1e-5)
    p.add_argument('--weight_decay',    type=float, default=1e-2)
    p.add_argument('--warmup',          type=int,   default=500)
    p.add_argument('--steps',           type=int,   default=30000)
    p.add_argument('--log_every',       type=int,   default=100)
    p.add_argument('--save_every',      type=int,   default=2000)
    p.add_argument('--val_every',       type=int,   default=2000)
    p.add_argument('--d_model',         type=int,   default=384)
    p.add_argument('--num_layers',      type=int,   default=4)
    p.add_argument('--num_heads',       type=int,   default=6)
    p.add_argument('--d_embed',         type=int,   default=384)
    p.add_argument('--bert',            default='models/bert-base-uncased')
    p.add_argument('--unfreeze_layers', type=int,   default=4)
    p.add_argument('--workers',         type=int,   default=4)
    p.add_argument('--resume',          default='')
    return p


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (1 + math.cos(math.pi * t)) / 2


def run_val(raw_model, val_loader, device):
    """用 raw_model 做验证，不经过 DataParallel。"""
    raw_model.eval()
    total_loss = total_g2t = total_t2g = n_batches = n_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            coords  = batch['node_coords'].to(device)
            adj     = batch['adj_matrix'].to(device)
            mask    = batch['node_mask'].to(device)
            member  = batch['room_membership'].to(device)
            ptok    = batch['prompt_tokens'].to(device)
            pmsk    = batch['prompt_mask'].to(device)
            loss             = raw_model(coords, adj, mask, member, ptok, pmsk)
            acc_g2t, acc_t2g = raw_model.compute_metrics(coords, adj, mask, member, ptok, pmsk)
            total_loss += loss.item()
            total_g2t  += acc_g2t
            total_t2g  += acc_t2g
            n_samples  += mask.shape[0]
            n_batches  += 1
            if n_samples >= VAL_SAMPLES:
                break
    raw_model.train()
    nb = max(n_batches, 1)
    return total_loss / nb, total_g2t / nb, total_t2g / nb


def main():
    args = build_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus  = len(args.gpus.split(','))
    print(f"device: {device}  gpus: {args.gpus}  n_gpus: {n_gpus}")

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader = load_align_data(args.train, args.batch, shuffle=True,
                                      num_workers=args.workers)
    val_loader = None
    if args.val:
        _, val_loader = load_align_data(args.val, batch_size=64,
                                        shuffle=False, num_workers=args.workers)

    raw_model = TextGraphAlign(
        bert_name       = args.bert,
        unfreeze_layers = args.unfreeze_layers,
        d_model         = args.d_model,
        num_layers      = args.num_layers,
        num_heads       = args.num_heads,
        d_embed         = args.d_embed,
    ).to(device)

    model = nn.DataParallel(raw_model) if n_gpus > 1 else raw_model

    # 差异化学习率：BERT 解冻层用 bert_lr，其余用 lr
    bert_param_ids = set(id(p) for p in raw_model.text_enc.bert.parameters()
                         if p.requires_grad)
    bert_params    = [p for p in raw_model.parameters()
                      if p.requires_grad and id(p) in bert_param_ids]
    other_params   = [p for p in raw_model.parameters()
                      if p.requires_grad and id(p) not in bert_param_ids]
    opt = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr,      'weight_decay': args.weight_decay},
        {'params': bert_params,  'lr': args.bert_lr, 'weight_decay': args.weight_decay},
    ])
    scaler = torch.amp.GradScaler('cuda')

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    log_f    = open(save_dir / 'log.jsonl', 'a', encoding='utf-8', buffering=1)
    best_val = float('inf')

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
            scale = args.bert_lr / args.lr if pg['params'] is bert_params else 1.0
            pg['lr'] = lr * scale

        batch   = next(data_iter)
        coords  = batch['node_coords'].to(device)
        adj     = batch['adj_matrix'].to(device)
        mask    = batch['node_mask'].to(device)
        member  = batch['room_membership'].to(device)
        ptok    = batch['prompt_tokens'].to(device)
        pmsk    = batch['prompt_mask'].to(device)

        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = model(coords, adj, mask, member, ptok, pmsk)

        scaler.scale(loss.mean()).backward()
        scaler.unscale_(opt)
        trainable = [p for p in model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        loss_acc += loss.mean().item()
        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                a_g2t, a_t2g = raw_model.compute_metrics(
                    coords[:64].to(device),  adj[:64].to(device),
                    mask[:64].to(device),    member[:64].to(device),
                    ptok[:64].to(device),    pmsk[:64].to(device))
            g2t_acc += a_g2t
            t2g_acc += a_t2g

        if (step + 1) % args.log_every == 0:
            n        = args.log_every
            avg_loss = loss_acc / n
            avg_g2t  = g2t_acc          # 只算一次，不除以 n
            avg_t2g  = t2g_acc
            elapsed  = time.perf_counter() - t0
            tau      = 1.0 / raw_model.logit_scale.exp().item()
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
            torch.save({'step': step, 'model': raw_model.state_dict(),
                        'opt': opt.state_dict()},
                       save_dir / 'align_latest.pt')
            torch.save(raw_model.text_enc.bert.state_dict(),
                       save_dir / 'bert_aligned_latest.pt')
            print(f"saved latest (step {step+1})")

        if val_loader and (step + 1) % args.val_every == 0:
            v_loss, v_g2t, v_t2g = run_val(raw_model, val_loader, device)
            is_best = v_loss < best_val
            if is_best:
                best_val = v_loss
                torch.save({'step': step, 'model': raw_model.state_dict()},
                           save_dir / 'align_best.pt')
                torch.save(raw_model.text_enc.bert.state_dict(),
                           save_dir / 'bert_aligned_best.pt')
            print(f"  [val] loss {v_loss:.4f} | g2t {v_g2t:.2%} | t2g {v_t2g:.2%}"
                  + (" ← best" if is_best else ""))
            log_f.write(json.dumps({
                'step': step+1,
                'val_loss': round(v_loss, 4),
                'val_acc_g2t': round(v_g2t, 4),
                'val_acc_t2g': round(v_t2g, 4),
                'is_best': is_best,
            }) + '\n')

    log_f.close()
    print("done.")


if __name__ == '__main__':
    main()
