"""
DiGress 风格离散扩散训练：文本条件图结构生成。

用法：
  python -m graph_diffusion.train

Kaggle 路径示例：
  python -m graph_diffusion.train \
    --data /kaggle/input/.../graph_dataset.npz \
    --vocab /kaggle/working/PlanDiffusion_wzm/node_diffusion/unified_vocab/vocab_config.json \
    --save-dir /kaggle/working/checkpoints/graph_diffusion
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from graph_diffusion.dataset  import make_loader
from graph_diffusion.model    import GraphTransformer
from graph_diffusion.diffusion import (
    GaussianNoiseSchedule, DiscreteUniformTransition, apply_noise
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed/graph_diffusion/graph_dataset.npz")
    p.add_argument("--vocab",      default="node_diffusion/unified_vocab/vocab_config.json")
    p.add_argument("--save-dir",   default="checkpoints/graph_diffusion")
    p.add_argument("--resume",     default=None)

    # 训练超参数
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--total-steps",type=int,   default=500_000)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=200)
    p.add_argument("--save-every", type=int,   default=10_000)
    p.add_argument("--timesteps",  type=int,   default=500)

    # 模型超参数
    p.add_argument("--n-layers",   type=int,   default=6)
    p.add_argument("--dx",         type=int,   default=256)
    p.add_argument("--de",         type=int,   default=64)
    p.add_argument("--dy",         type=int,   default=256)
    p.add_argument("--n-head",     type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def compute_loss(pred_X, pred_E, true_X, true_E, node_mask):
    """
    pred_X : (B, N, Kx) logits
    true_X : (B, N, Kx) one-hot
    pred_E : (B, N, N, Ke) logits
    true_E : (B, N, N, Ke) one-hot
    """
    B, N, Kx = pred_X.shape
    Ke = pred_E.shape[-1]
    x_mask = node_mask.float()                                     # (B, N)
    e_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).float()  # (B, N, N)

    # 节点损失
    true_X_idx = true_X.argmax(-1)                                 # (B, N)
    loss_x = F.cross_entropy(
        pred_X.reshape(B * N, Kx),
        true_X_idx.reshape(B * N),
        reduction='none'
    ).reshape(B, N)
    loss_x = (loss_x * x_mask).sum() / (x_mask.sum() + 1e-8)

    # 边损失（对称，只算上三角避免重复）
    true_E_idx = true_E.argmax(-1)                                 # (B, N, N)
    loss_e = F.cross_entropy(
        pred_E.reshape(B * N * N, Ke),
        true_E_idx.reshape(B * N * N),
        reduction='none'
    ).reshape(B, N, N)

    # 只取上三角
    triu = torch.triu(torch.ones(N, N, device=pred_E.device, dtype=torch.bool), diagonal=1)
    triu_mask = e_mask * triu.unsqueeze(0)
    loss_e = (loss_e * triu_mask).sum() / (triu_mask.sum() + 1e-8)

    return loss_x, loss_e


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 词表配置
    vocab_cfg    = json.loads(open(args.vocab, encoding='utf-8').read())
    bpe_vocab    = vocab_cfg['bpe_vocab_size']

    # 数据
    loader = make_loader(args.data, args.batch_size, shuffle=True, num_workers=4)

    # 模型
    model = GraphTransformer(
        x_classes      = 32,
        e_classes      = 2,
        bpe_vocab_size = bpe_vocab,
        text_embed_dim = 128,
        n_layers       = args.n_layers,
        dx             = args.dx,
        de             = args.de,
        dy             = args.dy,
        n_head         = args.n_head,
        dropout        = args.dropout,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f'使用 {torch.cuda.device_count()} 张 GPU')
        model = nn.DataParallel(model)

    # 扩散过程
    schedule   = GaussianNoiseSchedule(T=args.timesteps)
    transition = DiscreteUniformTransition(x_classes=32, e_classes=2)

    # 优化器
    opt    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_step   = 0
    best_loss    = float('inf')
    running_loss = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        sd   = ckpt['model']
        sd   = {k.replace('module.', ''): v for k, v in sd.items()}
        (model.module if hasattr(model, 'module') else model).load_state_dict(sd)
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1
        print(f'resumed from step {start_step}')

    # 无限数据迭代器
    def infinite():
        while True:
            yield from loader

    data_iter = infinite()
    t0 = time.perf_counter()

    model.train()
    for step in range(start_step, args.total_steps):
        X, E, node_mask, ptokens, plens = next(data_iter)
        X         = X.to(device)
        E         = E.to(device)
        node_mask = node_mask.to(device)
        ptokens   = ptokens.to(device)
        plens     = plens.to(device) if isinstance(plens, torch.Tensor) else torch.tensor(plens).to(device)

        B = X.shape[0]
        t_int   = torch.randint(1, args.timesteps + 1, (B,), device=device)
        t_float = t_int.float() / args.timesteps

        # 加噪
        Xt, Et, params = apply_noise(X, E, node_mask, t_int - 1, schedule, transition, device)

        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred_X, pred_E = model(Xt, Et, node_mask, ptokens, plens, t_float)
            loss_x, loss_e = compute_loss(pred_X, pred_E, X, E, node_mask)
            loss = loss_x + 5.0 * loss_e   # 边损失权重更高（稀疏问题）

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()

        if step % args.log_every == 0 and step > 0:
            avg = running_loss / args.log_every
            running_loss = 0.0
            elapsed = time.perf_counter() - t0
            print(f'step {step:7d} | loss {avg:.4f} | x {loss_x.item():.4f} | e {loss_e.item():.4f} | {elapsed:.1f}s')
            t0 = time.perf_counter()

        if step % args.save_every == 0 and step > 0:
            path = save_dir / 'latest.pt'
            torch.save({
                'model':  model.state_dict(),
                'opt':    opt.state_dict(),
                'scaler': scaler.state_dict(),
                'step':   step,
            }, path)
            print(f'  saved → {path}')

    torch.save({
        'model':  model.state_dict(),
        'opt':    opt.state_dict(),
        'scaler': scaler.state_dict(),
        'step':   args.total_steps,
    }, save_dir / 'final.pt')
    print('训练完成')


if __name__ == '__main__':
    main()
