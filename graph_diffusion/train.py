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
    p.add_argument("--batch-size", type=int,   default=64)
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
    p.add_argument("--n-head",     type=int,   default=8)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def compute_loss_and_acc(pred_X, pred_E, true_X, true_E, node_mask):
    """
    pred_X : (B, N, Kx) logits
    true_X : (B, N, Kx) one-hot
    pred_E : (B, N, N, Ke) logits
    true_E : (B, N, N, Ke) one-hot
    返回: loss_x, loss_e, acc_x, acc_e
    """
    B, N, Kx = pred_X.shape
    Ke = pred_E.shape[-1]
    x_mask = node_mask.float()
    e_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).float()

    true_X_idx = true_X.argmax(-1)   # (B, N)
    true_E_idx = true_E.argmax(-1)   # (B, N, N)

    # ── 节点损失 + 准确率 ──────────────────────────────────────────
    loss_x = F.cross_entropy(
        pred_X.reshape(B * N, Kx),
        true_X_idx.reshape(B * N),
        reduction='none'
    ).reshape(B, N)
    loss_x = (loss_x * x_mask).sum() / (x_mask.sum() + 1e-8)

    with torch.no_grad():
        pred_X_idx = pred_X.argmax(-1)
        acc_x = ((pred_X_idx == true_X_idx).float() * x_mask).sum() / (x_mask.sum() + 1e-8)

    # ── 边损失 + 准确率（只算上三角）─────────────────────────────
    triu = torch.triu(torch.ones(N, N, device=pred_E.device, dtype=torch.bool), diagonal=1)
    triu_mask = e_mask * triu.unsqueeze(0)

    loss_e = F.cross_entropy(
        pred_E.reshape(B * N * N, Ke),
        true_E_idx.reshape(B * N * N),
        reduction='none'
    ).reshape(B, N, N)
    loss_e = (loss_e * triu_mask).sum() / (triu_mask.sum() + 1e-8)

    with torch.no_grad():
        pred_E_idx = pred_E.argmax(-1)
        # recall_e：实际存在的边中预测对了多少（避免全预测0的虚高准确率）
        actual_pos  = (true_E_idx == 1).float() * triu_mask
        correct_pos = ((pred_E_idx == 1) & (true_E_idx == 1)).float() * triu_mask
        recall_e = correct_pos.sum() / (actual_pos.sum() + 1e-8)

    return loss_x, loss_e, acc_x.item(), recall_e.item()


def main():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # cuDNN 自动选最优卷积算法
    torch.backends.cudnn.benchmark = True

    # 词表配置
    vocab_cfg    = json.loads(open(args.vocab, encoding='utf-8').read())
    bpe_vocab    = vocab_cfg['bpe_vocab_size']

    # 数据
    # Windows 下 num_workers>0 会报错，自动降为 0
    import platform
    nw = 0 if platform.system() == 'Windows' else 4
    loader = make_loader(args.data, args.batch_size, shuffle=True,
                         num_workers=nw, persistent_workers=(nw > 0))

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

    # PyTorch 2.0+ 编译加速（Windows 不支持 Triton，跳过）
    if platform.system() != 'Windows':
        try:
            model = torch.compile(model)
            print('torch.compile 已启用')
        except Exception:
            pass
    else:
        print('Windows：跳过 torch.compile')

    # 扩散过程
    schedule   = GaussianNoiseSchedule(T=args.timesteps)
    transition = DiscreteUniformTransition(x_classes=32, e_classes=2)

    # 优化器
    opt    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                   betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    # Cosine LR 衰减（last_epoch=-1 避免 init 时触发 step 警告）
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_step     = 0
    best_loss      = float('inf')
    running_loss   = 0.0
    running_loss_x = 0.0
    running_loss_e = 0.0
    running_acc_x  = 0.0
    running_acc_e  = 0.0
    save_window_loss  = 0.0   # 用于 best 判断的独立窗口
    save_window_steps = 0

    log_path = save_dir / 'train_log.jsonl'
    # 续训时追加，全新训练时新建
    log_mode = 'a' if args.resume and Path(args.resume).exists() else 'w'
    log_file = open(log_path, log_mode, encoding='utf-8', buffering=1)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        sd   = ckpt['model']
        # 去掉 torch.compile 的 _orig_mod. 前缀和 DataParallel 的 module. 前缀
        def clean_key(k):
            k = k.replace('_orig_mod.', '')
            k = k.replace('module.', '')
            return k
        sd = {clean_key(k): v for k, v in sd.items()}
        # 找到原始模型（剥离 compile 和 DataParallel 包装）
        raw_model = model
        if hasattr(raw_model, '_orig_mod'):   # torch.compile
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, 'module'):      # DataParallel
            raw_model = raw_model.module
        raw_model.load_state_dict(sd, strict=True)
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
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
            loss_x, loss_e, acc_x, recall_e = compute_loss_and_acc(pred_X, pred_E, X, E, node_mask)
            loss = loss_x + 5.0 * loss_e   # 边损失权重更高（稀疏问题）

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        running_loss   += loss.item()
        running_loss_x += loss_x.item()
        running_loss_e += loss_e.item()
        running_acc_x  += acc_x
        running_acc_e  += recall_e
        save_window_loss  += loss.item()
        save_window_steps += 1

        if step % 100 == 0:
            torch.cuda.empty_cache()

        if step % args.log_every == 0 and step > 0:
            n = args.log_every
            avg   = running_loss   / n
            avg_x = running_loss_x / n
            avg_e = running_loss_e / n
            avg_ax = running_acc_x / n
            avg_ae = running_acc_e / n
            running_loss = running_loss_x = running_loss_e = 0.0
            running_acc_x = running_acc_e = 0.0
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()

            msg = (f'step {step:7d} | loss {avg:.4f} '
                   f'| loss_x {avg_x:.4f} acc_x {avg_ax:.3f} '
                   f'| loss_e {avg_e:.4f} recall_e {avg_ae:.3f} '
                   f'| {elapsed:.1f}s')
            print(msg)

            import json as _json
            log_file.write(_json.dumps({
                'step':     step,
                'loss':     round(avg,   4),
                'loss_x':   round(avg_x, 4),
                'loss_e':   round(avg_e, 4),
                'acc_x':    round(avg_ax, 4),
                'recall_e': round(avg_ae, 4),
                'elapsed':  round(elapsed, 1),
            }, ensure_ascii=False) + '\n')

        if step % args.save_every == 0 and step > 0:
            window_avg = save_window_loss / max(save_window_steps, 1)
            save_window_loss  = 0.0
            save_window_steps = 0
            if window_avg < best_loss:
                best_loss = window_avg
                _m = model
                if hasattr(_m, '_orig_mod'): _m = _m._orig_mod
                if hasattr(_m, 'module'):    _m = _m.module
                torch.save({
                    'model':     _m.state_dict(),
                    'opt':       opt.state_dict(),
                    'scaler':    scaler.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step':      step,
                }, save_dir / 'best.pt')
                print(f'  best saved → step={step} loss={best_loss:.4f}')

    _m = model
    if hasattr(_m, '_orig_mod'): _m = _m._orig_mod
    if hasattr(_m, 'module'):    _m = _m.module
    torch.save({
        'model':     _m.state_dict(),
        'opt':       opt.state_dict(),
        'scaler':    scaler.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step':      args.total_steps,
    }, save_dir / 'final.pt')
    log_file.close()
    print('训练完成')


if __name__ == '__main__':
    main()
