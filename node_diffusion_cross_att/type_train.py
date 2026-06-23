"""
Train NodeTypeClassifier: predict node type from clean coordinates + graph + text.

Usage:
  python -m node_diffusion_cross_att.type_train \\
      --data_path data/processed/node_diffusion_cross_att/graph_dataset.npz \\
      --bert      models/bert-base-uncased
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import TypeDataset
from .type_model import NodeTypeClassifier


def build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',     default=defaults.get('data_path',     'data/processed/node_diffusion_cross_att/type_dataset.npz'))
    p.add_argument('--save_dir',      default=defaults.get('save_dir',      'checkpoints/node_type'))
    p.add_argument('--resume',        default='',  help='path to checkpoint .pt')
    p.add_argument('--batch_size',    type=int,   default=defaults.get('batch_size',    64))
    p.add_argument('--lr',            type=float, default=defaults.get('lr',            1e-4))
    p.add_argument('--weight_decay',  type=float, default=defaults.get('weight_decay',  1e-4))
    p.add_argument('--total_steps',   type=int,   default=defaults.get('total_steps',   500000))
    p.add_argument('--log_interval',  type=int,   default=defaults.get('log_interval',  100))
    p.add_argument('--save_interval', type=int,   default=defaults.get('save_interval', 10000))
    p.add_argument('--model_channels',  type=int,   default=defaults.get('model_channels',  384))
    p.add_argument('--num_layers',      type=int,   default=defaults.get('num_layers',      4))
    p.add_argument('--num_heads',       type=int,   default=defaults.get('num_heads',       6))
    p.add_argument('--dropout',         type=float, default=defaults.get('dropout',         0.4))
    p.add_argument('--bert',            default=defaults.get('bert', 'models/bert-base-uncased'))
    p.add_argument('--unfreeze_layers', type=int,   default=defaults.get('unfreeze_layers', 0))
    p.add_argument('--gpu',             type=int,   default=None)
    return p


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


def compute_acc(logits, targets, node_mask):
    """Accuracy over valid (non-padding) nodes only."""
    pred  = logits.argmax(dim=-1)          # [B, N]
    valid = node_mask > 0.5                # [B, N]
    correct = (pred == targets) & valid
    return correct.sum().float() / valid.sum().float().clamp(min=1)


def inf_loader(loader):
    while True:
        yield from loader


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / 'log.jsonl'
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    print(f'日志: {log_path}')

    model = NodeTypeClassifier(
        model_channels  = args.model_channels,
        num_layers      = args.num_layers,
        num_heads       = args.num_heads,
        dropout         = args.dropout,
        bert_name       = args.bert,
        unfreeze_layers = args.unfreeze_layers,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    opt     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_sd = ckpt['model']
        if any(k.startswith('module.') for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        model.load_state_dict(raw_sd)
        opt.load_state_dict(ckpt['opt'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1
        print(f'resumed from step {start_step}')

    dataset = TypeDataset(args.data_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, drop_last=True)
    data = inf_loader(loader)

    model.train()
    running_loss = running_acc = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x    = x.to(device)
        cond = move_cond(cond, device)

        # 模拟 θ₂ 推理误差（coord_rmse ≈ 4px），提升 θ₃ 对不完美坐标的鲁棒性
        noise_scale = torch.empty(1).uniform_(2.0, 6.0).item()
        x = x + torch.randn_like(x) * noise_scale

        targets   = cond['node_types']      # [B, N]  1-32, 0=padding
        node_mask = cond['node_mask']       # [B, N]

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                x,
                adj_matrix    = cond['adj_matrix'],
                node_mask     = node_mask,
                prompt_tokens = cond['prompt_tokens'],
                prompt_mask   = cond['prompt_mask'],
            )                               # [B, N, 33]
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1).long(),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            acc = compute_acc(logits.float(), targets, node_mask).item()

        running_loss += loss.item()
        running_acc  += acc

        if step % args.log_interval == 0 and step > 0:
            n        = args.log_interval
            avg_loss = running_loss / n
            avg_acc  = running_acc  / n
            running_loss = running_acc = 0.0
            elapsed  = time.perf_counter() - t0
            t0       = time.perf_counter()

            print(f'step {step:6d} | loss {avg_loss:.4f} | acc {avg_acc:.4f} | {elapsed:.1f}s')
            log_file.write(json.dumps({
                'step': step,
                'loss': round(avg_loss, 4),
                'acc':  round(avg_acc,  4),
                'elapsed': round(elapsed, 1),
            }) + '\n')

        if step > 0 and step % args.save_interval == 0:
            ckpt_path = save_dir / 'model_latest.pt'
            torch.save({
                'model':  model.state_dict(),
                'opt':    opt.state_dict(),
                'scaler': scaler.state_dict(),
                'step':   step,
            }, ckpt_path)
            print(f'  saved -> {ckpt_path}')

    ckpt_path = save_dir / 'model_latest.pt'
    torch.save({
        'model':  model.state_dict(),
        'opt':    opt.state_dict(),
        'scaler': scaler.state_dict(),
        'step':   args.total_steps,
    }, ckpt_path)
    log_file.close()
    print(f'training done. saved -> {ckpt_path}')


if __name__ == '__main__':
    main()
