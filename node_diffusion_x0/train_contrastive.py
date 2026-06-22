"""
对比预训练：将 BERT 文本嵌入与户型图嵌入对齐（类 CLIP）。

训练完成后自动导出：
  <save_dir>/bert_contrastive/   ← 可直接作为 train.py 的 --bert 参数

用法：
  python -m node_diffusion_cross_att.train_contrastive \\
      --data_path data/processed/node_diffusion_cross_att/graph_dataset.npz \\
      --bert      models/bert-base-uncased \\
      --save_dir  checkpoints/contrastive

续训：
  python -m node_diffusion_cross_att.train_contrastive \\
      --resume checkpoints/contrastive/contrastive_0030000.pt

注意：--stride 应与 build_graph_npz.py 的 --augment 保持一致（默认均为 8），
      以避免同一图的不同节点排列在同一 batch 内构成误负样本对。
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .contrastive_model import ContrastivePretrain, info_nce_loss


# ── 数据集 ────────────────────────────────────────────────────────────────────

class ContrastiveDataset(Dataset):
    """
    从 NPZ 中按 stride 采样，跳过节点排列增强副本，只取原始顺序样本。
    stride=8 对应 build_graph_npz.py --augment 8 时每组第一条即原始排列。
    """

    def __init__(self, npz_path, stride=8):
        d   = np.load(npz_path, allow_pickle=True)
        idx = np.arange(0, len(d['node_coords']), stride)

        self.coords        = d['node_coords'][idx].astype(np.float32)    # [N, 40, 2]
        self.adj_matrix    = d['adj_matrix'][idx].astype(np.float32)     # [N, 40, 40]
        self.node_mask     = d['node_mask'][idx].astype(np.float32)      # [N, 40]
        self.node_types    = d['node_combo_ids'][idx].astype(np.int64)   # [N, 40]
        self.prompt_tokens = d['prompt_tokens'][idx].astype(np.int64)    # [N, 192]
        self.prompt_mask   = d['prompt_mask'][idx].astype(np.int64)      # [N, 192]
        print(f'ContrastiveDataset: {len(self.coords)} samples (stride={stride})')

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return {
            'input_ids':      torch.from_numpy(self.prompt_tokens[i]),
            'attention_mask': torch.from_numpy(self.prompt_mask[i]),
            'coords':         torch.from_numpy(self.coords[i]),
            'adj_matrix':     torch.from_numpy(self.adj_matrix[i]),
            'node_mask':      torch.from_numpy(self.node_mask[i]),
            'node_types':     torch.from_numpy(self.node_types[i]),
        }


# ── collate：动态截断到 batch 内实际最长 token 数 ─────────────────────────────

def collate_fn(batch):
    max_len = max(b['attention_mask'].sum().item() for b in batch)
    max_len = int(max_len)
    for b in batch:
        b['input_ids']      = b['input_ids'][:max_len]
        b['attention_mask'] = b['attention_mask'][:max_len]
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ── 参数 ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',        default='data/processed/node_diffusion_cross_att/graph_dataset.npz')
    p.add_argument('--bert',             default='models/bert-base-uncased')
    p.add_argument('--save_dir',         default='checkpoints/contrastive')
    p.add_argument('--resume',           default='')
    p.add_argument('--stride',           type=int,   default=8)
    p.add_argument('--batch_size',       type=int,   default=192)
    p.add_argument('--total_steps',      type=int,   default=60000)
    p.add_argument('--lr_bert',          type=float, default=5e-5,
                   help='解冻的 BERT 层学习率（较小）')
    p.add_argument('--lr_graph',         type=float, default=1e-4,
                   help='Graph Encoder 及投影头学习率')
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--log_interval',     type=int,   default=100)
    p.add_argument('--save_interval',    type=int,   default=10000)
    p.add_argument('--embed_dim',        type=int,   default=512)
    p.add_argument('--d_model',          type=int,   default=384)
    p.add_argument('--num_graph_layers', type=int,   default=4)
    p.add_argument('--num_heads',        type=int,   default=6)
    p.add_argument('--unfreeze_layers',  type=int,   default=2)
    return p


# ── 保存 ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, opt, scaler, step, save_dir):
    path = Path(save_dir) / f'contrastive_{step:07d}.pt'
    torch.save({
        'model':  model.state_dict(),
        'opt':    opt.state_dict(),
        'scaler': scaler.state_dict(),
        'step':   step,
    }, path)
    print(f'  saved → {path}')


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main(argv=None):
    args   = build_parser().parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(save_dir / 'log.jsonl', 'w', encoding='utf-8', buffering=1)

    # ── 模型 ──
    model = ContrastivePretrain(
        bert_name        = args.bert,
        embed_dim        = args.embed_dim,
        d_model          = args.d_model,
        num_graph_layers = args.num_graph_layers,
        num_heads        = args.num_heads,
        unfreeze_layers  = args.unfreeze_layers,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'参数: {trainable:,} trainable / {total:,} total')

    # ── 优化器：BERT 用较小 lr ──
    bert_params  = [p for p in model.text_enc.bert.parameters() if p.requires_grad]
    bert_ids     = {id(p) for p in bert_params}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in bert_ids]
    opt = AdamW([
        {'params': bert_params,  'lr': args.lr_bert},
        {'params': other_params, 'lr': args.lr_graph},
    ], weight_decay=args.weight_decay)

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1
        print(f'resumed from step {start_step}')

    # ── 数据 ──
    ds     = ContrastiveDataset(args.data_path, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True,
                        collate_fn=collate_fn)

    def infinite():
        while True:
            yield from loader

    data = infinite()

    # ── 训练循环 ──
    model.train()
    running_loss = running_acc = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.total_steps):
        batch = {k: v.to(device) for k, v in next(data).items()}

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                             enabled=use_amp):
            text_emb, graph_emb = model(
                input_ids      = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                coords         = batch['coords'],
                adj_matrix     = batch['adj_matrix'],
                node_mask      = batch['node_mask'],
                node_types     = batch['node_types'],
            )
            loss, acc = info_nce_loss(text_emb, graph_emb, model.logit_scale)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        running_acc  += acc

        if step % args.log_interval == 0 and step > 0:
            n        = args.log_interval
            avg_loss = running_loss / n
            avg_acc  = running_acc  / n
            running_loss = running_acc = 0.0
            elapsed  = time.perf_counter() - t0
            t0       = time.perf_counter()
            scale_v  = model.logit_scale.exp().item()
            print(f'step {step:6d} | loss {avg_loss:.4f} | '
                  f'acc {avg_acc:.3f} | scale {scale_v:.2f} | {elapsed:.1f}s')
            log_file.write(json.dumps({
                'step': step, 'loss': round(avg_loss, 4),
                'acc':  round(avg_acc,  4), 'scale': round(scale_v, 2),
                'elapsed': round(elapsed, 1),
            }) + '\n')

        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, opt, scaler, step, save_dir)

    save_checkpoint(model, opt, scaler, args.total_steps, save_dir)

    # ── 导出对齐后的 BERT 权重 ──
    bert_out = save_dir / 'bert_contrastive'
    model.text_enc.bert.save_pretrained(str(bert_out))
    model.text_enc.bert.config.save_pretrained(str(bert_out))
    print(f'\nBERT 权重已导出 → {bert_out}')
    print(f'后续扩散训练：')
    print(f'  python -m node_diffusion_cross_att.train --bert {bert_out}')

    log_file.close()


if __name__ == '__main__':
    main()
