"""
扩散模型训练，支持两种模式：

  MODE = 'graph_only'  只用图结构条件（邻接掩码），不需要文本
  MODE = 'text+graph'  文本 cross-attention + 图结构掩码联合训练

切换只需改下方 MODE 一行。
"""

import os
import argparse
import torch
from torch.optim import AdamW
from transformers import BertModel

from .model     import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .dataset   import load_node_data, load_node_text_data, load_hf_node_text_data

# ══════════════════════════════════════════════════════════════
MODE        = 'graph_only'  # 'graph_only' | 'text+graph'
DATA_SOURCE = 'local'       # 'local'      | 'hf'
# ══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument('--data_path',     default='data/processed/graph_tokens_combo_5w.npz')
    p.add_argument('--jsonl_path',    default='data/jsonl/mapped_type_data_zh.jsonl')
    p.add_argument('--bert_path',     default='models/bert-base-chinese')
    p.add_argument('--max_text_len',  type=int, default=64)
    # 训练
    p.add_argument('--save_dir',      default='checkpoints/node_diffusion')
    p.add_argument('--resume',        default='')
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--total_steps',   type=int,   default=200000)
    p.add_argument('--log_interval',  type=int,   default=100)
    p.add_argument('--save_interval', type=int,   default=10000)
    # 模型
    p.add_argument('--model_channels',type=int,   default=256)
    p.add_argument('--num_layers',    type=int,   default=6)
    p.add_argument('--num_heads',     type=int,   default=4)
    p.add_argument('--timesteps',     type=int,   default=1000)
    return p.parse_args()


def encode_text(bert, text_ids, text_mask, device):
    """BERT 编码，返回 last_hidden_state [B, L, 768]。"""
    with torch.no_grad():
        out = bert(input_ids=text_ids.to(device),
                   attention_mask=text_mask.to(device))
    return out.last_hidden_state


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"mode: {MODE}  device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 模型 ──────────────────────────────────────────────────
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    # ── 文本编码器（text+graph 模式才加载）───────────────────
    bert = None
    if MODE == 'text+graph':
        print(f'加载 BERT: {args.bert_path}')
        bert = BertModel.from_pretrained(args.bert_path).to(device).eval()
        for p in bert.parameters():
            p.requires_grad = False

    # ── 优化器 ────────────────────────────────────────────────
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    # ── 数据 ──────────────────────────────────────────────────
    if MODE == 'graph_only':
        data = load_node_data(args.data_path, args.batch_size)
    elif DATA_SOURCE == 'hf':
        data = load_hf_node_text_data(
            args.bert_path, args.batch_size, args.max_text_len,
        )
    else:
        data = load_node_text_data(
            args.data_path, args.jsonl_path, args.bert_path,
            args.batch_size, args.max_text_len,
        )

    # ── 训练循环 ──────────────────────────────────────────────
    model.train()
    running_loss = running_rmse = 0.0

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x = x.to(device)

        # 构建 model_kwargs
        model_kwargs = {
            'adj_matrix': cond['adj_matrix'].to(device),
            'node_mask':  cond['node_mask'].to(device),
        }

        if MODE == 'text+graph':
            model_kwargs['encoder_hidden'] = encode_text(
                bert, cond['text_ids'], cond['text_mask'], device
            )

        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
        loss, coord_rmse = diffusion.training_losses(model, x, t, model_kwargs)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()
        running_rmse += coord_rmse

        if step % args.log_interval == 0 and step > 0:
            avg      = running_loss / args.log_interval
            avg_rmse = running_rmse / args.log_interval
            running_loss = running_rmse = 0.0
            print(f"step {step:6d} | loss {avg:.4f} | coord_rmse {avg_rmse:.2f} px")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f'model_{step:07d}.pt')
            torch.save({'model': model.state_dict(),
                        'opt':   opt.state_dict(),
                        'step':  step}, path)
            print(f"  saved -> {path}")

    path = os.path.join(args.save_dir, f'model_{args.total_steps:07d}.pt')
    torch.save({'model': model.state_dict(),
                'opt':   opt.state_dict(),
                'step':  args.total_steps}, path)
    print(f"训练完成 ({MODE})  saved -> {path}")


if __name__ == '__main__':
    main()
