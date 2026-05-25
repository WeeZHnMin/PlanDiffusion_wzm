"""
Train NodeDiffusionTransformer on preprocessed node-coordinate data.

Usage:
    python -m node_diffusion.train \
        --data_path data/processed/nodes_train.npz \
        --save_dir checkpoints/node_diffusion \
        --batch_size 64 \
        --lr 1e-4 \
        --total_steps 200000 \
        --log_interval 100 \
        --save_interval 10000
"""

import os
import argparse
import torch
from torch.optim import AdamW

from .model     import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .dataset   import load_node_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',     default='data/processed/nodes_train.npz')
    p.add_argument('--save_dir',      default='checkpoints/node_diffusion')
    p.add_argument('--resume',        default='', help='path to checkpoint .pt')
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--total_steps',   type=int,   default=200000)
    p.add_argument('--log_interval',  type=int,   default=100)
    p.add_argument('--save_interval', type=int,   default=10000)
    p.add_argument('--model_channels',type=int,   default=256)
    p.add_argument('--num_layers',    type=int,   default=6)
    p.add_argument('--num_heads',     type=int,   default=4)
    p.add_argument('--timesteps',     type=int,   default=1000)
    return p.parse_args()


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    data = load_node_data(args.data_path, args.batch_size, shuffle=True)

    model.train()
    running_loss = 0.0

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x    = x.to(device)
        cond = move_cond(cond, device)

        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)

        loss = diffusion.training_losses(model, x, t, cond)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval if step > 0 else running_loss
            running_loss = 0.0
            print(f"step {step:6d} | loss {avg:.6f}")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f'model_{step:07d}.pt')
            torch.save({'model': model.state_dict(),
                        'opt':   opt.state_dict(),
                        'step':  step}, path)
            print(f"  saved → {path}")

    # final save
    path = os.path.join(args.save_dir, f'model_{args.total_steps:07d}.pt')
    torch.save({'model': model.state_dict(),
                'opt':   opt.state_dict(),
                'step':  args.total_steps}, path)
    print(f"training done. saved → {path}")


if __name__ == '__main__':
    main()
