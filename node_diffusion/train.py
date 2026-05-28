"""
Train NodeDiffusionTransformer on preprocessed node-coordinate data.
"""

import argparse
import os

import torch
from torch.optim import AdamW

from .dataset import load_node_data
from .diffusion import GaussianDiffusion
from .model import NodeDiffusionTransformer


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=defaults.get("data_path", "data/processed/nodes_train.npz"))
    parser.add_argument("--save_dir", default=defaults.get("save_dir", "checkpoints/node_diffusion"))
    parser.add_argument("--resume", default="", help="path to checkpoint .pt")
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--total_steps", type=int, default=defaults.get("total_steps", 200000))
    parser.add_argument("--log_interval", type=int, default=defaults.get("log_interval", 100))
    parser.add_argument("--save_interval", type=int, default=defaults.get("save_interval", 10000))
    parser.add_argument("--model_channels", type=int, default=defaults.get("model_channels", 256))
    parser.add_argument("--num_layers", type=int, default=defaults.get("num_layers", 6))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 4))
    parser.add_argument("--timesteps", type=int, default=defaults.get("timesteps", 1000))
    return parser


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}")

    data = load_node_data(args.data_path, args.batch_size, shuffle=True)

    model.train()
    running_loss = 0.0
    running_coord_rmse = 0.0

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x = x.to(device)
        cond = move_cond(cond, device)

        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
        loss, coord_rmse = diffusion.training_losses(model, x, t, cond)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()
        running_coord_rmse += coord_rmse

        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval if step > 0 else running_loss
            avg_rmse = running_coord_rmse / args.log_interval if step > 0 else running_coord_rmse
            running_loss = 0.0
            running_coord_rmse = 0.0
            print(f"step {step:6d} | loss {avg:.4f} | coord_rmse {avg_rmse:.2f} px")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"model_{step:07d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)
            print(f"  saved -> {path}")

    path = os.path.join(args.save_dir, f"model_{args.total_steps:07d}.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": args.total_steps}, path)
    print(f"training done. saved -> {path}")


if __name__ == "__main__":
    main()
