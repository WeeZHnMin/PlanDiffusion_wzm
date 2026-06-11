"""
Train NodeDiffusionTransformer on preprocessed node-coordinate data.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

from .dataset import load_node_data
from .diffusion import GaussianDiffusion
from .model import NodeDiffusionTransformer


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=defaults.get("data_path", "data/processed/node_diffusion/graph_dataset.npz"))
    parser.add_argument("--save_dir", default=defaults.get("save_dir", "checkpoints/node_diffusion"))
    parser.add_argument("--resume", default="", help="path to checkpoint .pt")
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--total_steps", type=int, default=defaults.get("total_steps", 200000))
    parser.add_argument("--log_interval", type=int, default=defaults.get("log_interval", 100))
    parser.add_argument("--save_interval", type=int, default=defaults.get("save_interval", 10000))
    parser.add_argument("--model_channels", type=int, default=defaults.get("model_channels", 384))
    parser.add_argument("--num_layers", type=int, default=defaults.get("num_layers", 6))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 6))
    parser.add_argument("--timesteps", type=int, default=defaults.get("timesteps", 1000))
    parser.add_argument("--bert", default=defaults.get("bert", "models/bert-base-uncased"))
    parser.add_argument("--unfreeze_layers", type=int, default=defaults.get("unfreeze_layers", 0),
                        help="BERT 最后几层解冻参与训练（0=全冻结）")
    parser.add_argument("--ablation", default="", choices=["", "no_text", "no_graph"],
                        help="消融变体: no_text=去掉文本条件, no_graph=去掉图结构条件")
    return parser


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / 'log.jsonl'
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
    print(f'日志: {log_path}')

    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
        unfreeze_layers=args.unfreeze_layers,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    # BERT 已冻结，只优化可训练参数
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_step = 0
    if args.resume:
        ckpt   = torch.load(args.resume, map_location=device)
        raw_sd = ckpt["model"]
        # 兼容 Kaggle DataParallel checkpoint（去掉 module. 前缀）
        if any(k.startswith('module.') for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        missing, unexpected = model.load_state_dict(raw_sd, strict=False)
        if missing:
            print(f"  missing keys (new params): {missing}")
        if unexpected:
            print(f"  unexpected keys (dropped): {unexpected}")
        opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}")

    data = load_node_data(args.data_path, args.batch_size, shuffle=True)

    model.train()
    running_loss = running_rmse = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x = x.to(device)
        cond = move_cond(cond, device)

        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, coord_rmse = diffusion.training_losses(model, x, t, cond)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        running_rmse += coord_rmse

        if step % args.log_interval == 0 and step > 0:
            n        = args.log_interval
            avg_loss = running_loss / n
            avg_rmse = running_rmse / n
            running_loss = running_rmse = 0.0
            elapsed  = time.perf_counter() - t0
            t0       = time.perf_counter()

            print(f"step {step:6d} | loss {avg_loss:.4f} | coord_rmse {avg_rmse:.2f} px | {elapsed:.1f}s")
            log_file.write(json.dumps({
                'step': step, 'loss': round(avg_loss, 4),
                'coord_rmse': round(avg_rmse, 2),
                'elapsed': round(elapsed, 1),
            }) + '\n')

        if step > 0 and step % args.save_interval == 0:
            ckpt_path = save_dir / f"model_{step:07d}.pt"
            torch.save({
                "model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
            }, ckpt_path)
            print(f"  saved -> {ckpt_path}")

    ckpt_path = save_dir / f"model_{args.total_steps:07d}.pt"
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "step": args.total_steps,
    }, ckpt_path)
    log_file.close()
    print(f"training done. saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
