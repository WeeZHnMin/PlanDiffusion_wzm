"""
Train NodeDiffusionTransformer (room_attn+global_attn) on preprocessed node-coordinate data.

使用方法（单卡）：
  python -m node_diffusion_room.train \\
      --data_path data/processed/node_diffusion_room/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room/run1

多卡 DDP：
  torchrun --nproc_per_node=2 -m node_diffusion_room.train \\
      --data_path data/processed/node_diffusion_room/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room/run1
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

from node_diffusion_cross_att.diffusion import GaussianDiffusion
from .dataset import load_node_data, NodeDataset
from .model import NodeDiffusionTransformer


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default=defaults.get("data_path", "data/processed/node_diffusion_room/graph_dataset.npz"))
    parser.add_argument("--save_dir",     default=defaults.get("save_dir",  "checkpoints/node_diffusion_room"))
    parser.add_argument("--resume",       default="", help="path to checkpoint .pt")
    parser.add_argument("--batch_size",   type=int,   default=defaults.get("batch_size",   512))
    parser.add_argument("--lr",           type=float, default=defaults.get("lr",           3e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--total_steps",  type=int,   default=defaults.get("total_steps",  350000))
    parser.add_argument("--log_interval", type=int,   default=defaults.get("log_interval", 100))
    parser.add_argument("--save_interval",type=int,   default=defaults.get("save_interval",5000))
    parser.add_argument("--model_channels",type=int,  default=defaults.get("model_channels",384))
    parser.add_argument("--num_layers",   type=int,   default=defaults.get("num_layers",   6))
    parser.add_argument("--num_heads",    type=int,   default=defaults.get("num_heads",    6))
    parser.add_argument("--timesteps",    type=int,   default=defaults.get("timesteps",    1000))
    parser.add_argument("--bert",         default=defaults.get("bert", "models/bert-base-uncased"))
    parser.add_argument("--unfreeze_layers", type=int, default=defaults.get("unfreeze_layers", 0))
    return parser


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    use_ddp    = local_rank >= 0
    if use_ddp:
        dist.init_process_group(backend='nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_master = (rank == 0)
    if is_master:
        print(f"device: {device}  world_size: {world_size}  ddp: {use_ddp}")

    save_dir = Path(args.save_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / 'log.jsonl'
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1) if is_master else None
    if is_master:
        print(f'日志: {log_path}')

    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
        unfreeze_layers=args.unfreeze_layers,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    no_decay = {'bias', 'norm', 'LayerNorm'}
    opt = AdamW(
        [
            {'params': [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ],
        lr=args.lr,
    )

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_step = 0
    if args.resume:
        ckpt   = torch.load(args.resume, map_location=device)
        raw_sd = ckpt["model"]
        if any(k.startswith('module.') for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        missing, unexpected = model.load_state_dict(raw_sd, strict=False)
        if is_master:
            if missing:
                print(f"  missing keys: {missing}")
            if unexpected:
                print(f"  unexpected keys: {unexpected}")
        opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        if is_master:
            print(f"resumed from step {start_step}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset = NodeDataset(args.data_path)
    if is_master:
        steps_per_epoch = len(dataset) / args.batch_size
        print(f"数据集: {len(dataset)} 条，batch={args.batch_size}，"
              f"~{steps_per_epoch:.0f} steps/epoch，"
              f"总 {args.total_steps} 步 ≈ {args.total_steps / steps_per_epoch:.1f} epoch")

    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True, drop_last=True)
        data = load_node_data(dataset, args.batch_size, sampler=sampler)
    else:
        data = load_node_data(dataset, args.batch_size, shuffle=True)

    model.train()
    running_loss = running_rmse = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x    = x.to(device)
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

        if is_master and step % args.log_interval == 0 and step > 0:
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

        if is_master and step > 0 and step % args.save_interval == 0:
            ckpt_path = save_dir / "latest.pt"
            raw_model = model.module if use_ddp else model
            torch.save({
                "model": raw_model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
            }, ckpt_path)
            print(f"  saved -> {ckpt_path}")

    if is_master:
        ckpt_path = save_dir / "latest.pt"
        raw_model = model.module if use_ddp else model
        torch.save({
            "model": raw_model.state_dict(), "opt": opt.state_dict(),
            "scaler": scaler.state_dict(), "step": args.total_steps,
        }, ckpt_path)
        log_file.close()
        print(f"training done. saved -> {ckpt_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
