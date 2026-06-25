"""
Train NodeDiffusionTransformer (三流注意力) on preprocessed node-coordinate data.

使用方法（单卡）：
  python -m node_diffusion_room.train \\
      --data_path data/processed/node_diffusion_room/graph_dataset_5k.npz \\
      --save_dir  checkpoints/node_diffusion_room/run1

多卡 DDP：
  torchrun --nproc_per_node=2 -m node_diffusion_room.train \\
      --data_path data/processed/node_diffusion_room/graph_dataset_5k.npz \\
      --save_dir  checkpoints/node_diffusion_room/run1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

from node_diffusion_cross_att.diffusion import GaussianDiffusion
from .dataset import load_node_data, NodeDataset
from .model import NodeDiffusionTransformer


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default=defaults.get("data_path", "data/processed/node_diffusion_room/graph_dataset_5k.npz"))
    parser.add_argument("--save_dir",     default=defaults.get("save_dir",  "checkpoints/node_diffusion_room"))
    parser.add_argument("--resume",       default="", help="path to checkpoint .pt")
    parser.add_argument("--batch_size",   type=int,   default=defaults.get("batch_size",   144))
    parser.add_argument("--lr",           type=float, default=defaults.get("lr",           1e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--total_steps",  type=int,   default=defaults.get("total_steps",  100000))
    parser.add_argument("--log_interval", type=int,   default=defaults.get("log_interval", 100))
    parser.add_argument("--save_interval",type=int,   default=defaults.get("save_interval",5000))
    parser.add_argument("--eval_interval",type=int,   default=defaults.get("eval_interval",5000),
                        help="每隔多少步计算一次 Room Invasion Rate")
    parser.add_argument("--eval_size",    type=int,   default=defaults.get("eval_size",    200),
                        help="固定评估子集大小（每次相同的 200 条）")
    parser.add_argument("--eval_batch_size", type=int, default=defaults.get("eval_batch_size", 16),
                        help="评估时的 batch size")
    parser.add_argument("--eval_seed",    type=int,   default=defaults.get("eval_seed",    0),
                        help="固定评估子集的随机种子")

    parser.add_argument("--model_channels",type=int,  default=defaults.get("model_channels",384))
    parser.add_argument("--num_layers",   type=int,   default=defaults.get("num_layers",   6))
    parser.add_argument("--num_heads",    type=int,   default=defaults.get("num_heads",    6))
    parser.add_argument("--timesteps",    type=int,   default=defaults.get("timesteps",    1000))
    parser.add_argument("--bert",         default=defaults.get("bert", "models/bert-base-uncased"))
    parser.add_argument("--unfreeze_layers", type=int, default=defaults.get("unfreeze_layers", 0))
    return parser


def move_cond(cond, device):
    return {k: v.to(device) for k, v in cond.items()}


# ── 完整 DDPM 采样（T=1000）────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, diffusion, cond, device):
    """
    标准 DDPM 反向采样（1000 步），返回预测坐标 [B, 2, N]。
    评估时 batch_size=16 以控制显存和时间。
    """
    B = cond['node_mask'].shape[0]
    N = cond['node_mask'].shape[1]
    x = torch.randn(B, 2, N, device=device)

    diffusion._to(device)
    T = diffusion.T

    for t_val in range(T - 1, -1, -1):
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        eps = model(x, t_tensor, **cond)

        alpha_t    = diffusion.alphas[t_val]
        abar_t     = diffusion.alphas_bar[t_val]
        post_var_t = diffusion.posterior_variance[t_val]

        coef = (1 - alpha_t) / (1 - abar_t).sqrt()
        mean = (x - coef * eps) / alpha_t.sqrt()

        if t_val > 0:
            x = mean + post_var_t.sqrt() * torch.randn_like(x)
        else:
            x = mean

    return x


# ── Room Invasion Rate ────────────────────────────────────────────────────────

def _room_invaded(coords_n2, room_ids_n):
    """
    coords_n2  : numpy [n, 2]，仅含有效节点
    room_ids_n : numpy [n]，int，0=无房间
    返回 True 如果有任意节点闯入其他房间的凸包。
    """
    from scipy.spatial import Delaunay

    unique_rooms = [r for r in set(room_ids_n.tolist()) if r > 0]
    if len(unique_rooms) < 2:
        return False

    # 为每个房间建凸包（至少需要3个节点）
    hulls = {}
    for rid in unique_rooms:
        pts = coords_n2[room_ids_n == rid]
        if len(pts) < 3:
            continue
        try:
            hulls[rid] = Delaunay(pts)
        except Exception:
            pass

    # 检测：其他房间的节点是否在此凸包内
    for rid_a, hull_a in hulls.items():
        other_pts = coords_n2[room_ids_n != rid_a]
        if hull_a.find_simplex(other_pts).max() >= 0:
            return True

    return False


@torch.no_grad()
def evaluate_rir(model, diffusion, eval_loader, device):
    """
    Room Invasion Rate：有节点侵入其他房间凸包的样本占比。
    越低越好，0 = 完全无侵入。使用完整 DDPM 1000 步推理，batch_size=16。
    """
    model.eval()
    invaded_count = 0
    total_count   = 0

    for x_batch, cond_batch in eval_loader:
        cond_batch = move_cond(cond_batch, device)

        pred = ddpm_sample(model, diffusion, cond_batch, device)
        # pred: [B, 2, N]  →  [B, N, 2]
        pred_np = pred.permute(0, 2, 1).cpu().numpy()

        node_mask_np = cond_batch['node_mask'].cpu().numpy()   # [B, N]
        room_ids_np  = cond_batch['room_ids'].cpu().numpy()    # [B, N]

        for b in range(pred_np.shape[0]):
            n = int(node_mask_np[b].sum())
            if n < 3:
                total_count += 1
                continue
            coords = pred_np[b, :n]
            rids   = room_ids_np[b, :n]
            if _room_invaded(coords, rids):
                invaded_count += 1
            total_count += 1

    model.train()
    return invaded_count / max(total_count, 1)


# ── 主训练循环 ─────────────────────────────────────────────────────────────────

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

    # 固定 eval 子集（只在 master 评估，种子固定保证每次相同 200 条）
    if is_master:
        eval_size = min(args.eval_size, len(dataset))
        rng_eval  = torch.Generator()
        rng_eval.manual_seed(args.eval_seed)
        eval_idx  = torch.randperm(len(dataset), generator=rng_eval)[:eval_size].tolist()
        eval_subset = Subset(dataset, eval_idx)
        eval_loader = DataLoader(eval_subset, batch_size=args.eval_batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print(f"eval 子集: {eval_size} 条样本（seed={args.eval_seed}），每 {args.eval_interval} 步评估一次（完整 DDPM 1000步）")

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

        # ── Room Invasion Rate 评估 ───────────────────────────────────────────
        if is_master and step > 0 and step % args.eval_interval == 0:
            eval_t0   = time.perf_counter()
            raw_model = model.module if use_ddp else model
            rir = evaluate_rir(raw_model, diffusion, eval_loader, device)
            eval_elapsed = time.perf_counter() - eval_t0
            print(f"  [eval] step {step:6d} | RIR={rir:.4f} "
                  f"({eval_size} 样本, DDPM 1000步, {eval_elapsed:.1f}s)")
            log_file.write(json.dumps({
                'step': step, 'rir': round(rir, 4),
                'eval_elapsed': round(eval_elapsed, 1),
            }) + '\n')
            model.train()
            t0 = time.perf_counter()   # 重置计时，排除 eval 耗时

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
