"""
Train NodeDiffusionTransformer（三流：adj_attn + room_attn + global_attn）。

需要先用 node_diffusion_room_tri/build_graph_npz.py 生成含 adj_matrix 的新 npz。

使用方法（单卡）：
  python -m node_diffusion_room_tri.train \\
      --data_path data/processed/node_diffusion_room_tri/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room_tri/run1

多卡 DDP：
  torchrun --nproc_per_node=2 -m node_diffusion_room_tri.train \\
      --data_path data/processed/node_diffusion_room_tri/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room_tri/run1
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import BertTokenizer

from .diffusion import GaussianDiffusion
from .dataset import load_node_data, NodeDataset
from .model import NodeDiffusionTransformer, _assign_room_membership_single, MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192


# ── DDIM 推理（验证用）────────────────────────────────────────────────────────

@torch.no_grad()
def _ddim_sample(model, diffusion, cond_batched, device, ddim_steps=200):
    """
    DDIM 确定性采样（eta=0），等间隔跳步，支持批次推理。

    cond_batched : dict，所有张量形状 [B, ...]
    返回         : [B, 2, MAX_NODES] float32 tensor
    """
    diffusion._to(device)
    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()

    B = next(iter(cond_batched.values())).shape[0]
    x = torch.randn(B, 2, MAX_NODES, device=device)

    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps  = model(x, t_tensor, **cond_batched)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)

        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0

    return x   # [B, 2, MAX_NODES]


# ── 验证函数 ──────────────────────────────────────────────────────────────────

def _run_val(model, diffusion, tokenizer, val_records, args, device, step, log_file):
    """
    从 val_records 随机采 val_n 条，批次 DDIM 推理，计算 micro/macro IoU。
    """
    from .eval_iou import (
        coords_to_polys_by_type,
        compute_iou,
        center_at_origin,
    )

    sample_recs = random.sample(val_records, min(args.val_n, len(val_records)))

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    # ── 第一步：预处理所有样本（CPU） ────────────────────────────────────────
    prepared = []
    for rec in sample_recs:
        n = int(rec["n_nodes"])
        if n < 3:
            continue

        raw_coords = np.array(rec["node_coords"][:n], dtype=np.float32)
        adj_raw    = np.array(rec["adj_matrix"],       dtype=np.int32)[:n, :n]
        np.fill_diagonal(adj_raw, 0)
        node_types = [
            (t if isinstance(t, list) else [t])
            for t in rec["node_types"][:n]
        ]

        gt_centered = center_at_origin(raw_coords, np.ones(n))
        adj_list    = adj_raw.tolist()
        gt_polys    = coords_to_polys_by_type(gt_centered, adj_list, node_types, n)
        if not gt_polys:
            continue

        mask_np    = np.zeros(MAX_NODES, dtype=np.float32); mask_np[:n] = 1.0
        adj_pad    = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        adj_pad[:n, :n] = adj_raw.astype(np.float32)

        membership = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
        membership[:n] = _assign_room_membership_single(adj_raw.astype(bool), n)

        prompt = rec.get("prompt", "").replace("\n", " ").strip()
        enc  = tokenizer(prompt, add_special_tokens=True,
                         max_length=MAX_TEXT_LEN, padding="max_length", truncation=True)
        ptok = np.array(enc["input_ids"],      dtype=np.int64)
        pmsk = np.array(enc["attention_mask"], dtype=np.float32)

        prepared.append({
            "mask_np":    mask_np,
            "adj_pad":    adj_pad,
            "membership": membership,
            "ptok":       ptok,
            "pmsk":       pmsk,
            "n":          n,
            "adj_list":   adj_list,
            "node_types": node_types,
            "gt_polys":   gt_polys,
            "mask_np_ref": mask_np,
        })

    t0 = time.perf_counter()
    print(f"[val step {step}] 批次推理 {len(prepared)} 条（DDIM {args.ddim_steps} 步，"
          f"batch={args.val_batch}）...", flush=True)

    # ── 第二步：批次 DDIM 推理 ────────────────────────────────────────────────
    all_pred_np = []
    VB = args.val_batch
    for bi in range(0, len(prepared), VB):
        chunk = prepared[bi: bi + VB]
        B = len(chunk)

        cond_b = {
            "node_mask":       torch.from_numpy(np.stack([s["mask_np"]    for s in chunk])).to(device),
            "room_membership": torch.from_numpy(np.stack([s["membership"] for s in chunk])).to(device),
            "adj_matrix":      torch.from_numpy(np.stack([s["adj_pad"]    for s in chunk])).to(device),
            "prompt_tokens":   torch.from_numpy(np.stack([s["ptok"]       for s in chunk])).to(device),
            "prompt_mask":     torch.from_numpy(np.stack([s["pmsk"]       for s in chunk])).to(device),
        }

        pred_xy = _ddim_sample(raw_model, diffusion, cond_b, device, args.ddim_steps)
        for j in range(B):
            all_pred_np.append(pred_xy[j].cpu().numpy().T)   # [MAX_NODES, 2]

        done = min(bi + VB, len(prepared))
        elapsed = time.perf_counter() - t0
        print(f"  [{done}/{len(prepared)}]  {elapsed:.1f}s", flush=True)

    raw_model.train()

    # ── 第三步：逐样本计算 IoU ────────────────────────────────────────────────
    micro_list, macro_list = [], []
    for s, pred_np in zip(prepared, all_pred_np):
        pred_cen   = center_at_origin(pred_np, s["mask_np_ref"])
        pred_polys = coords_to_polys_by_type(pred_cen[:s["n"]], s["adj_list"], s["node_types"], s["n"])
        micro, macro = compute_iou(s["gt_polys"], pred_polys)
        micro_list.append(micro)
        macro_list.append(macro)

    if not micro_list:
        print(f"[val step {step}] 所有样本均被跳过，无法计算 IoU")
        return None

    avg_micro = float(np.mean(micro_list))
    avg_macro = float(np.mean(macro_list))
    elapsed   = time.perf_counter() - t0
    print(f"[val step {step:6d}] n={len(micro_list)} | "
          f"micro_iou={avg_micro:.4f} | macro_iou={avg_macro:.4f} | {elapsed:.1f}s")
    if log_file:
        log_file.write(json.dumps({
            'step': step,
            'val_micro_iou': round(avg_micro, 6),
            'val_macro_iou': round(avg_macro, 6),
            'val_n': len(micro_list),
        }) + '\n')
    return avg_micro


# ── 训练入口 ──────────────────────────────────────────────────────────────────

def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default=defaults.get("data_path", "data/processed/node_diffusion_room_tri/graph_dataset.npz"))
    parser.add_argument("--save_dir",     default=defaults.get("save_dir",  "checkpoints/node_diffusion_room_tri"))
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
    # ── 验证参数 ────────────────────────────────────────────────────────────
    parser.add_argument("--val_jsonl",    default=defaults.get("val_jsonl",    "data/jsonl/val_graph_dataset_18k5.jsonl"),
                        help="验证集 jsonl 路径（空则跳过验证）")
    parser.add_argument("--val_interval", type=int, default=defaults.get("val_interval", 5000),
                        help="每隔多少步做一次验证")
    parser.add_argument("--val_n",        type=int, default=defaults.get("val_n",        224),
                        help="每次验证随机采样的样本数")
    parser.add_argument("--ddim_steps",   type=int, default=defaults.get("ddim_steps",   200),
                        help="DDIM 推理步数")
    parser.add_argument("--val_batch",    type=int, default=defaults.get("val_batch",    16),
                        help="验证推理批次大小")
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
        try:
            opt.load_state_dict(ckpt["opt"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
        except ValueError:
            print("  [warn] optimizer state 结构不兼容，跳过 opt 恢复，仅加载模型权重")
        start_step = ckpt["step"] + 1
        if is_master:
            print(f"resumed from step {start_step}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── 预加载验证集（仅 master）─────────────────────────────────────────────
    val_records   = []
    val_tokenizer = None
    if is_master and args.val_jsonl:
        with open(args.val_jsonl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    val_records.append(json.loads(line))
        print(f"验证集: {len(val_records)} 条记录，每 {args.val_interval} 步随机采 {args.val_n} 条验证")
        val_tokenizer = BertTokenizer.from_pretrained(args.bert)

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
    best_micro_iou = -1.0
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

        # ── 验证（每 val_interval 步，仅 master）────────────────────────────
        if is_master and val_records and step > 0 and step % args.val_interval == 0:
            micro = _run_val(model, diffusion, val_tokenizer, val_records,
                             args, device, step, log_file)
            if micro is not None and micro > best_micro_iou:
                best_micro_iou = micro
                raw_model = model.module if use_ddp else model
                best_path = save_dir / "best.pt"
                torch.save({
                    "model": raw_model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "micro_iou": micro,
                }, best_path)
                print(f"  best model saved (micro_iou={micro:.4f}) -> {best_path}")

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
