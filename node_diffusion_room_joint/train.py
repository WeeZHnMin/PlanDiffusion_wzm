"""
Train NodeDiffusionTransformer（三流：adj_attn + room_attn + global_attn）。

需要先用 node_diffusion_room_tri/build_graph_npz.py 生成含 adj_matrix 的新 npz。

使用方法（单卡）：
  python -m node_diffusion_room_joint.train \\
      --data_path data/processed/node_diffusion_room_tri/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room_joint/run1

多卡 DDP：
  torchrun --nproc_per_node=2 -m node_diffusion_room_joint.train \\
      --data_path data/processed/node_diffusion_room_tri/graph_dataset.npz \\
      --save_dir  checkpoints/node_diffusion_room_joint/run1
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
from .eval_iou import coords_to_polys_by_type, compute_iou, center_at_origin

MAX_NODES    = 40
MAX_TEXT_LEN = 192


# ── DDIM 推理（验证用）────────────────────────────────────────────────────────

@torch.no_grad()
def _ddim_sample(model, diffusion, cond_batched, device, ddim_steps=200):
    diffusion._to(device)
    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()
    B  = next(iter(cond_batched.values())).shape[0]
    x  = torch.randn(B, 2, MAX_NODES, device=device)
    # 预计算文本特征，避免 BERT 在循环内反复前向传播
    text_feat, text_mask = model.encode_text(
        cond_batched['prompt_tokens'], cond_batched.get('prompt_mask'))
    extra = {k: v for k, v in cond_batched.items()
             if k not in ('prompt_tokens', 'prompt_mask')}
    extra['text_feat'] = text_feat
    extra['text_mask'] = text_mask
    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps  = model(x, t_tensor, **extra)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)
        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0
    return x   # [B, 2, MAX_NODES]


# ── 验证函数（IoU）───────────────────────────────────────────────────────────

def _run_val(model, diffusion, tokenizer, val_records, args, device, step, log_file):
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    sample = random.sample(val_records, min(args.val_n, len(val_records)))

    prepared = []
    for rec in sample:
        n = int(rec["n_nodes"])
        if n < 3:
            continue
        raw_coords = np.array(rec["node_coords"][:n], dtype=np.float32)
        adj_raw    = np.array(rec["adj_matrix"], dtype=np.int32)[:n, :n]
        np.fill_diagonal(adj_raw, 0)

        gt_node_types = [
            (t if isinstance(t, list) else [t])
            for t in rec["node_types"][:n]
        ]
        gt_centered = center_at_origin(raw_coords, np.ones(n, dtype=np.float32))
        adj_list    = adj_raw.tolist()
        gt_polys    = coords_to_polys_by_type(gt_centered, adj_list, gt_node_types, n)
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
            "mask_np": mask_np, "adj_pad": adj_pad, "membership": membership,
            "ptok": ptok, "pmsk": pmsk, "n": n,
            "adj_list": adj_list, "gt_polys": gt_polys, "gt_node_types": gt_node_types,
        })

    micro_list, macro_list = [], []
    BS = args.val_batch
    t0 = time.perf_counter()
    for bi in range(0, len(prepared), BS):
        chunk = prepared[bi: bi + BS]
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
            s       = chunk[j]
            n       = s["n"]
            pred_np = pred_xy[j].cpu().numpy().T          # [MAX_NODES, 2]
            pred_cen = center_at_origin(pred_np, s["mask_np"])
            pred_polys = coords_to_polys_by_type(
                pred_cen[:n], s["adj_list"], s["gt_node_types"], n)
            micro, macro = compute_iou(s["gt_polys"], pred_polys)
            micro_list.append(micro)
            macro_list.append(macro)

    micro_iou = float(np.mean(micro_list)) if micro_list else 0.0
    macro_iou = float(np.mean(macro_list)) if macro_list else 0.0
    elapsed   = time.perf_counter() - t0
    print(f"[val step {step:6d}] n={len(micro_list)} | "
          f"micro_iou={micro_iou:.4f}  macro_iou={macro_iou:.4f} | {elapsed:.1f}s")
    log_file.write(json.dumps({'step': step,
                               'micro_iou': round(micro_iou, 4),
                               'macro_iou': round(macro_iou, 4),
                               'elapsed_val': round(elapsed, 1)}) + '\n')

    raw_model.train()
    return micro_iou


# ── 训练入口 ──────────────────────────────────────────────────────────────────

def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default=defaults.get("data_path", "data/processed/node_diffusion_room_tri/graph_dataset.npz"))
    parser.add_argument("--save_dir",     default=defaults.get("save_dir",  "checkpoints/node_diffusion_room_joint"))
    parser.add_argument("--resume",       default="", help="path to checkpoint .pt")
    parser.add_argument("--batch_size",   type=int,   default=defaults.get("batch_size",   384))
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
    parser.add_argument("--large_node_weight",    type=float, default=defaults.get("large_node_weight",    1.0),
                        help="节点数>=阈值的样本权重倍数，1.0=不启用")
    parser.add_argument("--large_node_threshold", type=int,   default=defaults.get("large_node_threshold", 23))
    parser.add_argument("--val_jsonl",    default=defaults.get("val_jsonl",    ""),
                        help="验证集 jsonl 路径，留空则不做验证")
    parser.add_argument("--val_interval", type=int, default=defaults.get("val_interval", 2000))
    parser.add_argument("--val_n",        type=int, default=defaults.get("val_n",        224))
    parser.add_argument("--ddim_steps",   type=int, default=defaults.get("ddim_steps",   200))
    parser.add_argument("--val_batch",    type=int, default=defaults.get("val_batch",    16))
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

    use_amp  = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler   = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── 预加载验证集（仅 master）─────────────────────────────────────────────
    val_records   = []
    val_tokenizer = None
    if is_master and args.val_jsonl:
        with open(args.val_jsonl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    val_records.append(json.loads(line))
        print(f"验证集: {len(val_records)} 条，每 {args.val_interval} 步随机采 {args.val_n} 条，DDIM {args.ddim_steps} 步")
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
        data = load_node_data(dataset, args.batch_size, shuffle=True,
                              large_node_weight=args.large_node_weight,
                              large_node_threshold=args.large_node_threshold)

    model.train()
    running_loss = running_coord = running_centroid = running_rmse = 0.0
    best_val_iou = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, args.total_steps):
        x, cond = next(data)
        x    = x.to(device)
        cond = move_cond(cond, device)

        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, coord_loss, centroid_loss, coord_rmse = diffusion.training_losses(model, x, t, cond, step=step)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        running_loss     += loss.item()
        running_coord    += coord_loss.item()
        running_centroid += centroid_loss.item()
        running_rmse     += coord_rmse

        if is_master and step % args.log_interval == 0 and step > 0:
            n            = args.log_interval
            avg_loss     = running_loss     / n
            avg_coord    = running_coord    / n
            avg_centroid = running_centroid / n
            avg_rmse     = running_rmse     / n
            running_loss = running_coord = running_centroid = running_rmse = 0.0
            elapsed  = time.perf_counter() - t0
            t0       = time.perf_counter()

            print(f"step {step:6d} | loss {avg_loss:.4f} | coord {avg_coord:.4f} | centroid {avg_centroid:.4f} | rmse {avg_rmse:.2f} px | {elapsed:.1f}s")
            log_file.write(json.dumps({
                'step': step, 'loss': round(avg_loss, 4),
                'coord_loss': round(avg_coord, 4),
                'centroid_loss': round(avg_centroid, 4),
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

        if is_master and val_records and step > 0 and step % args.val_interval == 0:
            micro_iou = _run_val(model, diffusion, val_tokenizer, val_records,
                                 args, device, step, log_file)
            if micro_iou > best_val_iou:
                best_val_iou = micro_iou
                raw_model = model.module if use_ddp else model
                best_path = save_dir / "best.pt"
                torch.save({
                    "model": raw_model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "micro_iou": micro_iou,
                }, best_path)
                print(f"  best model saved (micro_iou={micro_iou:.4f}) -> {best_path}")

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
