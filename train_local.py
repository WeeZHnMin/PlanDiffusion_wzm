"""
NodeDiffusion 本地训练脚本（无 HF 上传/下载）

用法：
  # 自动从本地 latest.pt 续训，没有则从头开始
  nohup python train_local.py > logs/train.log 2>&1 &

  # 指定 checkpoint
  nohup python train_local.py --resume checkpoints/node_diffusion_cross_att/latest.pt > logs/train.log 2>&1 &

  # 从头训练
  nohup python train_local.py --fresh > logs/train.log 2>&1 &
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from node_diffusion_cross_att.dataset import NodeDataset
from node_diffusion_cross_att.diffusion import GaussianDiffusion
from node_diffusion_cross_att.model import NodeDiffusionTransformer


def inf_loader(npz_path, batch_size, num_workers=4):
    ds = NodeDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True)
    while True:
        yield from loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",     default="data/processed/node_diffusion_cross_att/graph_dataset.npz")
    p.add_argument("--save_dir",      default="checkpoints/node_diffusion_cross_att")
    p.add_argument("--resume",        default="", help="指定本地 checkpoint 路径")
    p.add_argument("--fresh",         action="store_true", help="从头训练")
    p.add_argument("--bert",          default="bert-base-uncased")
    p.add_argument("--batch_size",    type=int,   default=384)
    p.add_argument("--total_steps",   type=int,   default=1000000)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--log_interval",  type=int,   default=100)
    p.add_argument("--save_interval", type=int,   default=1000)
    p.add_argument("--timesteps",     type=int,   default=1000)
    p.add_argument("--model_channels",type=int,   default=384)
    p.add_argument("--num_layers",    type=int,   default=6)
    p.add_argument("--num_heads",     type=int,   default=6)
    p.add_argument("--unfreeze_layers",type=int,  default=0)
    p.add_argument("--num_workers",   type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "latest.pt"
    log_path  = save_dir / "train_log.jsonl"
    if args.fresh:
        open(log_path, "w").close()  # 只有 --fresh 时才清空

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB", flush=True)

    # ── 确定 checkpoint ────────────────────────────────────────────────────────
    resume_path = ""
    if args.fresh:
        print("--fresh 模式：从头训练", flush=True)
    elif args.resume and os.path.exists(args.resume):
        resume_path = args.resume
        print(f"续训（指定）: {resume_path}", flush=True)
    elif ckpt_path.exists():
        resume_path = str(ckpt_path)
        print(f"续训（本地 latest.pt）: {resume_path}", flush=True)
    else:
        print("无本地 checkpoint，从头训练", flush=True)

    # ── 构建模型 ───────────────────────────────────────────────────────────────
    model = NodeDiffusionTransformer(
        model_channels  = args.model_channels,
        num_layers      = args.num_layers,
        num_heads       = args.num_heads,
        bert_name       = args.bert,
        unfreeze_layers = args.unfreeze_layers,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── 加载 checkpoint ────────────────────────────────────────────────────────
    start_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        raw_sd = ckpt["model"]
        if any(k.startswith("module.") for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        missing, unexpected = model.load_state_dict(raw_sd, strict=False)
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:3]}", flush=True)
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}", flush=True)
        if not unexpected:
            opt.load_state_dict(ckpt["opt"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}", flush=True)

    total_steps = args.total_steps
    if start_step >= total_steps:
        total_steps = start_step + 200000
        print(f"已达目标步数，续训至 {total_steps}", flush=True)

    # ── 训练 ───────────────────────────────────────────────────────────────────
    data = inf_loader(args.data_path, args.batch_size, args.num_workers)
    print(f"开始训练: step {start_step} → {total_steps}", flush=True)

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    model.train()
    running_loss = running_rmse = 0.0
    t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        x, cond = next(data)
        x    = x.to(device)
        cond = {k: v.to(device) for k, v in cond.items()}
        t    = torch.randint(0, args.timesteps, (x.shape[0],), device=device)

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, coord_rmse = diffusion.training_losses(model, x, t, cond)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        running_rmse += coord_rmse

        if step % args.log_interval == 0 and step > 0:
            n = args.log_interval
            avg_loss = running_loss / n
            avg_rmse = running_rmse / n
            running_loss = running_rmse = 0.0
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f"step {step:6d} | loss {avg_loss:.4f} | rmse {avg_rmse:.2f} | {elapsed:.1f}s", flush=True)
            log_file.write(json.dumps({
                "step": step, "loss": round(avg_loss, 4),
                "rmse": round(avg_rmse, 2), "elapsed": round(elapsed, 1),
            }) + "\n")

        if step > 0 and step % args.save_interval == 0:
            torch.save({
                "model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
            }, ckpt_path)
            log_file.flush()
            print(f"  saved → {ckpt_path}", flush=True)

    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "step": total_steps,
    }, ckpt_path)
    log_file.flush()
    log_file.close()
    print("训练完成", flush=True)


if __name__ == "__main__":
    main()
