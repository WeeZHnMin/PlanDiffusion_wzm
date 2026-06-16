"""
NodeDiffusion 服务器训练脚本（双流 dual-stream 变体）

用法：
  # 首次训练（自动从 HF 拉取权重续训，拉不到则从头开始）
  nohup python train_node_diffusion.py > logs/train.log 2>&1 &

  # 指定参数
  nohup python train_node_diffusion.py \
      --data_path data/processed/node_diffusion_cross_att/graph_dataset_6k.npz \
      --save_dir  checkpoints/node_diffusion_cross_att \
      --batch_size 128 \
      --total_steps 250000 \
      > logs/train.log 2>&1 &

  # 从本地 checkpoint 续训
  nohup python train_node_diffusion.py --resume checkpoints/node_diffusion_cross_att/latest.pt \
      > logs/train.log 2>&1 &
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from node_diffusion_cross_att.dataset import NodeDataset
from node_diffusion_cross_att.diffusion import GaussianDiffusion
from node_diffusion_cross_att.model import NodeDiffusionTransformer
from torch.utils.data import DataLoader


# ── 无限 DataLoader ────────────────────────────────────────────────────────────
def inf_loader(npz_path, batch_size, num_workers=4):
    ds = NodeDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True)
    while True:
        yield from loader


# ── HF 异步上传 ────────────────────────────────────────────────────────────────
# 同一时刻只允许一个上传线程运行，避免并发写同一文件
_hf_upload_lock = threading.Lock()
_hf_active_thread: threading.Thread | None = None


def _hf_push_worker(ckpt_path, log_path, step, repo_id, token):
    with _hf_upload_lock:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            # 确保 repo 存在，exist_ok=True 避免重复创建报错
            api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
            # 上传 checkpoint（直接覆盖，无需删库）
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo="latest.pt",
                repo_id=repo_id,
                commit_message=f"step {step}",
            )
            if os.path.exists(log_path):
                api.upload_file(
                    path_or_fileobj=str(log_path),
                    path_in_repo="train_log.jsonl",
                    repo_id=repo_id,
                    commit_message=f"log step {step}",
                )
            print(f"  [HF] step={step} → latest.pt + train_log.jsonl 已上传", flush=True)
        except Exception as e:
            print(f"  [HF] 上传失败: {e}", flush=True)


def push_to_hf_async(ckpt_path, log_path, step, repo_id, token):
    if not repo_id or not token:
        return
    global _hf_active_thread
    # 若上一次上传还在跑，跳过本次（不堆积队列）
    if _hf_active_thread is not None and _hf_active_thread.is_alive():
        print(f"  [HF] step={step} 上传跳过（上次仍在进行中）", flush=True)
        return
    _hf_active_thread = threading.Thread(
        target=_hf_push_worker,
        args=(ckpt_path, log_path, step, repo_id, token),
        daemon=False,  # 非 daemon，主进程退出时等待上传完成
    )
    _hf_active_thread.start()


# ── 从 HF 拉取 checkpoint ──────────────────────────────────────────────────────
def pull_from_hf(repo_id, token, save_dir):
    if not repo_id or not token:
        return None
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=repo_id,
            filename="latest.pt",
            token=token,
            local_dir=str(save_dir),
            force_download=True,
        )
        print(f"  [HF] 拉取权重成功: {local}", flush=True)
        return local
    except Exception as e:
        print(f"  [HF] 拉取权重失败（将从头训练）: {e}", flush=True)
        return None


# ── 参数解析 ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    default="data/processed/node_diffusion_cross_att/graph_dataset.npz")
    p.add_argument("--save_dir",     default="checkpoints/node_diffusion_cross_att")
    p.add_argument("--resume",       default="",  help="本地 checkpoint 路径（优先于 HF 拉取）")
    p.add_argument("--bert",         default="bert-base-uncased")
    p.add_argument("--hf_repo",      default="wzmmmm/plandiff-double-cross-6k")
    p.add_argument("--hf_token",     default="",  help="HF token（也可用 HF_TOKEN 环境变量）")
    p.add_argument("--batch_size",   type=int,   default=768)
    p.add_argument("--total_steps",  type=int,   default=1000000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int,   default=100)
    p.add_argument("--save_interval",  type=int,   default=1000)
    p.add_argument("--upload_interval",type=int,   default=2000,  help="每隔多少步上传一次 HF")
    p.add_argument("--timesteps",    type=int,   default=1000)
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",   type=int,   default=6)
    p.add_argument("--num_heads",    type=int,   default=6)
    p.add_argument("--unfreeze_layers", type=int, default=0)
    p.add_argument("--num_workers",  type=int,   default=4)
    return p.parse_args()


# ── 主训练逻辑 ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # HF token：命令行 > 环境变量
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "latest.pt"
    log_path  = save_dir / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB", flush=True)

    # ── 确定 checkpoint 来源 ───────────────────────────────────────────────────
    resume_path = ""
    if args.resume and os.path.exists(args.resume):
        resume_path = args.resume
        print(f"续训（命令行指定）: {resume_path}", flush=True)
    elif ckpt_path.exists():
        resume_path = str(ckpt_path)
        print(f"续训（本地 latest.pt）: {resume_path}", flush=True)
    else:
        print("本地无 checkpoint，尝试从 HF 拉取 ...", flush=True)
        pulled = pull_from_hf(args.hf_repo, hf_token, save_dir)
        if pulled:
            resume_path = pulled

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

    # ── 数据 ───────────────────────────────────────────────────────────────────
    data = inf_loader(args.data_path, args.batch_size, args.num_workers)
    print(f"开始训练: step {start_step} → {total_steps}", flush=True)

    # ── 训练循环 ───────────────────────────────────────────────────────────────
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

        if step % args.log_interval == 0 and step > start_step:
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
            if step % args.upload_interval == 0:
                push_to_hf_async(ckpt_path, log_path, step, args.hf_repo, hf_token)

    # ── 训练完成 ───────────────────────────────────────────────────────────────
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "step": total_steps,
    }, ckpt_path)
    log_file.flush()
    log_file.close()
    push_to_hf_async(ckpt_path, log_path, total_steps, args.hf_repo, hf_token)
    print("训练完成", flush=True)
    # 等待最后一次上传线程结束（非 daemon，会自然阻塞到完成）
    if _hf_active_thread is not None and _hf_active_thread.is_alive():
        print("等待最后一次 HF 上传完成 ...", flush=True)
        _hf_active_thread.join()


if __name__ == "__main__":
    main()
