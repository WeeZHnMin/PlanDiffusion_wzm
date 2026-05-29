"""
第一阶段：纯图序列预训练
模型：LLaMA（随机初始化）
数据：data/processed/unified_dataset/graph_only.npz
词表：data/processed/unified_vocab/vocab_config.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM


# ── 数据集 ─────────────────────────────────────────────────────────
class GraphDataset(Dataset):
    def __init__(self, npz_path: Path):
        raw = np.load(npz_path)
        self.tokens  = raw["tokens"].astype(np.int32)   # (N, max_len)
        self.lengths = raw["lengths"].astype(np.int32)  # (N,)
        print(f"数据集: {len(self.tokens)} 条, 最大序列长度 {self.tokens.shape[1]}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = int(self.lengths[idx])
        return torch.tensor(self.tokens[idx, :length], dtype=torch.long)


def collate_fn(batch, pad_id: int):
    max_len = max(x.shape[0] for x in batch)
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask   = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        tokens[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return tokens, mask


# ── 工具 ───────────────────────────────────────────────────────────
def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> float:
    pred  = logits.argmax(dim=-1)
    valid = labels.ne(pad_id)
    return (pred.eq(labels) & valid).sum().item() / max(valid.sum().item(), 1)


# ── 主程序 ─────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=Path, default=Path("data/processed/unified_dataset/graph_only.npz"))
    p.add_argument("--vocab",      type=Path, default=Path("data/processed/unified_vocab/vocab_config.json"))
    p.add_argument("--save-dir",   type=Path, default=Path("checkpoints/stage1_graph"))
    p.add_argument("--batch-size", type=int,  default=24)
    p.add_argument("--max-epochs", type=int,  default=100)
    p.add_argument("--lr",         type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,  default=200)
    p.add_argument("--save-every", type=int,  default=2000)
    p.add_argument("--seed",       type=int,  default=42)
    # 模型配置
    p.add_argument("--hidden-size",    type=int, default=512)
    p.add_argument("--num-layers",     type=int, default=8)
    p.add_argument("--num-heads",      type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--max-position-embeddings", type=int, default=384)
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 词表配置
    cfg = json.loads(args.vocab.read_text(encoding="utf-8"))
    vocab_size = cfg["total_vocab_size"]
    pad_id     = cfg["PAD_ID"]
    bos_id     = cfg["BOS_ID"]
    eos_id     = cfg["EOS_ID"]
    print(f"词表大小: {vocab_size}, PAD={pad_id}, BOS={bos_id}, EOS={eos_id}")

    # 数据
    dataset = GraphDataset(args.data)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        drop_last=True, num_workers=0,
    )

    # 模型
    model_cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.1f}M")

    # 优化器 + 调度器
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = args.max_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for tokens, mask in loader:
            tokens = tokens.to(device)
            mask   = mask.to(device)
            x, y   = tokens[:, :-1], tokens[:, 1:]
            m      = mask[:, :-1]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=x, attention_mask=m).logits
                loss   = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                acc = masked_token_accuracy(logits.detach(), y, pad_id)
                print(f"epoch {epoch+1:3d} | step {global_step:6d} | "
                      f"loss {loss.item():.4f} | acc {acc:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % args.save_every == 0:
                ckpt = args.save_dir / f"step_{global_step:07d}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  saved -> {ckpt}")

        avg_loss = epoch_loss / len(loader)
        epoch_time = time.perf_counter() - t0
        print(f"=== epoch {epoch+1} | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s ===")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_dir / "best.pt")
            print(f"  best -> {args.save_dir}/best.pt | loss={best_loss:.4f}\n")

    print(f"训练完成，最优 loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
