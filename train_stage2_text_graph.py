"""
第二阶段：文本条件图生成微调
模型：从第一阶段checkpoint初始化的LLaMA
数据：data/processed/unified_dataset/text_graph.npz
序列格式：text_tokens... BOS_G node1 type1 < nbr > ... EOS_G
Loss：只计算图序列部分（BOS_G之后）
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
class TextGraphDataset(Dataset):
    def __init__(self, npz_path: Path):
        raw = np.load(npz_path)
        self.tokens    = raw["tokens"].astype(np.int32)    # (N, max_len)
        self.lengths   = raw["lengths"].astype(np.int32)   # (N,)
        self.text_lens = raw["text_lens"].astype(np.int32) # (N,) 文本部分长度
        print(f"数据集: {len(self.tokens)} 条, "
              f"最大序列长度 {self.tokens.shape[1]}, "
              f"文本长度 min={self.text_lens.min()} max={self.text_lens.max()} "
              f"mean={self.text_lens.mean():.1f}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length   = int(self.lengths[idx])
        text_len = int(self.text_lens[idx])
        return (
            torch.tensor(self.tokens[idx, :length], dtype=torch.long),
            text_len,
        )


def collate_fn(batch, pad_id: int):
    seqs      = [b[0] for b in batch]
    text_lens = [b[1] for b in batch]
    max_len   = max(s.shape[0] for s in seqs)

    tokens = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask   = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        tokens[i, :len(seq)] = seq
        mask[i, :len(seq)]   = 1

    return tokens, mask, torch.tensor(text_lens, dtype=torch.long)


def make_labels(tokens: torch.Tensor, text_lens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    y = tokens shifted right by 1（teacher forcing）
    文本部分（前 text_lens[i]-1 个位置）label 设为 -100，不参与 loss
    从 text_lens[i]-1 开始（预测 BOS_G）计算 loss
    """
    B, L = tokens.shape
    y = tokens[:, 1:].clone()  # (B, L-1)

    for i in range(B):
        text_mask_len = max(0, int(text_lens[i]) - 1)
        if text_mask_len > 0:
            y[i, :text_mask_len] = -100

    # pad位置也设为-100（CrossEntropyLoss ignore_index=pad_id处理，但双保险）
    y[y == pad_id] = -100
    return y


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred  = logits.argmax(dim=-1)
    valid = labels.ne(-100)
    return (pred.eq(labels) & valid).sum().item() / max(valid.sum().item(), 1)


# ── 主程序 ─────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=Path, default=Path("data/processed/unified_dataset/text_graph.npz"))
    p.add_argument("--vocab",      type=Path, default=Path("data/processed/unified_vocab/vocab_config.json"))
    p.add_argument("--stage1-ckpt",type=Path, default=Path("checkpoints/stage1_graph/best.pt"))
    p.add_argument("--save-dir",   type=Path, default=Path("checkpoints/stage2_text_graph"))
    p.add_argument("--batch-size", type=int,  default=24)
    p.add_argument("--max-epochs", type=int,  default=20)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float,default=0.01)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,  default=200)
    p.add_argument("--save-every", type=int,  default=2000)
    p.add_argument("--seed",       type=int,  default=42)
    # 模型配置（需与第一阶段一致）
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

    cfg        = json.loads(args.vocab.read_text(encoding="utf-8"))
    vocab_size = cfg["total_vocab_size"]
    pad_id     = cfg["PAD_ID"]
    bos_id     = cfg["BOS_ID"]
    eos_id     = cfg["EOS_ID"]

    dataset = TextGraphDataset(args.data)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        drop_last=True, num_workers=0,
    )

    # 模型（与第一阶段相同配置）
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

    # 从第一阶段加载权重
    if args.stage1_ckpt.exists():
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"stage1 ckpt loaded | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"警告: stage1 checkpoint 不存在 {args.stage1_ckpt}，从零初始化")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = args.max_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    use_amp    = device.type == "cuda"
    amp_dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for tokens, mask, text_lens in loader:
            tokens    = tokens.to(device)
            mask      = mask.to(device)
            text_lens = text_lens.to(device)

            x = tokens[:, :-1]
            m = mask[:, :-1]
            y = make_labels(tokens, text_lens, pad_id).to(device)

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
                acc = masked_token_accuracy(logits.detach(), y)
                print(f"epoch {epoch+1:3d} | step {global_step:6d} | "
                      f"loss {loss.item():.4f} | acc {acc:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % args.save_every == 0:
                ckpt_path = args.save_dir / f"step_{global_step:07d}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  saved -> {ckpt_path}")

        avg_loss   = epoch_loss / len(loader)
        epoch_time = time.perf_counter() - t0
        print(f"=== epoch {epoch+1} | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s ===")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_dir / "best.pt")
            print(f"  best -> {args.save_dir}/best.pt | loss={best_loss:.4f}\n")

    print(f"训练完成，最优 loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
