"""Evaluate a trained TextGraphAlign checkpoint on a full split.

Supports both NPZ files produced by text_graph_align.build_npz and raw JSONL
files. JSONL inputs are converted in memory with the same process_jsonl logic
used by training validation, so coordinates are normalized consistently.

Usage:
  python -m text_graph_align.eval \
      --ckpt checkpoints/text_graph_align/align_best.pt \
      --data data/jsonl/graphs_160k_spatial_val.jsonl \
      --bert models/bert-base-uncased \
      --batch 512
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import BertTokenizer

from .dataset import load_align_data, load_align_jsonl_data
from .model import TextGraphAlign


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="align_best.pt or align_latest.pt")
    p.add_argument("--data", required=True, help="Evaluation split, .npz or .jsonl")
    p.add_argument("--bert", default="models/bert-base-uncased")
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0, help="Only for JSONL; 0=full split")
    p.add_argument("--unfreeze_layers", type=int, default=0,
                   help="Architecture compatibility only; checkpoint weights decide values")
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_embed", type=int, default=384)
    p.add_argument("--out", default="", help="Optional summary JSON path")
    return p.parse_args()


def load_eval_data(args: argparse.Namespace):
    suffix = Path(args.data).suffix.lower()
    if suffix == ".jsonl":
        tokenizer = BertTokenizer.from_pretrained(args.bert)
        return load_align_jsonl_data(
            args.data,
            tokenizer,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            max_samples=args.max_samples,
        )
    return load_align_data(
        args.data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
    )


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[TextGraphAlign, int | None]:
    model = TextGraphAlign(
        bert_name=args.bert,
        unfreeze_layers=args.unfreeze_layers,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_embed=args.d_embed,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    model.eval()
    return model, step


@torch.no_grad()
def evaluate(model: TextGraphAlign, loader, device: torch.device) -> dict:
    total_loss = 0.0
    total_g2t = 0.0
    total_t2g = 0.0
    n_batches = 0
    n_samples = 0

    for batch in loader:
        coords = batch["node_coords"].to(device)
        adj = batch["adj_matrix"].to(device)
        mask = batch["node_mask"].to(device)
        member = batch["room_membership"].to(device)
        ptok = batch["prompt_tokens"].to(device)
        pmsk = batch["prompt_mask"].to(device)

        loss = model(coords, adj, mask, member, ptok, pmsk)
        acc_g2t, acc_t2g = model.compute_metrics(coords, adj, mask, member, ptok, pmsk)
        bs = int(mask.shape[0])
        total_loss += float(loss.item())
        total_g2t += float(acc_g2t)
        total_t2g += float(acc_t2g)
        n_batches += 1
        n_samples += bs

    denom = max(n_batches, 1)
    return {
        "samples": n_samples,
        "batches": n_batches,
        "loss": total_loss / denom,
        "acc_g2t": total_g2t / denom,
        "acc_t2g": total_t2g / denom,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  gpus: {args.gpus}")

    t0 = time.perf_counter()
    _, loader = load_eval_data(args)
    model, step = load_model(args, device)
    metrics = evaluate(model, loader, device)
    metrics.update({
        "ckpt": args.ckpt,
        "ckpt_step": step,
        "data": args.data,
        "batch": args.batch,
        "elapsed": round(time.perf_counter() - t0, 2),
    })

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved summary -> {out_path}")


if __name__ == "__main__":
    main()
