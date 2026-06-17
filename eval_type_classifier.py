"""
NodeTypeClassifier 评估脚本（θ₃）

计算 Type Acc.：给定真实坐标和图结构，预测节点类型的准确率（有效节点上）。

用法：
  python eval_type_classifier.py --ckpt checkpoints/node_type/XXXXXX/model_latest.pt
  python eval_type_classifier.py --ckpt checkpoints/node_type/XXXXXX/model_latest.pt --n_eval 1000
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from node_diffusion_cross_att.dataset import TypeDataset
from node_diffusion_cross_att.type_model import NodeTypeClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      required=True, help="checkpoint 路径")
    p.add_argument("--data_path", default="data/processed/node_diffusion_cross_att/type_dataset_test_10k.npz")
    p.add_argument("--bert",      default="models/bert-base-uncased")
    p.add_argument("--n_eval",    type=int, default=0, help="评估条数，0=全部")
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="type_eval_results.json")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = NodeTypeClassifier(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    raw_sd = ckpt["model"]
    if any(k.startswith("module.") for k in raw_sd):
        raw_sd = {k[7:]: v for k, v in raw_sd.items()}
    model.load_state_dict(raw_sd)
    model.eval()
    print(f"checkpoint: {args.ckpt}  step={ckpt.get('step', '?')}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = TypeDataset(args.data_path)
    total = len(dataset)
    print(f"数据集: {total} 条  path={args.data_path}")

    if args.n_eval > 0 and args.n_eval < total:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(total, size=args.n_eval, replace=False).tolist()
        dataset = Subset(dataset, idxs)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    total_correct = 0
    total_valid   = 0

    with torch.no_grad():
        for i, (x, cond) in enumerate(loader):
            x    = x.to(device)
            cond = {k: v.to(device) for k, v in cond.items()}

            logits    = model(
                x,
                adj_matrix    = cond["adj_matrix"],
                node_mask     = cond["node_mask"],
                prompt_tokens = cond["prompt_tokens"],
                prompt_mask   = cond["prompt_mask"],
            )                                       # [B, N, 33]

            targets   = cond["node_types"]          # [B, N]
            node_mask = cond["node_mask"]           # [B, N]

            pred  = logits.argmax(dim=-1)           # [B, N]
            valid = node_mask > 0.5
            total_correct += ((pred == targets) & valid).sum().item()
            total_valid   += valid.sum().item()

            if (i + 1) % 20 == 0:
                acc_so_far = total_correct / max(total_valid, 1)
                print(f"  [{(i+1)*args.batch_size}/{len(dataset)}] acc={acc_so_far:.4f}", flush=True)

    acc = total_correct / max(total_valid, 1)
    print(f"\n{'─'*40}")
    print(f"  Type Acc. : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  valid nodes: {total_valid}")
    print(f"{'─'*40}")

    result = {
        "type_acc":    round(acc, 6),
        "type_acc_pct": round(acc * 100, 2),
        "total_valid_nodes": total_valid,
        "ckpt": args.ckpt,
        "data": args.data_path,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果保存至 {args.out}")


if __name__ == "__main__":
    main()
