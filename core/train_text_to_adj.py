"""
Train a text encoder + MLP head to predict 0/1 adjacency matrices.

Default data:
    data/jsonl/train_nodes.jsonl
Default encoder:
    models/bert-base-chinese
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--save", type=Path, default=Path("core/weights/adj_text_encoder.pt"))
    return parser.parse_args()


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No records in {path}")
    return records


class AdjPredictor(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_max: int):
        super().__init__()
        self.n_max = n_max
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_max * n_max),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out.view(-1, self.n_max, self.n_max)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.bert.exists():
        raise SystemExit(f"Missing bert path: {args.bert}")

    records = load_records(args.data)
    n_max = len(records[0]["adj_matrix"])
    n = len(records)
    print(f"records={n}, n_max={n_max}, device={device}")

    prompts = [r["prompt"] for r in records]
    adj_tensor = torch.tensor([r["adj_matrix"] for r in records], dtype=torch.float32, device=device)

    node_counts = torch.tensor([int(r["n_nodes"]) for r in records], dtype=torch.long, device=device)
    valid_masks = torch.zeros((n, n_max, n_max), dtype=torch.float32, device=device)
    for i, cnt in enumerate(node_counts.tolist()):
        valid_masks[i, :cnt, :cnt] = 1.0
    pad_masks = (1.0 - valid_masks).bool()

    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert = BertModel.from_pretrained(str(args.bert)).to(device)
    for p in bert.parameters():
        p.requires_grad = False
    bert.eval()

    print("encoding prompts...")
    cls_encs = []
    with torch.no_grad():
        for p in prompts:
            inp = tokenizer(
                p,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            out = bert(**inp)
            cls_encs.append(out.last_hidden_state[0, 0])
    cls_encs = torch.stack(cls_encs)
    d_in = cls_encs.shape[-1]
    print(f"cls_encs={tuple(cls_encs.shape)}")

    model = AdjPredictor(d_in=d_in, d_hidden=args.hidden, n_max=n_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        logits = model(cls_encs)
        logits_valid = logits[valid_masks.bool()]
        labels_valid = adj_tensor[valid_masks.bool()]
        loss_valid = F.binary_cross_entropy_with_logits(logits_valid, labels_valid)

        logits_pad = logits[pad_masks]
        loss_pad = F.binary_cross_entropy_with_logits(logits_pad, torch.zeros_like(logits_pad))
        loss = loss_valid + loss_pad

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                preds = (logits.sigmoid() > 0.5).float()
                acc = (preds[valid_masks.bool()] == labels_valid).float().mean().item()
                edge_mask = (adj_tensor * valid_masks).bool()
                recall = (preds[edge_mask] == 1.0).float().mean().item() if edge_mask.sum() > 0 else 0.0
                pad_fp = preds[pad_masks].mean().item()
            print(
                f"epoch={epoch+1:4d} loss={loss.item():.4f} "
                f"acc={acc*100:.2f}% recall={recall*100:.2f}% pad_fp={pad_fp*100:.2f}%"
            )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_max": n_max,
            "d_in": d_in,
            "d_hidden": args.hidden,
            "bert_path": str(args.bert),
            "data_path": str(args.data),
        },
        args.save,
    )
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
