"""
Train text -> adjacency(0/1) with a Transformer Decoder head.

- Text encoder: frozen bert-base-chinese
- Head: learned node queries + TransformerDecoder + bilinear edge scorer
- Target: adj_matrix in data/jsonl/train_nodes.jsonl
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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--save", type=Path, default=Path("core/weights/adj_text_decoder.pt"))
    return parser.parse_args()


def scan_jsonl(path: Path):
    offsets = []
    n_max = None
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] skip bad json near byte offset {pos}")
                continue
            if n_max is None:
                n_max = len(obj["adj_matrix"])
            offsets.append(pos)
    if not offsets:
        raise SystemExit(f"No valid records in {path}")
    if bad:
        print(f"[WARN] skipped bad lines: {bad}")
    return offsets, n_max


def read_batch(path: Path, offsets, s: int, e: int):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i in range(s, e):
            f.seek(offsets[i])
            line = f.readline().strip()
            records.append(json.loads(line))
    return records


class TextToAdjDecoder(nn.Module):
    def __init__(self, n_max: int, d_model: int, nhead: int, layers: int, ffn: int):
        super().__init__()
        self.n_max = n_max
        self.text_proj = nn.Linear(768, d_model)
        self.query_embed = nn.Parameter(torch.randn(n_max, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.edge_w = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.edge_b = nn.Parameter(torch.zeros(1))

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        # text_tokens: [B, L, 768], text_mask: [B, L] with 1=valid
        memory = self.text_proj(text_tokens)  # [B, L, D]
        bsz = memory.size(0)
        tgt = self.query_embed.unsqueeze(0).expand(bsz, self.n_max, -1)  # [B, N, D]
        h = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=(text_mask == 0),
        )  # [B, N, D]

        logits = torch.einsum("bnd,df,bmf->bnm", h, self.edge_w, h) + self.edge_b
        # Keep symmetry; diagonal is learned through loss.
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits


def iterate_batches(n: int, batch_size: int):
    for s in range(0, n, batch_size):
        yield s, min(s + batch_size, n)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.bert.exists():
        raise SystemExit(f"Missing bert path: {args.bert}")

    offsets, n_max = scan_jsonl(args.data)
    n = len(offsets)
    print(f"records={n}, n_max={n_max}, device={device}")

    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert = BertModel.from_pretrained(str(args.bert)).to(device)
    for p in bert.parameters():
        p.requires_grad = False
    bert.eval()

    model = TextToAdjDecoder(
        n_max=n_max,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ffn=args.ffn,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0

        for s, e in iterate_batches(n, args.batch_size):
            batch_recs = read_batch(args.data, offsets, s, e)
            batch_prompts = [r["prompt"] for r in batch_recs]
            batch_adj = torch.tensor([r["adj_matrix"] for r in batch_recs], dtype=torch.float32, device=device)
            batch_n = torch.tensor([int(r["n_nodes"]) for r in batch_recs], dtype=torch.long, device=device)
            batch_valid = torch.zeros((e - s, n_max, n_max), dtype=torch.float32, device=device)
            for bi, cnt in enumerate(batch_n.tolist()):
                batch_valid[bi, :cnt, :cnt] = 1.0
            batch_pad = (1.0 - batch_valid).bool()

            with torch.no_grad():
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                text_out = bert(**enc).last_hidden_state
                text_mask = enc["attention_mask"].float()

            logits = model(text_out, text_mask)
            logits_valid = logits[batch_valid.bool()]
            labels_valid = batch_adj[batch_valid.bool()]
            loss_valid = F.binary_cross_entropy_with_logits(logits_valid, labels_valid)

            logits_pad = logits[batch_pad]
            loss_pad = F.binary_cross_entropy_with_logits(logits_pad, torch.zeros_like(logits_pad))
            loss = loss_valid + loss_pad

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = e - s
            total_loss += loss.item() * bs
            total_items += bs

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # quick metric on first batch
                s, e = 0, min(args.batch_size, n)
                batch_recs = read_batch(args.data, offsets, s, e)
                batch_prompts = [r["prompt"] for r in batch_recs]
                batch_adj = torch.tensor([r["adj_matrix"] for r in batch_recs], dtype=torch.float32, device=device)
                batch_n = torch.tensor([int(r["n_nodes"]) for r in batch_recs], dtype=torch.long, device=device)
                batch_valid = torch.zeros((e - s, n_max, n_max), dtype=torch.float32, device=device)
                for bi, cnt in enumerate(batch_n.tolist()):
                    batch_valid[bi, :cnt, :cnt] = 1.0
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                text_out = bert(**enc).last_hidden_state
                text_mask = enc["attention_mask"].float()
                logits = model(text_out, text_mask)
                preds = (logits.sigmoid() > 0.5).float()
                vmask = batch_valid.bool()
                labels_valid = batch_adj[vmask]
                acc = (preds[vmask] == labels_valid).float().mean().item()
            print(
                f"epoch={epoch+1:4d} loss={total_loss/max(total_items,1):.4f} "
                f"sample_acc={acc*100:.2f}%"
            )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_max": n_max,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "ffn": args.ffn,
            "bert_path": str(args.bert),
            "data_path": str(args.data),
        },
        args.save,
    )
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
