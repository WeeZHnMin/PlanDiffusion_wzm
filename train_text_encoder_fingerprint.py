"""
Phase 1: Fine-tune BERT text encoder via fixed floor-plan fingerprint regression.

[1] FloorPlanFingerprint: fixed random projection of (adj upper-triangle + coords).
    No learning, different structure → different vector (injective w.p.1).

[2] TextEncoder: BERT (last 4 layers unfrozen) + MLP projection head → 256-dim L2-norm.

Loss: cosine similarity between text_emb and fingerprint target.
      (equivalent to MSE when both are L2-normalized)

Goal: force BERT to output different vectors for different floor plan descriptions.
After training, BERT is saved in HuggingFace format for direct use in diffusion trainer.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert",        type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--save-dir",    type=Path, default=Path("models/bert-finetuned-fp"),
                        help="output dir for fine-tuned BERT (HuggingFace format)")
    parser.add_argument("--n-samples",   type=int,  default=5000)
    parser.add_argument("--epochs",      type=int,  default=80)
    parser.add_argument("--batch-size",  type=int,  default=128)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--bert-lr",     type=float, default=1e-5)
    parser.add_argument("--weight-decay",type=float, default=1e-2)
    parser.add_argument("--max-length",  type=int,  default=128)
    parser.add_argument("--emb-dim",     type=int,  default=256)
    parser.add_argument("--fp-seed",     type=int,  default=12345,
                        help="seed for fixed random projection matrix (must stay constant)")
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--num-workers", type=int,  default=4)
    parser.add_argument("--amp",         action="store_true", default=True)
    parser.add_argument("--no-amp",      action="store_false", dest="amp")
    return parser.parse_args()


# ── 固定指纹编码器（不学习）────────────────────────────────────────────────────

class FloorPlanFingerprint(nn.Module):
    """
    Deterministic, frozen encoder: (adj, coords, node_mask) → 256-dim L2-normalized vector.
    Uses a fixed random projection matrix (seeded), guaranteed injective w.p.1.
    """
    def __init__(self, n_max: int, emb_dim: int = 256, seed: int = 12345):
        super().__init__()
        n_adj     = n_max * (n_max - 1) // 2   # upper-triangle entries
        input_dim = n_adj + n_max * 2           # adj flat + coords flat
        g = torch.Generator()
        g.manual_seed(seed)
        W = torch.randn(input_dim, emb_dim, generator=g)
        self.register_buffer("W", W)            # frozen, not a Parameter
        tri = torch.triu_indices(n_max, n_max, offset=1)
        self.register_buffer("tri_row", tri[0])
        self.register_buffer("tri_col", tri[1])

    def forward(self, coords: torch.Tensor, adj: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, 2)  per-record normalized
        # adj:    (B, N, N)  binary
        # node_mask: (B, N)
        adj_flat    = adj[:, self.tri_row, self.tri_col]               # (B, n_adj)
        coords_flat = (coords * node_mask.unsqueeze(-1)).reshape(coords.size(0), -1)  # (B, N*2)
        flat        = torch.cat([adj_flat, coords_flat], dim=-1)       # (B, input_dim)
        out         = flat @ self.W                                     # (B, emb_dim)
        return F.normalize(out, dim=-1)


# ── 文本编码器（可学习部分）────────────────────────────────────────────────────

class TextProjectionHead(nn.Module):
    def __init__(self, text_hidden: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_hidden, text_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_hidden, emb_dim),
        )

    def forward(self, last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class FPDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, n_max):
        self.n_max = n_max
        prompts    = [r["prompt"] for r in records]
        enc        = tokenizer(prompts, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=max_length)
        self.input_ids  = enc["input_ids"]
        self.attn_mask  = enc["attention_mask"]

        n = len(records)
        self.coords    = torch.zeros((n, n_max, 2),     dtype=torch.float32)
        self.adj       = torch.zeros((n, n_max, n_max), dtype=torch.float32)
        self.node_mask = torch.zeros((n, n_max),        dtype=torch.float32)

        for i, r in enumerate(records):
            n_nodes = int(r["n_nodes"])
            self.node_mask[i, :n_nodes] = 1.0
            raw = r["node_coords"][:n_nodes]
            xs  = [c[0] for c in raw]; ys = [c[1] for c in raw]
            xmin = min(xs); xrng = max(max(xs) - xmin, 1)
            ymin = min(ys); yrng = max(max(ys) - ymin, 1)
            for k, (x, y) in enumerate(raw):
                self.coords[i, k, 0] = 2.0 * (x - xmin) / xrng - 1.0
                self.coords[i, k, 1] = 2.0 * (y - ymin) / yrng - 1.0
            raw_adj = r["adj_matrix"]
            for ri in range(min(len(raw_adj), n_max)):
                for ci in range(min(len(raw_adj[ri]), n_max)):
                    self.adj[i, ri, ci] = float(raw_adj[ri][ci])

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn_mask[idx],
                self.coords[idx], self.adj[idx], self.node_mask[idx])


# ── 相似度诊断 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def text_sim_stats(mat: torch.Tensor) -> dict:
    n   = mat.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    sim = (mat @ mat.T)[idx[0], idx[1]]
    return {"mean": sim.mean().item(), "std": sim.std().item(),
            "p50": sim.median().item(), "p95": sim.float().quantile(0.95).item()}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    print("loading records...")
    all_records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    all_records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    records = random.sample(all_records, min(args.n_samples, len(all_records)))
    n_max   = max(len(r["adj_matrix"]) for r in records)
    print(f"sampled={len(records)}, n_max={n_max}, device={device}, amp={use_amp}")

    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert      = BertModel.from_pretrained(str(args.bert)).to(device)
    # unfreeze last 4 layers
    for name, p in bert.named_parameters():
        layer_id = None
        for part in name.split("."):
            if part.isdigit():
                layer_id = int(part); break
        p.requires_grad = layer_id is not None and layer_id >= 8
    print(f"bert trainable: {sum(p.numel() for p in bert.parameters() if p.requires_grad):,}")

    print("building dataset...")
    dataset  = FPDataset(records, tokenizer, args.max_length, n_max)
    val_n    = max(1, int(len(dataset) * args.val_ratio))
    train_n  = len(dataset) - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=(args.num_workers > 0))

    fp_model  = FloorPlanFingerprint(n_max, args.emb_dim, args.fp_seed).to(device)
    text_head = TextProjectionHead(bert.config.hidden_size, args.emb_dim).to(device)
    print(f"text_head params={sum(p.numel() for p in text_head.parameters()):,}")

    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": text_head.parameters(), "lr": args.lr},
        {"params": bert_params,            "lr": args.bert_lr},
    ], weight_decay=args.weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # baseline text similarity before training
    print("\ncomputing baseline text similarity (before fine-tuning)...")
    bert.eval(); text_head.eval()
    sample_ids   = dataset.input_ids[:200].to(device)
    sample_mask  = dataset.attn_mask[:200].to(device)
    with torch.no_grad():
        hs  = bert(input_ids=sample_ids, attention_mask=sample_mask).last_hidden_state
        emb = text_head(hs, sample_mask)
    stats = text_sim_stats(emb.cpu())
    print(f"  text cos_sim BEFORE: mean={stats['mean']:.4f} std={stats['std']:.4f} p95={stats['p95']:.4f}")

    best_val = float("inf")

    for epoch in range(args.epochs):
        bert.train(); text_head.train()
        train_loss = 0.0; train_sim = 0.0; n_train = 0

        for b_ids, b_mask, b_coords, b_adj, b_nmask in train_loader:
            b_ids    = b_ids.to(device,    non_blocking=True)
            b_mask   = b_mask.to(device,   non_blocking=True)
            b_coords = b_coords.to(device, non_blocking=True)
            b_adj    = b_adj.to(device,    non_blocking=True)
            b_nmask  = b_nmask.to(device,  non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                target   = fp_model(b_coords, b_adj, b_nmask)          # fixed fingerprint
                hs       = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                text_emb = text_head(hs, b_mask)
                sim      = (text_emb * target).sum(dim=-1)              # cosine similarity
                loss     = (1.0 - sim).mean()                           # minimize distance

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(text_head.parameters()) + bert_params, 1.0)
            scaler.step(opt); scaler.update()

            bs = b_ids.size(0)
            train_loss += loss.item() * bs
            train_sim  += sim.mean().item() * bs
            n_train    += bs

        sched.step()

        bert.eval(); text_head.eval()
        val_loss = 0.0; val_sim = 0.0; n_val = 0
        all_text_emb = []

        with torch.no_grad():
            for b_ids, b_mask, b_coords, b_adj, b_nmask in val_loader:
                b_ids    = b_ids.to(device,    non_blocking=True)
                b_mask   = b_mask.to(device,   non_blocking=True)
                b_coords = b_coords.to(device, non_blocking=True)
                b_adj    = b_adj.to(device,    non_blocking=True)
                b_nmask  = b_nmask.to(device,  non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    target   = fp_model(b_coords, b_adj, b_nmask)
                    hs       = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                    text_emb = text_head(hs, b_mask)
                    sim      = (text_emb * target).sum(dim=-1)
                    loss     = (1.0 - sim).mean()

                bs = b_ids.size(0)
                val_loss += loss.item() * bs
                val_sim  += sim.mean().item() * bs
                n_val    += bs
                all_text_emb.append(text_emb.cpu())

        tl = train_loss / max(n_train, 1)
        vl = val_loss   / max(n_val,   1)
        ts = train_sim  / max(n_train, 1)
        vs = val_sim    / max(n_val,   1)

        improved = vl < best_val
        if improved:
            best_val = vl

        print(f"epoch={epoch+1:3d} "
              f"train_loss={tl:.4f} train_sim={ts:.4f} "
              f"val_loss={vl:.4f} val_sim={vs:.4f} "
              f"{'(best)' if improved else ''}")

    # ── 训后文本相似度诊断 ─────────────────────────────────────────────────────
    t_mat = F.normalize(torch.cat(all_text_emb, dim=0), dim=-1)
    stats = text_sim_stats(t_mat)
    print(f"\ntext cos_sim AFTER:  mean={stats['mean']:.4f} std={stats['std']:.4f} p95={stats['p95']:.4f}")
    print(f"(baseline was ~0.90 mean before fine-tuning)")

    # ── 保存微调后的 BERT（HuggingFace 格式，可直接 from_pretrained 加载）──────
    args.save_dir.mkdir(parents=True, exist_ok=True)
    bert.save_pretrained(str(args.save_dir))
    tokenizer.save_pretrained(str(args.save_dir))
    # 也保存 projection head，供分析用
    torch.save({
        "text_head": text_head.state_dict(),
        "emb_dim":   args.emb_dim,
        "fp_seed":   args.fp_seed,
        "n_max":     n_max,
    }, args.save_dir / "text_head.pt")
    print(f"\nsaved fine-tuned BERT → {args.save_dir}")
    print(f"usage: --bert {args.save_dir}")


if __name__ == "__main__":
    main()
