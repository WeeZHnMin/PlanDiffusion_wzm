"""
CLIP-style contrastive pre-training for floor plan + text alignment.

Text encoder  : BERT (last 4 layers unfrozen) + projection head → 256-dim L2-normalized
FP encoder    : GCN aggregation + self-attention over nodes → mean pool + projection head
Loss          : InfoNCE (symmetric, learnable temperature)

Goal: force BERT to produce discriminative embeddings for different floor plan descriptions.
After training, compare cosine similarity distribution with the pre-training baseline (~0.90).
"""

import argparse
import json
import math
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
    parser.add_argument("--save",        type=Path, default=Path("core/weights/clip_floorplan.pt"))
    parser.add_argument("--n-samples",   type=int,  default=5000)
    parser.add_argument("--epochs",      type=int,  default=60)
    parser.add_argument("--batch-size",  type=int,  default=256)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--bert-lr",     type=float, default=1e-5)
    parser.add_argument("--weight-decay",type=float, default=1e-2)
    parser.add_argument("--max-length",  type=int,  default=128)
    parser.add_argument("--emb-dim",     type=int,  default=256)
    parser.add_argument("--fp-d-model",  type=int,  default=256)
    parser.add_argument("--fp-n-heads",  type=int,  default=8)
    parser.add_argument("--fp-n-layers", type=int,  default=4)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--num-workers", type=int,  default=4)
    parser.add_argument("--amp",         action="store_true", default=True)
    parser.add_argument("--no-amp",      action="store_false", dest="amp")
    return parser.parse_args()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FloorPlanDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, n_max):
        self.n_max = n_max
        self.max_length = max_length

        prompts = [r["prompt"] for r in records]
        enc = tokenizer(prompts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_length)
        self.input_ids    = enc["input_ids"]
        self.attn_mask    = enc["attention_mask"]

        n = len(records)
        self.coords    = torch.zeros((n, n_max, 2),     dtype=torch.float32)
        self.adj       = torch.zeros((n, n_max, n_max), dtype=torch.float32)
        self.node_mask = torch.zeros((n, n_max),        dtype=torch.float32)

        for i, r in enumerate(records):
            n_nodes = int(r["n_nodes"])
            self.node_mask[i, :n_nodes] = 1.0
            raw = r["node_coords"][:n_nodes]
            xs = [c[0] for c in raw]; ys = [c[1] for c in raw]
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


# ─── Floor Plan Encoder ────────────────────────────────────────────────────────

class FPSelfAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class FloorPlanEncoder(nn.Module):
    """
    Encodes (coords, adj, node_mask) → fixed-size L2-normalized embedding.
    Input projection: coords (2) + 2-hop GCN aggregation (4) = 6 dims → d_model
    """
    def __init__(self, d_model, n_heads, n_layers, emb_dim):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(6, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([FPSelfAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, emb_dim),
        )

    def forward(self, coords, adj, node_mask):
        # 2-hop GCN aggregation
        deg  = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        a1   = (adj / deg) * node_mask.unsqueeze(-1)
        a2   = (torch.bmm(a1, a1) / deg.clamp(min=1.0)) * node_mask.unsqueeze(-1)
        feat = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)

        x = self.input_proj(feat)
        pad_mask = (node_mask == 0)
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.norm(x)

        # mean pool over valid nodes
        mask3 = node_mask.unsqueeze(-1)
        pooled = (x * mask3).sum(dim=1) / mask3.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)


# ─── Text Projection Head ──────────────────────────────────────────────────────

class TextProjection(nn.Module):
    def __init__(self, text_hidden, emb_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_hidden, text_hidden), nn.GELU(),
            nn.Linear(text_hidden, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


# ─── InfoNCE Loss ──────────────────────────────────────────────────────────────

class InfoNCE(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        # learnable log-temperature, clamped to [log(0.01), log(0.5)]
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, text_emb, fp_emb):
        temp = self.log_temp.exp().clamp(min=0.01, max=0.5)
        logits = (text_emb @ fp_emb.T) / temp      # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t = F.cross_entropy(logits,   labels)
        loss_f = F.cross_entropy(logits.T, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return (loss_t + loss_f) / 2, acc, temp.item()


# ─── Cosine Similarity Report ──────────────────────────────────────────────────

@torch.no_grad()
def report_sim(mat, name):
    n = mat.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    vals = (mat @ mat.T)[idx[0], idx[1]]
    print(f"  {name}: mean={vals.mean():.4f}  std={vals.std():.4f}"
          f"  p50={vals.median():.4f}  p95={vals.float().quantile(0.95):.4f}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp  = bool(args.amp and device.type == "cuda")

    print("loading records...")
    records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    print(f"total={len(records)}, sampling {args.n_samples}")
    records = random.sample(records, min(args.n_samples, len(records)))
    n_max   = max(int(r["n_nodes"]) for r in records)
    n_max   = max(n_max, len(records[0]["adj_matrix"]))

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
    bert_trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print(f"bert trainable: {bert_trainable:,}")

    print("building dataset...")
    dataset  = FloorPlanDataset(records, tokenizer, args.max_length, n_max)
    val_n    = max(1, int(len(dataset) * args.val_ratio))
    train_n  = len(dataset) - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                              persistent_workers=(args.num_workers > 0))

    text_hidden = bert.config.hidden_size
    fp_enc      = FloorPlanEncoder(args.fp_d_model, args.fp_n_heads,
                                   args.fp_n_layers, args.emb_dim).to(device)
    text_proj   = TextProjection(text_hidden, args.emb_dim).to(device)
    nce_loss    = InfoNCE().to(device)

    print(f"fp_encoder params={sum(p.numel() for p in fp_enc.parameters()):,}")
    print(f"text_proj  params={sum(p.numel() for p in text_proj.parameters()):,}")

    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": fp_enc.parameters(),    "lr": args.lr},
        {"params": text_proj.parameters(), "lr": args.lr},
        {"params": nce_loss.parameters(),  "lr": args.lr},
        {"params": bert_params,            "lr": args.bert_lr},
    ], weight_decay=args.weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        bert.train(); fp_enc.train(); text_proj.train(); nce_loss.train()
        train_loss_sum = 0.0; train_acc_sum = 0.0; train_items = 0

        for b_ids, b_mask, b_coords, b_adj, b_nmask in train_loader:
            b_ids    = b_ids.to(device,    non_blocking=True)
            b_mask   = b_mask.to(device,   non_blocking=True)
            b_coords = b_coords.to(device, non_blocking=True)
            b_adj    = b_adj.to(device,    non_blocking=True)
            b_nmask  = b_nmask.to(device,  non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # text side: mean pool over non-padding tokens
                last_hs   = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                m         = b_mask.unsqueeze(-1).float()
                text_pool = (last_hs * m).sum(1) / m.sum(1).clamp(min=1.0)
                t_emb     = text_proj(text_pool)

                # floor plan side
                fp_emb = fp_enc(b_coords, b_adj, b_nmask)

                loss, acc, temp = nce_loss(t_emb, fp_emb)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(fp_enc.parameters()) + list(text_proj.parameters()) + bert_params, 1.0)
            scaler.step(opt)
            scaler.update()

            bs = b_ids.size(0)
            train_loss_sum += loss.item() * bs
            train_acc_sum  += acc.item()  * bs
            train_items    += bs

        sched.step()

        bert.eval(); fp_enc.eval(); text_proj.eval(); nce_loss.eval()
        val_loss_sum = 0.0; val_acc_sum = 0.0; val_items = 0
        all_t_emb = []; all_fp_emb = []

        with torch.no_grad():
            for b_ids, b_mask, b_coords, b_adj, b_nmask in val_loader:
                b_ids    = b_ids.to(device,    non_blocking=True)
                b_mask   = b_mask.to(device,   non_blocking=True)
                b_coords = b_coords.to(device, non_blocking=True)
                b_adj    = b_adj.to(device,    non_blocking=True)
                b_nmask  = b_nmask.to(device,  non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    last_hs   = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                    m         = b_mask.unsqueeze(-1).float()
                    text_pool = (last_hs * m).sum(1) / m.sum(1).clamp(min=1.0)
                    t_emb     = text_proj(text_pool)
                    fp_emb    = fp_enc(b_coords, b_adj, b_nmask)
                    loss, acc, _ = nce_loss(t_emb, fp_emb)

                bs = b_ids.size(0)
                val_loss_sum += loss.item() * bs
                val_acc_sum  += acc.item()  * bs
                val_items    += bs
                all_t_emb.append(t_emb.cpu())
                all_fp_emb.append(fp_emb.cpu())

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss   = val_loss_sum   / max(val_items,   1)
        train_acc  = train_acc_sum  / max(train_items, 1)
        val_acc    = val_acc_sum    / max(val_items,   1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "fp_encoder":   fp_enc.state_dict(),
                "text_proj":    text_proj.state_dict(),
                "bert_state":   {k: v for k, v in bert.state_dict().items()},
                "nce_loss":     nce_loss.state_dict(),
                "fp_d_model":   args.fp_d_model,
                "fp_n_heads":   args.fp_n_heads,
                "fp_n_layers":  args.fp_n_layers,
                "emb_dim":      args.emb_dim,
                "n_max":        n_max,
                "bert_path":    str(args.bert),
                "epoch":        epoch + 1,
                "best_val_loss": best_val_loss,
            }, args.save)

        print(f"epoch={epoch+1:3d} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% "
              f"temp={temp:.4f} {'(saved)' if improved else ''}")

    # ── post-training similarity report ──────────────────────────────────────
    print("\n=== post-training embedding similarity (val set) ===")
    t_mat  = F.normalize(torch.cat(all_t_emb,  dim=0), dim=-1)
    fp_mat = F.normalize(torch.cat(all_fp_emb, dim=0), dim=-1)
    report_sim(t_mat,  "text emb (after CLIP)")
    report_sim(fp_mat, "fp emb")
    # diagonal = matched pairs
    diag = (t_mat * fp_mat).sum(dim=-1)
    print(f"  matched-pair cos_sim: mean={diag.mean():.4f}  min={diag.min():.4f}  max={diag.max():.4f}")
    print(f"\nbest_val_loss={best_val_loss:.4f}  saved={args.save}")


if __name__ == "__main__":
    main()
