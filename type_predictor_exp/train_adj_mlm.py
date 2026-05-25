"""
邻接矩阵 MLM。

把上三角展开成序列：
  位置 k 对应节点对 (i,j)，token = adj[i][j] ∈ {0, 1}
  词表: 0=无边, 1=有边, 2=MASK
  序列长度 = n_max*(n_max-1)//2  (N=40 → 780)

每个位置的输入特征:
  tok_emb(token) + pos_emb(k) + type_proj(type_emb(i), type_emb(j))

随机 mask 15% 有效位置，cross-entropy 只算 masked 位置。
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOM_TYPES = ["bedroom", "bathroom", "living_room", "kitchen", "corridor"]
TYPE2IDX   = {t: i for i, t in enumerate(ROOM_TYPES)}
N_TYPES    = len(ROOM_TYPES)
MASK_ID    = 2   # vocab: 0=no edge, 1=edge, 2=MASK


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=Path,  default=Path("data/jsonl/mapped_node_data_zh.jsonl"))
    p.add_argument("--save",        type=Path,  default=Path("type_predictor_exp/weights/adj_mlm.pt"))
    p.add_argument("--n-samples",   type=int,   default=0,    help="0=all")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--wd",          type=float, default=1e-2)
    p.add_argument("--mask-ratio",  type=float, default=0.15)
    p.add_argument("--d-model",     type=int,   default=256)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--n-layers",    type=int,   default=6)
    p.add_argument("--val-ratio",   type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      action="store_false", dest="amp")
    return p.parse_args()


# ── 模型 ──────────────────────────────────────────────────────────────────────

class AdjMLM(nn.Module):
    """
    输入: tokens (B, L), type_i (B, L), type_j (B, L), seq_valid (B, L)
    输出: logits (B, L, 2)  → cross-entropy 只算 masked 位置
    """
    def __init__(self, n_max, n_types, d_model, n_heads, n_layers):
        super().__init__()
        n_pairs = n_max * (n_max - 1) // 2

        self.tok_emb   = nn.Embedding(3, d_model)           # 0/1/MASK
        self.pos_emb   = nn.Embedding(n_pairs, d_model)
        self.type_emb  = nn.Embedding(n_types + 1, d_model, padding_idx=n_types)
        self.type_proj = nn.Linear(d_model * 2, d_model, bias=False)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2))

    def forward(self, tokens, type_i, type_j, seq_valid):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)

        x = self.tok_emb(tokens) + self.pos_emb(pos)
        ti = self.type_emb(type_i)
        tj = self.type_emb(type_j)
        x = x + self.type_proj(torch.cat([ti, tj], dim=-1))

        key_pad = (seq_valid == 0)   # padding 位置不参与 attention
        x = self.transformer(x, src_key_padding_mask=key_pad)
        return self.head(x)          # (B, L, 2)


# ── MLM masking ───────────────────────────────────────────────────────────────

def apply_mlm_mask(tokens, seq_valid, mask_ratio, device):
    """
    对每个样本的有效位置随机 mask mask_ratio 比例。
    返回: masked_tokens (B,L), labels (B,L)  labels=-100 表示不计算 loss
    """
    masked = tokens.clone()
    labels = torch.full_like(tokens, -100)

    B = tokens.size(0)
    for b in range(B):
        valid = seq_valid[b].nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        n_mask = max(1, int(len(valid) * mask_ratio))
        chosen = valid[torch.randperm(len(valid), device=device)[:n_mask]]
        labels[b, chosen]  = tokens[b, chosen]
        masked[b, chosen]  = MASK_ID

    return masked, labels


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def build_pairs(n_max):
    """上三角 (i,j) 对，i<j，共 n_max*(n_max-1)//2 个"""
    return [(i, j) for i in range(n_max) for j in range(i + 1, n_max)]


def load_tensors(records, n_max, pairs):
    N      = len(records)
    L      = len(pairs)
    tokens  = torch.zeros((N, L), dtype=torch.long)
    type_i  = torch.full((N, L), N_TYPES, dtype=torch.long)
    type_j  = torch.full((N, L), N_TYPES, dtype=torch.long)
    valid   = torch.zeros((N, L), dtype=torch.float32)

    for s, r in enumerate(records):
        n    = int(r["n_nodes"])
        ntypes = [TYPE2IDX.get(t, N_TYPES - 1) for t in r["node_types"][:n]]
        adj  = r["adj_matrix"]
        for k, (i, j) in enumerate(pairs):
            if i < n and j < n:
                tokens[s, k]  = int(adj[i][j])
                type_i[s, k]  = ntypes[i]
                type_j[s, k]  = ntypes[j]
                valid[s, k]   = 1.0

    return tokens, type_i, type_j, valid


# ── 评估指标 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, mask_ratio, device, use_amp):
    model.eval()
    total_loss = total_acc = total_n = 0.0
    tp = fp = fn = 0.0

    for tokens, ti, tj, sv in loader:
        tokens = tokens.to(device); ti = ti.to(device)
        tj     = tj.to(device);     sv = sv.to(device)
        masked, labels = apply_mlm_mask(tokens, sv, mask_ratio, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(masked, ti, tj, sv)   # (B, L, 2)
            loss   = F.cross_entropy(
                logits.view(-1, 2), labels.view(-1), ignore_index=-100)

        mask_pos = (labels != -100)
        pred = logits.argmax(-1)
        true = labels.clone(); true[true == -100] = 0

        total_loss += loss.item()
        total_acc  += (pred[mask_pos] == true[mask_pos]).float().sum().item()
        total_n    += mask_pos.sum().item()

        # precision / recall on masked positions
        p = pred[mask_pos]; t = true[mask_pos]
        tp += ((p == 1) & (t == 1)).sum().item()
        fp += ((p == 1) & (t == 0)).sum().item()
        fn += ((p == 0) & (t == 1)).sum().item()

    n_batches = max(len(loader), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    return {
        "loss":  total_loss / n_batches,
        "acc":   total_acc  / max(total_n, 1),
        "prec":  prec,
        "rec":   rec,
        "f1":    2 * prec * rec / max(prec + rec, 1e-8),
    }


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    print("loading records...")
    all_records = []
    with args.data.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                all_records.append(json.loads(s))

    records = (random.sample(all_records, min(args.n_samples, len(all_records)))
               if args.n_samples > 0 else all_records)
    n_max = max(len(r["adj_matrix"]) for r in records)
    pairs = build_pairs(n_max)
    L     = len(pairs)
    print(f"total={len(records)}  n_max={n_max}  seq_len={L}  device={device}  amp={use_amp}")

    tokens, type_i, type_j, valid = load_tensors(records, n_max, pairs)
    del all_records, records

    dataset = TensorDataset(tokens, type_i, type_j, valid)
    val_n   = max(1, int(len(dataset) * args.val_ratio))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - val_n, val_n],
        generator=torch.Generator().manual_seed(args.seed))

    pin          = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)

    model = AdjMLM(n_max, N_TYPES, args.d_model, args.n_heads, args.n_layers).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0; tr_n = 0

        for tokens, ti, tj, sv in train_loader:
            tokens = tokens.to(device); ti = ti.to(device)
            tj     = tj.to(device);     sv = sv.to(device)
            masked, labels = apply_mlm_mask(tokens, sv, args.mask_ratio, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(masked, ti, tj, sv)
                loss   = F.cross_entropy(
                    logits.view(-1, 2), labels.view(-1), ignore_index=-100)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()
            tr_n    += 1
        sched.step()

        m = evaluate(model, val_loader, args.mask_ratio, device, use_amp)
        tl = tr_loss / max(tr_n, 1)
        improved = m["loss"] < best_val
        if improved:
            best_val = m["loss"]
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": n_max, "n_types": N_TYPES,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "pairs": pairs,
            }, args.save)

        print(f"epoch={epoch+1:4d}  train_loss={tl:.4f}  "
              f"val_loss={m['loss']:.4f}  acc={m['acc']:.3f}  "
              f"prec={m['prec']:.3f}  rec={m['rec']:.3f}  f1={m['f1']:.3f}  "
              f"{'(saved)' if improved else ''}")

    print(f"\ndone.  best_val={best_val:.6f}  saved -> {args.save}")


if __name__ == "__main__":
    main()
