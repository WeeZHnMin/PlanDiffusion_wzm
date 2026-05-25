"""
Unconditional epsilon-prediction diffusion for rel_adj_half.

Architecture per forward pass:
  1. in_proj(2->D) + row/col pos emb + timestep emb
  2. CNN(k=3) over j-dim per row  [residual]
  3. Per-row self-attention -> scalar logit -> sigmoid (BCE aux loss) / softmax (mask)
     [residual + FFN]
  4. feat_cnn (x) softmax mask -> feat_dot
  5. Per-row self-attention on feat_dot  [residual + FFN]
  6. Linear(D->2) -> predicted noise

Diagonal (j==i) and lower triangle (j<i) are always masked in attention and excluded from loss.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── noise schedule ────────────────────────────────────────────────────────────

def cosine_alpha_bars(timesteps: int, s: float = 0.008) -> torch.Tensor:
    ts = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((ts / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor,
             noise: torch.Tensor, alpha_bars: torch.Tensor) -> torch.Tensor:
    ab = alpha_bars[t_idx].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ─── model building blocks ─────────────────────────────────────────────────────

class _FFN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class _RowAttnBlock(nn.Module):
    """Pre-norm self-attention + residual + FFN, operating on a row sequence."""
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.ffn = _FFN(d)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # x: [B, L, D]
        normed = self.norm(x)
        out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        return self.ffn(x + out)


# ─── main model ────────────────────────────────────────────────────────────────

class RelAdjUncondModel(nn.Module):
    def __init__(self, n_max: int = 40, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.n_max = n_max
        self.d_model = d_model

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        self.in_proj = nn.Linear(2, d_model)
        self.cnn = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.cnn_norm = nn.LayerNorm(d_model)
        self.row_pos = nn.Embedding(n_max, d_model)
        self.col_pos = nn.Embedding(n_max, d_model)

        # Step 3: per-row attention -> scalar mask logit
        self.mask_attn = _RowAttnBlock(d_model, n_heads)
        self.mask_head = nn.Linear(d_model, 1)

        # Step 5: per-row attention after dot mask
        self.post_attn = _RowAttnBlock(d_model, n_heads)

        self.out_head = nn.Linear(d_model, 2)

    def _build_kpm(self, B: int, N: int,
                   node_mask: torch.Tensor, device) -> torch.Tensor:
        """key_padding_mask [B*N, N]: True = ignore.
           Row i masks j<=i (lower tri + diagonal) and invalid node-j positions."""
        # tril[i, j] = True when j <= i
        tril = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))
        row_idx = torch.arange(N, device=device).repeat(B)  # [B*N]
        kpm = tril[row_idx]                                  # [B*N, N]
        invalid_j = (node_mask == 0).unsqueeze(1).expand(B, N, N).reshape(B * N, N)
        return kpm | invalid_j

    def forward(
        self,
        x_t: torch.Tensor,        # [B, N, N, 2]  upper-tri noisy offsets
        t_idx: torch.Tensor,       # [B]
        node_mask: torch.Tensor,   # [B, N] float, 1=valid node
    ):
        B, N = x_t.shape[:2]
        device = x_t.device

        t_emb = self.time_mlp(sinusoidal_emb(t_idx, self.d_model))  # [B, D]

        # ── Step 1-2: project + CNN ───────────────────────────────────
        feat = self.in_proj(x_t)   # [B, N, N, D]
        feat = feat + self.row_pos(torch.arange(N, device=device))[None, :, None, :]
        feat = feat + self.col_pos(torch.arange(N, device=device))[None, None, :, :]
        feat = feat + t_emb[:, None, None, :]

        x2d = feat.reshape(B * N, N, self.d_model).permute(0, 2, 1)   # [B*N, D, N]
        conv = self.cnn(x2d).permute(0, 2, 1).reshape(B, N, N, self.d_model)
        feat_cnn = feat + self.cnn_norm(conv)                           # [B, N, N, D]

        # ── Step 3: per-row attention -> attention mask ───────────────
        kpm = self._build_kpm(B, N, node_mask, device)  # [B*N, N]
        attn_out = self.mask_attn(
            feat_cnn.reshape(B * N, N, self.d_model), key_padding_mask=kpm
        ).reshape(B, N, N, self.d_model)

        logit = self.mask_head(attn_out).squeeze(-1)  # [B, N, N]

        # valid positions for softmax: j > i, both nodes valid
        upper = ~torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))  # j>i
        valid = upper[None] & (node_mask[:, :, None] * node_mask[:, None, :]).bool()
        atten = F.softmax(logit.masked_fill(~valid, float("-inf")), dim=-1)
        atten = torch.nan_to_num(atten, nan=0.0)  # [B, N, N]

        # ── Step 4: dot mask ─────────────────────────────────────────
        feat_dot = feat_cnn * atten.unsqueeze(-1)  # [B, N, N, D]

        # ── Step 5: post attention ───────────────────────────────────
        feat_out = self.post_attn(
            feat_dot.reshape(B * N, N, self.d_model), key_padding_mask=kpm
        ).reshape(B, N, N, self.d_model)

        pred_noise = self.out_head(feat_out)  # [B, N, N, 2]
        return pred_noise, logit


# ─── data loading ──────────────────────────────────────────────────────────────

def build_tensors(path: Path, cache: Path, n_max: int, coord_scale: float):
    if cache.exists():
        print(f"tensor cache hit: {cache}")
        d = torch.load(cache, map_location="cpu")
        return d["rel"], d["adj"], d["node_mask"]

    print(f"building tensors from {path} ...")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))

    n = len(records)
    rel_t = torch.zeros(n, n_max, n_max, 2, dtype=torch.float32)
    adj_t = torch.zeros(n, n_max, n_max, dtype=torch.float32)
    nmask = torch.zeros(n, n_max, dtype=torch.float32)

    for idx, r in enumerate(records):
        nm = r["node_mask"]
        nmask[idx, :len(nm)] = torch.tensor(nm[:n_max], dtype=torch.float32)

        for i, row in enumerate(r["rel_adj_half"]):
            if i >= n_max:
                break
            for k, val in enumerate(row):
                j = i + k
                if j < n_max:
                    rel_t[idx, i, j, 0] = val[0] / coord_scale
                    rel_t[idx, i, j, 1] = val[1] / coord_scale

        for i, row in enumerate(r["adj_matrix"]):
            if i >= n_max:
                break
            for k, val in enumerate(row):
                j = i + k
                if j < n_max:
                    adj_t[idx, i, j] = float(val)

        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{n}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rel": rel_t, "adj": adj_t, "node_mask": nmask}, cache)
    print(f"tensor cache saved: {cache}")
    return rel_t, adj_t, nmask


# ─── training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes_rel_half.jsonl"))
    p.add_argument("--save", type=Path, default=Path("core/weights/rel_adj_uncond.pt"))
    p.add_argument("--cache", type=Path, default=Path("data/jsonl/train_nodes_rel_half.tensors.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-max", type=int, default=40)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--adj-lambda", type=float, default=0.1)
    p.add_argument("--coord-scale", type=float, default=256.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    torch.manual_seed(42)

    rel, adj, nmask = build_tensors(args.data, args.cache, args.n_max, args.coord_scale)
    n = len(rel)
    val_n = max(1, int(n * args.val_ratio))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[val_n:], perm[:val_n]

    pin = device.type == "cuda"
    nw = args.num_workers
    train_ds = TensorDataset(rel[train_idx], adj[train_idx], nmask[train_idx])
    val_ds   = TensorDataset(rel[val_idx],   adj[val_idx],   nmask[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))

    model = RelAdjUncondModel(n_max=args.n_max, d_model=args.d_model,
                              n_heads=args.n_heads).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)

    N = args.n_max
    # upper[i,j] = True when j>i; used for loss masking
    upper_tri = ~torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))

    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        tl, ti = 0.0, 0

        for b_rel, b_adj, b_nm in train_loader:
            b_rel = b_rel.to(device, non_blocking=True)
            b_adj = b_adj.to(device, non_blocking=True)
            b_nm  = b_nm.to(device,  non_blocking=True)
            B = b_rel.size(0)

            t_idx = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(b_rel)
            x_t   = q_sample(b_rel, t_idx, noise, alpha_bars)
            # keep lower tri zero (no signal there)
            x_t = x_t * upper_tri[None, :, :, None].float()

            # valid positions for loss: j>i, both nodes valid
            valid = upper_tri[None] & (b_nm[:, :, None] * b_nm[:, None, :]).bool()

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_noise, logit = model(x_t, t_idx, b_nm)
                loss_diff = ((pred_noise - noise)[valid.unsqueeze(-1).expand_as(pred_noise)] ** 2).mean()
                loss_adj  = F.binary_cross_entropy_with_logits(logit[valid], b_adj[valid])
                loss = loss_diff + args.adj_lambda * loss_adj

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tl += loss_diff.item() * B
            ti += B
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"ep={epoch+1:4d} step={global_step:7d} diff_loss={tl/ti:.6f}")

        sched.step()

        model.eval()
        vl, vi = 0.0, 0
        with torch.no_grad():
            for b_rel, b_adj, b_nm in val_loader:
                b_rel = b_rel.to(device, non_blocking=True)
                b_nm  = b_nm.to(device,  non_blocking=True)
                B = b_rel.size(0)
                t_idx = torch.randint(0, args.timesteps, (B,), device=device)
                noise = torch.randn_like(b_rel)
                x_t   = q_sample(b_rel, t_idx, noise, alpha_bars) * upper_tri[None, :, :, None].float()
                valid = upper_tri[None] & (b_nm[:, :, None] * b_nm[:, None, :]).bool()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_noise, _ = model(x_t, t_idx, b_nm)
                    loss_diff = ((pred_noise - noise)[valid.unsqueeze(-1).expand_as(pred_noise)] ** 2).mean()
                vl += loss_diff.item() * B
                vi += B

        train_l = tl / max(ti, 1)
        val_l   = vl / max(vi, 1)
        saved = val_l < best_val
        if saved:
            best_val = val_l
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": args.n_max, "d_model": args.d_model,
                "n_heads": args.n_heads, "timesteps": args.timesteps,
                "coord_scale": args.coord_scale, "epoch": epoch + 1,
            }, args.save)
        print(f"ep={epoch+1:4d}  train={train_l:.6f}  val={val_l:.6f}  best={best_val:.6f}{'  (saved)' if saved else ''}")

    print(f"done. best_val={best_val:.6f}  ->  {args.save}")


if __name__ == "__main__":
    main()
