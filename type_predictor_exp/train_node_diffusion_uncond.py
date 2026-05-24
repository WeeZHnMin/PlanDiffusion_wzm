"""
无条件节点扩散：验证纯空间建模能力。
去掉 BERT / cross-attention，只保留 self-attention + GCN + 时间嵌入。
默认抽 500 条数据，x0-prediction，观察训练集 loss 能否持续下降。
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=Path,  default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--save",         type=Path,  default=Path("type_predictor_exp/weights/uncond_diffusion.pt"))
    parser.add_argument("--n-samples",    type=int,   default=500)
    parser.add_argument("--epochs",       type=int,   default=6000)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--timesteps",    type=int,   default=400)
    parser.add_argument("--d-model",      type=int,   default=256)
    parser.add_argument("--n-heads",      type=int,   default=4)
    parser.add_argument("--n-layers",     type=int,   default=12)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--amp",          action="store_true", default=True)
    parser.add_argument("--no-amp",       action="store_false", dest="amp")
    return parser.parse_args()


# ── 扩散调度 ──────────────────────────────────────────────────────────────────

def cosine_alpha_bars(T, s=0.008):
    ts = torch.arange(T + 1, dtype=torch.float64)
    f  = torch.cos((ts / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def q_sample(x0, t_idx, noise, alpha_bars):
    ab = alpha_bars[t_idx].view(-1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / max(half - 1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ── 模型 ──────────────────────────────────────────────────────────────────────

class GCNProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, coords):
        return self.proj(coords)


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift are predicted from time embedding."""
    def __init__(self, d_model, d_time):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_time, d_model * 2)

    def forward(self, x, t_emb):
        # t_emb: (B, d_time)  →  scale/shift: (B, 1, d_model)
        ss    = self.proj(t_emb).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_time):
        super().__init__()
        self.ada1 = AdaLayerNorm(d_model, d_time)
        self.ada2 = AdaLayerNorm(d_model, d_time)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model))

    def forward(self, x, t_emb, key_pad=None):
        h = self.ada1(x, t_emb)
        x = x + self.attn(h, h, h, key_padding_mask=key_pad)[0]
        x = x + self.ffn(self.ada2(x, t_emb))
        return x


class NodeDiffusionUncond(nn.Module):
    def __init__(self, n_max, d_model, n_heads, n_layers):
        super().__init__()
        self.n_max    = n_max
        self.d_model  = d_model
        d_time        = d_model * 4
        self.gcn_proj = GCNProj(d_model)
        self.pos_emb  = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_time), nn.SiLU(), nn.Linear(d_time, d_time))
        self.layers     = nn.ModuleList([SelfAttnLayer(d_model, n_heads, d_time) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out        = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, node_mask):
        pos_idx  = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)
        t_emb    = self.time_proj(sinusoidal_emb(t_idx, self.d_model))  # (B, d_time)
        x        = self.gcn_proj(noisy_coords) + self.pos_emb(pos_idx)
        key_pad  = (node_mask == 0)
        for layer in self.layers:
            x = layer(x, t_emb, key_pad=key_pad)
        return self.out(self.final_norm(x))


# ── 主程序 ────────────────────────────────────────────────────────────────────

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
    N       = len(records)
    print(f"sampled={N}  n_max={n_max}  device={device}  amp={use_amp}")

    coords_raw = torch.zeros((N, n_max, 2), dtype=torch.float32)
    node_masks = torch.zeros((N, n_max),    dtype=torch.float32)
    for i, r in enumerate(records):
        n = int(r["n_nodes"])
        node_masks[i, :n] = 1.0
        raw  = r["node_coords"][:n]
        xs   = [c[0] for c in raw]; ys = [c[1] for c in raw]
        xmin = min(xs); xrng = max(max(xs) - xmin, 1)
        ymin = min(ys); yrng = max(max(ys) - ymin, 1)
        for k, (x, y) in enumerate(raw):
            coords_raw[i, k, 0] = 2.0 * (x - xmin) / xrng - 1.0
            coords_raw[i, k, 1] = 2.0 * (y - ymin) / yrng - 1.0

    dataset = TensorDataset(coords_raw, node_masks)
    val_n   = max(1, int(N * args.val_ratio))
    train_n = N - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    pin          = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = NodeDiffusionUncond(n_max, args.d_model, args.n_heads, args.n_layers).to(device)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val   = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0; tr_n = 0
        for b_coords, b_mask in train_loader:
            b_coords = b_coords.to(device, non_blocking=True)
            b_mask   = b_mask.to(device,   non_blocking=True)
            t_idx    = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
            noise    = torch.randn_like(b_coords)
            mask3    = b_mask.unsqueeze(-1)
            x_t      = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_t, t_idx, b_mask)
                loss = ((pred - b_coords) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * b_coords.size(0)
            tr_n    += b_coords.size(0)
        sched.step()

        model.eval()
        vl_loss = 0.0; vl_n = 0
        with torch.no_grad():
            for b_coords, b_mask in val_loader:
                b_coords = b_coords.to(device, non_blocking=True)
                b_mask   = b_mask.to(device,   non_blocking=True)
                t_idx    = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
                noise    = torch.randn_like(b_coords)
                mask3    = b_mask.unsqueeze(-1)
                x_t      = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x_t, t_idx, b_mask)
                    loss = ((pred - b_coords) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)
                vl_loss += loss.item() * b_coords.size(0)
                vl_n    += b_coords.size(0)

        tl = tr_loss / max(tr_n, 1)
        vl = vl_loss / max(vl_n, 1)
        improved = vl < best_val
        if improved:
            best_val = vl
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": n_max, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "timesteps": args.timesteps, "epoch": epoch + 1,
            }, args.save)

        print(f"epoch={epoch+1:3d}  train_loss={tl:.6f}  val_loss={vl:.6f}  "
              f"best_val={best_val:.6f}  {'(saved)' if improved else ''}")

    print(f"\ndone.  best_val={best_val:.6f}  saved → {args.save}")


if __name__ == "__main__":
    main()
