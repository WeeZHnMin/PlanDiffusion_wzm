"""
Unconditional node diffusion — no text encoder, no cross-attention.
Diagnostic baseline: if this can't learn, the diffusion setup itself is broken.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--save", type=Path, default=Path("type_predictor_exp/weights/diffusion_uncond.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    return parser.parse_args()


def load_jsonl_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    if not records:
        raise SystemExit(f"No valid records in {path}")
    return records


def cosine_alpha_bars(timesteps: int, s: float = 0.008):
    ts = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((ts / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def q_sample(x0, t_idx, noise, alpha_bars):
    ab = alpha_bars[t_idx].view(-1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class NodeDiffusionUncond(nn.Module):
    def __init__(self, n_max: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.n_max = n_max
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([SelfAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, node_mask):
        pos_idx = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)
        time_emb = self.time_proj(sinusoidal_emb(t_idx, self.d_model)).unsqueeze(1)
        x = self.input_proj(noisy_coords) + self.pos_emb(pos_idx) + time_emb
        pad_mask = (node_mask == 0)
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        return self.out(self.final_norm(x))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(42)

    print("loading records...")
    records = load_jsonl_records(args.data)
    n_total = len(records)
    n_max = len(records[0]["adj_matrix"])

    val_size = max(1, int(n_total * args.val_ratio))
    train_n = n_total - val_size
    print(f"records={n_total}, train={train_n}, val={val_size}, n_max={n_max}, device={device}, amp={use_amp}")

    print("building tensors...")
    coords_raw = torch.zeros((n_total, n_max, 2), dtype=torch.float32)
    node_masks = torch.zeros((n_total, n_max), dtype=torch.float32)

    for i, r in enumerate(records):
        n_nodes = int(r["n_nodes"])
        node_masks[i, :n_nodes] = 1.0
        raw = r["node_coords"][:n_nodes]
        xs = [c[0] for c in raw]
        ys = [c[1] for c in raw]
        xmin_r = min(xs); xrange_r = max(max(xs) - min(xs), 1)
        ymin_r = min(ys); yrange_r = max(max(ys) - min(ys), 1)
        for k, (x, y) in enumerate(raw):
            coords_raw[i, k, 0] = 2.0 * (x - xmin_r) / xrange_r - 1.0
            coords_raw[i, k, 1] = 2.0 * (y - ymin_r) / yrange_r - 1.0
        if (i + 1) % 10000 == 0:
            print(f"  prepared {i + 1}/{n_total}")

    dataset = TensorDataset(coords_raw, node_masks)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_size], generator=torch.Generator().manual_seed(42)
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            persistent_workers=(args.num_workers > 0),
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    model = NodeDiffusionUncond(n_max=n_max, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for b_coords, b_mask in train_loader:
            b_coords = b_coords.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            t_idx = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
            noise = torch.randn_like(b_coords)
            mask3 = b_mask.unsqueeze(-1)
            x_t = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_eps = model(x_t, t_idx, b_mask)
                loss = ((pred_eps - noise) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss_sum += loss.item() * b_coords.size(0)
            train_items += b_coords.size(0)
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch={epoch+1:4d} step={global_step:7d} train_loss={train_loss_sum/max(train_items,1):.6f}")

        sched.step()

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for b_coords, b_mask in val_loader:
                b_coords = b_coords.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                t_idx = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
                noise = torch.randn_like(b_coords)
                mask3 = b_mask.unsqueeze(-1)
                x_t = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_eps = model(x_t, t_idx, b_mask)
                    loss = ((pred_eps - noise) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)
                val_loss_sum += loss.item() * b_coords.size(0)
                val_items += b_coords.size(0)

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss = val_loss_sum / max(val_items, 1)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": n_max, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "timesteps": args.timesteps, "pred_type": "epsilon",
                "norm_type": "per_record", "epoch": epoch + 1,
                "best_val_loss": best_val,
            }, args.save)
        print(f"epoch={epoch+1:4d} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f} {'(saved)' if improved else ''}")

    print(f"best_saved={args.save}, best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
