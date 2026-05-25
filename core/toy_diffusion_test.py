"""
Toy diffusion sanity check.
Data: 8000 sequences of length 80, each position ~ U(50, 250).
If diffusion works, loss should drop well below 1.0.
If model collapses, loss stays near 1.0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── data ────────────────────────────────────────────────────────────────────────

def make_data(n=8000, seq_len=80, lo=50.0, hi=250.0):
    x = torch.rand(n, seq_len) * (hi - lo) + lo
    # normalize to [-1, 1]
    x = (x - (hi + lo) / 2) / ((hi - lo) / 2)
    return x   # [8000, 80]


# ── diffusion ───────────────────────────────────────────────────────────────────

def cosine_alpha_bars(timesteps, s=0.008):
    ts = torch.arange(timesteps + 1, dtype=torch.float64)
    f  = torch.cos((ts / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def q_sample(x0, t_idx, noise, alpha_bars):
    ab = alpha_bars[t_idx].view(-1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ── model ───────────────────────────────────────────────────────────────────────

class ToyDiffModel(nn.Module):
    """Simple transformer: sequence [B, L] -> predicted noise [B, L]."""
    def __init__(self, seq_len=80, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.in_proj  = nn.Linear(1, d_model)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x_t, t_idx):
        # x_t: [B, L]
        B, L = x_t.shape
        feat = self.in_proj(x_t.unsqueeze(-1))                           # [B, L, D]
        feat = feat + self.pos_emb(torch.arange(L, device=x_t.device))  # pos
        feat = feat + self.time_mlp(sinusoidal_emb(t_idx, feat.shape[-1])).unsqueeze(1)  # time
        feat = self.encoder(feat)                                         # [B, L, D]
        return self.out_proj(feat).squeeze(-1)                            # [B, L]


# ── training ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = 1000
    epochs    = 8000
    batch_size = 128
    lr        = 3e-4

    data       = make_data().to(device)          # [8000, 80]
    alpha_bars = cosine_alpha_bars(timesteps).to(device)
    model      = ToyDiffModel().to(device)
    opt        = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    print(f"device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"data shape={list(data.shape)}  range=[{data.min():.2f}, {data.max():.2f}]")
    print(f"baseline MSE (predict-zero) = 1.0000  ← model must beat this\n")

    n = data.size(0)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            x0    = data[perm[i:i + batch_size]]
            B     = x0.size(0)
            t_idx = torch.randint(0, timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t   = q_sample(x0, t_idx, noise, alpha_bars)

            pred  = model(x_t, t_idx)
            loss  = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item() * B
            steps      += B

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"ep={epoch+1:5d}  loss={epoch_loss/steps:.6f}")


if __name__ == "__main__":
    main()
