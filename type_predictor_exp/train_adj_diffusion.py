"""
邻接矩阵扩散学习。
数据: data/jsonl/mapped_node_data.jsonl
  - node_types : 每节点房间类型字符串 (5类)
  - adj_matrix : 01对称邻接矩阵，形状 n_max x n_max
  - node_mask  : 有效节点掩码

节点特征 X = one-hot(room_type)  →  只扩散 A (adj)，X 作为固定条件。
x0 预测，纯 MSE loss，对称输出，对角线清零。
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 房间类型 ──────────────────────────────────────────────────────────────────
ROOM_TYPES = ["bedroom", "bathroom", "living_room", "kitchen", "corridor"]
TYPE2IDX   = {t: i for i, t in enumerate(ROOM_TYPES)}
N_TYPES    = len(ROOM_TYPES)   # 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path, default=Path("data/jsonl/mapped_node_data.jsonl"))
    parser.add_argument("--save",       type=Path, default=Path("type_predictor_exp/weights/adj_diffusion.pt"))
    parser.add_argument("--n-samples",  type=int,  default=0,    help="0 = all")
    parser.add_argument("--epochs",     type=int,  default=200)
    parser.add_argument("--batch-size", type=int,  default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd",         type=float, default=1e-2)
    parser.add_argument("--timesteps",  type=int,  default=400)
    parser.add_argument("--d-model",    type=int,  default=256)
    parser.add_argument("--n-heads",    type=int,  default=4)
    parser.add_argument("--n-layers",   type=int,  default=8)
    parser.add_argument("--val-ratio",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--amp",        action="store_true", default=True)
    parser.add_argument("--no-amp",     action="store_false", dest="amp")
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

class AdaLayerNorm(nn.Module):
    def __init__(self, d_model, d_time):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_time, d_model * 2)

    def forward(self, x, t_emb):
        ss = self.proj(t_emb).unsqueeze(1)          # (B, 1, 2*d)
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


class AdjDiffusionNet(nn.Module):
    """
    输入:
      noisy_adj  (B, N, N)  — 加噪后的邻接矩阵
      type_idx   (B, N)     — 房间类型整数索引 (0..4, padding=-1)
      t_idx      (B,)       — 时间步
      node_mask  (B, N)     — 有效节点 0/1

    输出:
      pred_adj   (B, N, N)  — 预测的干净邻接矩阵 A_0
    """

    def __init__(self, n_max, n_types, d_model, n_heads, n_layers):
        super().__init__()
        self.n_max   = n_max
        self.d_model = d_model
        d_time = d_model * 4

        # 节点类型嵌入 (padding_idx=n_types → 零向量)
        self.type_emb  = nn.Embedding(n_types + 1, d_model, padding_idx=n_types)
        # 每行邻接矩阵投影到 d_model（感知当前图结构）
        self.adj_proj  = nn.Linear(n_max, d_model)
        # 位置嵌入
        self.pos_emb   = nn.Embedding(n_max, d_model)
        # 时间投影
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_time), nn.SiLU(), nn.Linear(d_time, d_time))

        self.layers = nn.ModuleList(
            [SelfAttnLayer(d_model, n_heads, d_time) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # 边分类 MLP: [h_i, h_j, h_i*h_j] → 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.SiLU(),
            nn.Linear(d_model, 1))

    def forward(self, noisy_adj, type_idx, t_idx, node_mask):
        B, N = node_mask.shape
        pos  = torch.arange(N, device=noisy_adj.device).unsqueeze(0)   # (1, N)
        t_emb = self.time_proj(sinusoidal_emb(t_idx, self.d_model))    # (B, d_time)

        x  = self.type_emb(type_idx)          # (B, N, d)
        x  = x + self.adj_proj(noisy_adj)     # 每行 adj 当作 N 维特征
        x  = x + self.pos_emb(pos)            # 位置嵌入

        key_pad = (node_mask == 0)             # (B, N) bool
        for layer in self.layers:
            x = layer(x, t_emb, key_pad=key_pad)
        x = self.final_norm(x)                 # (B, N, d)

        # 从节点对特征预测边存在概率
        h_i = x.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, d)
        h_j = x.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, d)
        edge_feat = torch.cat([h_i, h_j, h_i * h_j], dim=-1)  # (B, N, N, 3d)
        edge_score = self.edge_mlp(edge_feat).squeeze(-1)       # (B, N, N)

        # 强制对称，清零对角线
        edge_score = (edge_score + edge_score.transpose(1, 2)) * 0.5
        diag_mask  = torch.eye(N, device=noisy_adj.device, dtype=torch.bool).unsqueeze(0)
        edge_score = edge_score.masked_fill(diag_mask, 0.0)

        # 用 node_mask 清除无效节点对应的边
        valid_pair = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)  # (B, N, N)
        edge_score = edge_score * valid_pair

        return edge_score


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def load_tensors(records, n_max):
    N = len(records)
    adj_t   = torch.zeros((N, n_max, n_max), dtype=torch.float32)
    type_t  = torch.full( (N, n_max),        N_TYPES, dtype=torch.long)   # padding = N_TYPES
    mask_t  = torch.zeros((N, n_max),        dtype=torch.float32)

    for i, r in enumerate(records):
        n = int(r["n_nodes"])
        mask_t[i, :n] = 1.0

        # 节点类型
        for k, t in enumerate(r["node_types"][:n]):
            type_t[i, k] = TYPE2IDX.get(t, N_TYPES - 1)   # unknown → corridor(4)

        # 邻接矩阵（截断到 n_max）
        rows = r["adj_matrix"]
        for row_i, row in enumerate(rows[:n_max]):
            for col_j, v in enumerate(row[:n_max]):
                adj_t[i, row_i, col_j] = float(v)

    return adj_t, type_t, mask_t


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
                all_records.append(json.loads(s))

    if args.n_samples > 0:
        records = random.sample(all_records, min(args.n_samples, len(all_records)))
    else:
        records = all_records
    n_max = max(len(r["adj_matrix"]) for r in records)
    print(f"total={len(records)}  n_max={n_max}  device={device}  amp={use_amp}")

    adj_t, type_t, mask_t = load_tensors(records, n_max)
    del all_records, records

    dataset = TensorDataset(adj_t, type_t, mask_t)
    val_n   = max(1, int(len(dataset) * args.val_ratio))
    train_n = len(dataset) - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    pin          = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    model = AdjDiffusionNet(n_max, N_TYPES, args.d_model, args.n_heads, args.n_layers).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val   = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0; tr_n = 0
        for b_adj, b_type, b_mask in train_loader:
            b_adj  = b_adj.to(device,  non_blocking=True)
            b_type = b_type.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)

            t_idx = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
            noise = torch.randn_like(b_adj)
            # 保持对称性: 加噪时也对称
            noise = (noise + noise.transpose(1, 2)) * 0.5
            noisy = q_sample(b_adj, t_idx, noise, alpha_bars)
            # mask 无效节点对
            valid_pair = b_mask.unsqueeze(2) * b_mask.unsqueeze(1)
            noisy = noisy * valid_pair

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(noisy, b_type, t_idx, b_mask)
                loss = ((pred - b_adj) ** 2 * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * b_adj.size(0)
            tr_n    += b_adj.size(0)
        sched.step()

        model.eval()
        vl_loss = 0.0; vl_n = 0
        with torch.no_grad():
            for b_adj, b_type, b_mask in val_loader:
                b_adj  = b_adj.to(device,  non_blocking=True)
                b_type = b_type.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)

                t_idx = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
                noise = torch.randn_like(b_adj)
                noise = (noise + noise.transpose(1, 2)) * 0.5
                noisy = q_sample(b_adj, t_idx, noise, alpha_bars)
                valid_pair = b_mask.unsqueeze(2) * b_mask.unsqueeze(1)
                noisy = noisy * valid_pair

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(noisy, b_type, t_idx, b_mask)
                    loss = ((pred - b_adj) ** 2 * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)
                vl_loss += loss.item() * b_adj.size(0)
                vl_n    += b_adj.size(0)

        tl = tr_loss / max(tr_n, 1)
        vl = vl_loss / max(vl_n, 1)
        improved = vl < best_val
        if improved:
            best_val = vl
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": n_max, "n_types": N_TYPES,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "timesteps": args.timesteps,
                "epoch": epoch + 1,
            }, args.save)

        print(f"epoch={epoch+1:4d}  train={tl:.6f}  val={vl:.6f}  "
              f"best_val={best_val:.6f}  {'(saved)' if improved else ''}")

    print(f"\ndone.  best_val={best_val:.6f}  saved → {args.save}")


if __name__ == "__main__":
    main()
