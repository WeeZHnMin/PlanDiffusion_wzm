"""
邻接矩阵扩散学习 —— DiGress XEy Transformer 架构。

数据: data/jsonl/mapped_node_data.jsonl
  X : one-hot(room_type)          (B, N, n_types)   节点特征，固定不扩散
  E : noisy adj + 原始 adj 拼接   (B, N, N, 2)      边特征，扩散目标
  y : sinusoidal time embedding   (B, dy)            全局特征

每层 XEyTransformerLayer 同时更新 X / E / y，
输出 E_out[:,:,:,0] 作为预测的干净邻接矩阵 A_0。
x0 预测，纯 MSE loss。
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
N_TYPES    = len(ROOM_TYPES)   # 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path,  default=Path("data/jsonl/mapped_node_data.jsonl"))
    parser.add_argument("--save",       type=Path,  default=Path("type_predictor_exp/weights/adj_diffusion.pt"))
    parser.add_argument("--n-samples",  type=int,   default=0,    help="0 = all")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd",         type=float, default=1e-2)
    parser.add_argument("--timesteps",  type=int,   default=400)
    parser.add_argument("--dx",         type=int,   default=256,  help="node hidden dim")
    parser.add_argument("--de",         type=int,   default=64,   help="edge hidden dim")
    parser.add_argument("--dy",         type=int,   default=256,  help="global/time hidden dim")
    parser.add_argument("--n-heads",    type=int,   default=4)
    parser.add_argument("--n-layers",   type=int,   default=6)
    parser.add_argument("--val-ratio",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
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


# ── DiGress XEy Transformer ───────────────────────────────────────────────────

def masked_softmax(x, mask, dim):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, dim=dim)


class Xtoy(nn.Module):
    """节点特征聚合为全局特征 (mean/min/max/std)"""
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        m   = X.mean(dim=1)
        mi  = X.min(dim=1)[0]
        ma  = X.max(dim=1)[0]
        std = X.std(dim=1)
        return self.lin(torch.cat([m, mi, ma, std], dim=-1))


class Etoy(nn.Module):
    """边特征聚合为全局特征"""
    def __init__(self, de, dy):
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E):
        m   = E.mean(dim=(1, 2))
        mi  = E.min(dim=2)[0].min(dim=1)[0]
        ma  = E.max(dim=2)[0].max(dim=1)[0]
        std = E.std(dim=(1, 2))
        return self.lin(torch.cat([m, mi, ma, std], dim=-1))


class NodeEdgeBlock(nn.Module):
    """
    DiGress 核心 attention block。
    每个节点对 (i,j) 的边特征 E[i,j] 通过 FiLM 调制 Q_i * K_j 的注意力分数，
    调制后的分数再反过来更新边特征。节点、边、全局三路协同演化。
    """
    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0
        self.dx     = dx
        self.de     = de
        self.df     = dx // n_head
        self.n_head = n_head

        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM: 边特征 → 调制注意力分数
        self.e_mul = nn.Linear(de, dx)
        self.e_add = nn.Linear(de, dx)

        # FiLM: 全局特征 → 调制边
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM: 全局特征 → 调制节点
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # 全局特征更新
        self.y_y  = nn.Linear(dy, dy)
        self.x_y  = Xtoy(dx, dy)
        self.e_y  = Etoy(de, dy)

        # 输出投影
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        X : (B, N, dx)
        E : (B, N, N, de)
        y : (B, dy)
        node_mask : (B, N)  1=有效 0=padding
        """
        B, N, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)          # (B, N, 1)
        e_mask1 = x_mask.unsqueeze(2)             # (B, N, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)             # (B, 1, N, 1)

        Q = self.q(X) * x_mask                   # (B, N, dx)
        K = self.k(X) * x_mask

        Q = Q.view(B, N, self.n_head, self.df).unsqueeze(2)   # (B, 1, N, n_head, df)
        K = K.view(B, N, self.n_head, self.df).unsqueeze(1)   # (B, N, 1, n_head, df)

        Y = Q * K / math.sqrt(self.df)            # (B, N, N, n_head, df)

        E1 = self.e_mul(E) * e_mask1 * e_mask2   # (B, N, N, dx)
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E1 = E1.view(B, N, N, self.n_head, self.df)
        E2 = E2.view(B, N, N, self.n_head, self.df)

        # 边特征 FiLM 调制注意力分数
        Y = Y * (E1 + 1) + E2                    # (B, N, N, n_head, df)

        # 注意力分数 → 新边特征
        newE = Y.flatten(start_dim=3)             # (B, N, N, dx)
        ye1  = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dx)
        ye2  = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2       # (B, N, N, de)

        # masked softmax：无效节点不参与 softmax
        softmax_mask = e_mask2.expand(B, N, N, self.n_head)  # (B, N, N, n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)         # (B, N, N, n_head, df) softmax over dim=2

        V = self.v(X) * x_mask                   # (B, N, dx)
        V = V.view(B, N, self.n_head, self.df).unsqueeze(1)  # (B, 1, N, n_head, df)

        weighted_V = (attn * V).sum(dim=2)        # (B, N, n_head, df)
        weighted_V = weighted_V.flatten(start_dim=2)  # (B, N, dx)

        # 全局特征 FiLM 调制节点
        yx1  = self.y_x_add(y).unsqueeze(1)
        yx2  = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask          # (B, N, dx)

        # 更新全局特征
        new_y = self.y_out(self.y_y(y) + self.x_y(X) + self.e_y(E))

        return newX, newE, new_y


class XEyTransformerLayer(nn.Module):
    def __init__(self, dx, de, dy, n_head, dim_ffX=512, dim_ffE=128, dim_ffy=512):
        super().__init__()
        self.attn   = NodeEdgeBlock(dx, de, dy, n_head)
        self.normX1 = nn.LayerNorm(dx); self.normX2 = nn.LayerNorm(dx)
        self.normE1 = nn.LayerNorm(de); self.normE2 = nn.LayerNorm(de)
        self.norm_y1 = nn.LayerNorm(dy); self.norm_y2 = nn.LayerNorm(dy)
        self.ffX = nn.Sequential(nn.Linear(dx, dim_ffX), nn.ReLU(), nn.Linear(dim_ffX, dx))
        self.ffE = nn.Sequential(nn.Linear(de, dim_ffE), nn.ReLU(), nn.Linear(dim_ffE, de))
        self.ffy = nn.Sequential(nn.Linear(dy, dim_ffy), nn.ReLU(), nn.Linear(dim_ffy, dy))

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.attn(X, E, y, node_mask)
        X = self.normX1(X + newX)
        E = self.normE1(E + newE)
        y = self.norm_y1(y + new_y)
        X = self.normX2(X + self.ffX(X))
        E = self.normE2(E + self.ffE(E))
        y = self.norm_y2(y + self.ffy(y))
        return X, E, y


class AdjDiffusionNet(nn.Module):
    """
    输入:
      noisy_adj  (B, N, N)  加噪邻接矩阵
      type_onehot(B, N, n_types)  房间类型 one-hot（固定，不扩散）
      t_idx      (B,)       时间步
      node_mask  (B, N)     有效节点掩码

    输出:
      pred_adj   (B, N, N)  预测干净邻接矩阵 A_0
    """
    def __init__(self, n_max, n_types, dx, de, dy, n_head, n_layers):
        super().__init__()
        self.n_max   = n_max
        E_in  = 2        # [noisy_adj, 原始adj拼接] → 这里只用1维(noisy_adj标量)
        E_in  = 1

        # 输入投影
        self.mlp_in_X = nn.Sequential(
            nn.Linear(n_types, dx), nn.ReLU(), nn.Linear(dx, dx))
        self.mlp_in_E = nn.Sequential(
            nn.Linear(E_in, de), nn.ReLU(), nn.Linear(de, de))
        self.mlp_in_y = nn.Sequential(
            nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        # 时间嵌入
        self.time_proj = nn.Sequential(
            nn.Linear(dy, dy * 2), nn.SiLU(), nn.Linear(dy * 2, dy))

        self.layers = nn.ModuleList([
            XEyTransformerLayer(dx, de, dy, n_head) for _ in range(n_layers)])

        # 输出投影
        self.mlp_out_X = nn.Sequential(
            nn.Linear(dx, dx), nn.ReLU(), nn.Linear(dx, n_types))
        self.mlp_out_E = nn.Sequential(
            nn.Linear(de, de), nn.ReLU(), nn.Linear(de, 1))

    def forward(self, noisy_adj, type_onehot, t_idx, node_mask):
        B, N = node_mask.shape
        dy   = self.mlp_in_y[0].in_features

        # 时间步 → 全局特征
        t_sin = sinusoidal_emb(t_idx, dy)          # (B, dy)
        y     = self.mlp_in_y(self.time_proj(t_sin))  # (B, dy)

        X = self.mlp_in_X(type_onehot)                 # (B, N, dx)
        E = self.mlp_in_E(noisy_adj.unsqueeze(-1))     # (B, N, N, de)

        # 对称 + mask
        E = (E + E.transpose(1, 2)) * 0.5
        x_mask = node_mask.unsqueeze(-1)
        e_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        X = X * x_mask
        E = E * e_mask.unsqueeze(-1)

        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)

        pred_X   = self.mlp_out_X(X)               # (B, N, n_types)
        pred_adj = self.mlp_out_E(E).squeeze(-1)   # (B, N, N)

        # 强制对称，清零对角线，mask 无效节点对
        pred_adj = (pred_adj + pred_adj.transpose(1, 2)) * 0.5
        diag     = torch.eye(N, device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
        pred_adj = pred_adj.masked_fill(diag, 0.0)
        pred_adj = pred_adj * e_mask

        return pred_adj, pred_X


# ── 数据加载 ───────────────────────────────────────────────────────────────────

def load_tensors(records, n_max):
    N = len(records)
    adj_t    = torch.zeros((N, n_max, n_max), dtype=torch.float32)
    onehot_t = torch.zeros((N, n_max, N_TYPES), dtype=torch.float32)
    mask_t   = torch.zeros((N, n_max),          dtype=torch.float32)

    for i, r in enumerate(records):
        n = int(r["n_nodes"])
        mask_t[i, :n] = 1.0

        for k, t in enumerate(r["node_types"][:n]):
            idx = TYPE2IDX.get(t, N_TYPES - 1)
            onehot_t[i, k, idx] = 1.0

        rows = r["adj_matrix"]
        for ri, row in enumerate(rows[:n_max]):
            for ci, v in enumerate(row[:n_max]):
                adj_t[i, ri, ci] = float(v)

    return adj_t, onehot_t, mask_t


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    print("loading records...")
    all_records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                all_records.append(json.loads(s))

    records = (random.sample(all_records, min(args.n_samples, len(all_records)))
               if args.n_samples > 0 else all_records)
    n_max = max(len(r["adj_matrix"]) for r in records)
    print(f"total={len(records)}  n_max={n_max}  device={device}  amp={use_amp}")

    adj_t, onehot_t, mask_t = load_tensors(records, n_max)
    del all_records, records

    dataset  = TensorDataset(adj_t, onehot_t, mask_t)
    val_n    = max(1, int(len(dataset) * args.val_ratio))
    train_n  = len(dataset) - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    pin          = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)

    model = AdjDiffusionNet(n_max, N_TYPES, args.dx, args.de, args.dy,
                            args.n_heads, args.n_layers).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val   = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0; tr_n = 0
        for b_adj, b_onehot, b_mask in train_loader:
            b_adj    = b_adj.to(device,    non_blocking=True)
            b_onehot = b_onehot.to(device, non_blocking=True)
            b_mask   = b_mask.to(device,   non_blocking=True)

            t_idx = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
            noise = torch.randn_like(b_adj)
            noise = (noise + noise.transpose(1, 2)) * 0.5   # 对称噪声
            noisy = q_sample(b_adj, t_idx, noise, alpha_bars)
            valid_pair = b_mask.unsqueeze(2) * b_mask.unsqueeze(1)
            noisy = noisy * valid_pair

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_adj, _ = model(noisy, b_onehot, t_idx, b_mask)
                loss = ((pred_adj - b_adj) ** 2 * valid_pair).sum() / \
                       valid_pair.sum().clamp(min=1.0)

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
            for b_adj, b_onehot, b_mask in val_loader:
                b_adj    = b_adj.to(device,    non_blocking=True)
                b_onehot = b_onehot.to(device, non_blocking=True)
                b_mask   = b_mask.to(device,   non_blocking=True)
                t_idx    = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
                noise    = torch.randn_like(b_adj)
                noise    = (noise + noise.transpose(1, 2)) * 0.5
                noisy    = q_sample(b_adj, t_idx, noise, alpha_bars)
                valid_pair = b_mask.unsqueeze(2) * b_mask.unsqueeze(1)
                noisy    = noisy * valid_pair
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_adj, _ = model(noisy, b_onehot, t_idx, b_mask)
                    loss = ((pred_adj - b_adj) ** 2 * valid_pair).sum() / \
                           valid_pair.sum().clamp(min=1.0)
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
                "dx": args.dx, "de": args.de, "dy": args.dy,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "timesteps": args.timesteps, "epoch": epoch + 1,
            }, args.save)

        print(f"epoch={epoch+1:4d}  train={tl:.6f}  val={vl:.6f}  "
              f"best_val={best_val:.6f}  {'(saved)' if improved else ''}")

    print(f"\ndone.  best_val={best_val:.6f}  saved -> {args.save}")


if __name__ == "__main__":
    main()
