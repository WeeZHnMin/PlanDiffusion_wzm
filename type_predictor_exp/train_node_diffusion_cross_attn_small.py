"""
小规模验证实验：从 jsonl 随机抽 N 条数据，验证 cross-attn x0-pred (--adj no) 能否收敛。
结构与 train_node_diffusion_cross_attn.py 完全相同，仅参数化路径与样本数。
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert",      type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--out-dir",   type=Path, default=Path("type_predictor_exp/small_exp_results"))
    parser.add_argument("--n-samples", type=int,  default=100)
    parser.add_argument("--epochs",    type=int,  default=6000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--max-length",type=int,  default=128)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--timesteps", type=int,  default=400)
    parser.add_argument("--d-model",   type=int,  default=256)
    parser.add_argument("--n-heads",   type=int,  default=4)
    parser.add_argument("--n-layers",  type=int,  default=4)
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


# ── 模型结构（与原版完全相同）────────────────────────────────────────────────

class TwoHopGCNProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, coords, a1, a2):
        agg = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)
        return self.proj(agg)


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model))

    def forward(self, x, text_kv, node_key_pad=None, text_key_pad=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=node_key_pad)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, text_kv, text_kv, key_padding_mask=text_key_pad)[0]
        x = x + self.ffn(self.norm3(x))
        return x


class NodeDiffusionCrossAttn(nn.Module):
    def __init__(self, n_max, d_model, n_heads, n_layers, text_hidden):
        super().__init__()
        self.n_max   = n_max
        self.d_model = d_model
        self.gcn_proj     = TwoHopGCNProj(d_model)
        self.pos_emb      = nn.Embedding(n_max, d_model)
        self.time_proj    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.text_kv_proj = nn.Linear(text_hidden, d_model)
        self.layers       = nn.ModuleList([CrossAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm   = nn.LayerNorm(d_model)
        self.out          = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, text_enc, text_pad_mask, node_mask):
        pos_idx  = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)
        time_emb = self.time_proj(sinusoidal_emb(t_idx, self.d_model)).unsqueeze(1)
        # --adj no: a1=a2=zeros, GCN聚合为零，节点只用自身坐标
        zeros = torch.zeros_like(noisy_coords)
        x = self.gcn_proj(noisy_coords, zeros, zeros) + self.pos_emb(pos_idx) + time_emb

        text_kv      = self.text_kv_proj(text_enc)
        text_key_pad = (text_pad_mask == 0)
        node_key_pad = (node_mask == 0)

        for layer in self.layers:
            x = layer(x, text_kv, node_key_pad=node_key_pad, text_key_pad=text_key_pad)
        return self.out(self.final_norm(x))


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载并采样数据 ────────────────────────────────────────────────────────
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
    N     = len(records)
    N_MAX = max(len(r["adj_matrix"]) for r in records)
    print(f"sampled={N}, N_MAX={N_MAX}, device={device}")

    # ── 全局归一化（与原版相同）──────────────────────────────────────────────
    all_x = [c[0] for r in records for c in r["node_coords"][:r["n_nodes"]]]
    all_y = [c[1] for r in records for c in r["node_coords"][:r["n_nodes"]]]
    xmin, xrange = min(all_x), max(all_x) - min(all_x)
    ymin, yrange = min(all_y), max(all_y) - min(all_y)

    def norm_xy(x, y):
        return 2*(x-xmin)/max(xrange,1)-1, 2*(y-ymin)/max(yrange,1)-1
    def denorm_x(v): return np.round((v+1)/2*xrange+xmin).astype(int)
    def denorm_y(v): return np.round((v+1)/2*yrange+ymin).astype(int)

    coords_raw = torch.zeros(N, N_MAX, 2, device=device)
    node_masks = torch.zeros(N, N_MAX,    device=device)
    for i, r in enumerate(records):
        n = int(r["n_nodes"])
        node_masks[i, :n] = 1.0
        for k, (x, y) in enumerate(r["node_coords"][:n]):
            nx, ny = norm_xy(x, y)
            coords_raw[i, k, 0] = nx
            coords_raw[i, k, 1] = ny

    # ── BERT 全冻结（与原版相同）────────────────────────────────────────────
    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert      = BertModel.from_pretrained(str(args.bert)).to(device)
    for p in bert.parameters():
        p.requires_grad = False
    bert.eval()

    print("encoding prompts...")
    with torch.no_grad():
        enc = tokenizer([r["prompt"] for r in records],
                        return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_length)
        enc       = {k: v.to(device) for k, v in enc.items()}
        out       = bert(**enc)
        text_encs = out.last_hidden_state          # (N, seq_len, hidden)
        text_mask = enc["attention_mask"].float()  # (N, seq_len)
    print(f"text_encs={tuple(text_encs.shape)}")

    # ── 扩散调度 ─────────────────────────────────────────────────────────────
    T           = args.timesteps
    alpha_bars  = cosine_alpha_bars(T).to(device)
    alphas      = alpha_bars / torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    betas       = (1.0 - alphas).clamp(max=0.999)
    ab_prev     = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    post_c1     = ab_prev.sqrt() * betas / (1.0 - alpha_bars)
    post_c2     = alphas.sqrt() * (1.0 - ab_prev) / (1.0 - alpha_bars)
    post_var    = betas * (1.0 - ab_prev) / (1.0 - alpha_bars)

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = NodeDiffusionCrossAttn(
        n_max=N_MAX, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, text_hidden=bert.config.hidden_size,
    ).to(device)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    mask3 = node_masks.unsqueeze(-1)

    # ── 训练 ─────────────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        t_idx   = torch.randint(0, T, (N,), device=device)
        noise   = torch.randn_like(coords_raw)
        x_t     = q_sample(coords_raw, t_idx, noise, alpha_bars) * mask3
        pred_x0 = model(x_t, t_idx, text_encs, text_mask, node_masks)
        loss    = ((pred_x0 - coords_raw) ** 2 * mask3).sum() / mask3.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (epoch + 1) % 500 == 0:
            print(f"  epoch={epoch+1:5d}  loss={loss.item():.6f}")

    # ── 推理（DDPM reverse）─────────────────────────────────────────────────
    print("running inference...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(N, N_MAX, 2, device=device) * mask3
        for step in reversed(range(T)):
            t_idx   = torch.full((N,), step, dtype=torch.long, device=device)
            pred_x0 = model(x, t_idx, text_encs, text_mask, node_masks) * mask3
            pred_x0 = pred_x0.clamp(-1.5, 1.5)
            if step == 0:
                x = pred_x0
            else:
                mean = post_c1[step] * pred_x0 + post_c2[step] * x
                x    = mean + post_var[step].sqrt() * torch.randn_like(x) * mask3
        pred_coords = x

    # ── MAE ──────────────────────────────────────────────────────────────────
    pred_np  = pred_coords.cpu().numpy()
    gt_np    = coords_raw.cpu().numpy()
    mask_np  = node_masks.cpu().numpy()
    adj_np   = np.array([r["adj_matrix"] for r in records])

    total_err, total_cnt = 0.0, 0
    for i in range(N):
        n  = int(mask_np[i].sum())
        px = denorm_x(pred_np[i, :n, 0]); py = denorm_y(pred_np[i, :n, 1])
        gx = denorm_x(gt_np[i,   :n, 0]); gy = denorm_y(gt_np[i,   :n, 1])
        total_err += np.abs(px - gx).sum() + np.abs(py - gy).sum()
        total_cnt += n * 2
    mae = total_err / max(total_cnt, 1)
    print(f"\n整体 MAE（像素）= {mae:.2f}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    viz_n   = min(N, 20)
    viz_min = min(all_x) - 5
    viz_max = max(all_x) + 5

    def draw_graph(ax, xs, ys, adj, n, color, title):
        for a in range(n):
            for b in range(a+1, n):
                if adj[a, b] > 0.5:
                    ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                            color="steelblue", lw=1, alpha=0.6, zorder=1)
        ax.scatter(xs, ys, c=color, s=40, zorder=3)
        ax.set_title(title, fontsize=7)
        ax.set_xlim(viz_min, viz_max); ax.set_ylim(viz_max, viz_min)
        ax.set_aspect("equal")

    fig, axes = plt.subplots(viz_n, 2, figsize=(8, viz_n * 3.5))
    if viz_n == 1:
        axes = axes[None, :]
    for i in range(viz_n):
        r  = records[i]
        n  = int(mask_np[i].sum())
        gx = denorm_x(gt_np[i,   :n, 0]); gy = denorm_y(gt_np[i,   :n, 1])
        px = denorm_x(pred_np[i, :n, 0]); py = denorm_y(pred_np[i, :n, 1])
        per_mae = (np.abs(gx-px).mean() + np.abs(gy-py).mean()) / 2.0
        draw_graph(axes[i, 0], gx, gy, adj_np[i], n, "green",
                   f"[{i+1}] GT  |  {r['prompt'][:45]}")
        draw_graph(axes[i, 1], px, py, adj_np[i], n, "tomato",
                   f"Pred  MAE={per_mae:.1f}px")

    fig.suptitle(f"cross-attn x0-pred adj=no  N={N}  MAE={mae:.2f}px", fontsize=11)
    fig.tight_layout()
    out_path = args.out_dir / "result_small_exp.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"图像已保存 → {out_path}")


if __name__ == "__main__":
    main()
