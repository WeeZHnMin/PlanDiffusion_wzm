"""
x0-prediction 版本：模型直接预测干净坐标 x0，而非噪声 ε
- 训练 loss：MSE(pred_x0, x0)，直接在坐标空间优化
- 反向扩散：用 x0 后验均值公式逐步去噪
- 其余结构（GCN、Transformer、调度）与 train_node_diffusion_gcn.py 完全相同
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BertTokenizer, BertModel

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PATH = Path(__file__).parent.parent / "models" / "bert-base-uncased"
DATA_PATH = Path(__file__).parent / "node_data.jsonl"
OUT_DIR   = Path(__file__).parent / "node_diffusion_results"
OUT_DIR.mkdir(exist_ok=True)

T        = 400
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
EPOCHS   = 6000
LR       = 1e-3

_norm_params = {}

def norm_xy(x, y):
    nx = 2 * (x - _norm_params["xmin"]) / _norm_params["xrange"] - 1
    ny = 2 * (y - _norm_params["ymin"]) / _norm_params["yrange"] - 1
    return nx, ny

def denorm_x(v): return np.round((v + 1) / 2 * _norm_params["xrange"] + _norm_params["xmin"]).astype(int)
def denorm_y(v): return np.round((v + 1) / 2 * _norm_params["yrange"] + _norm_params["ymin"]).astype(int)

def _snap_1d(vals):
    if len(vals) <= 1:
        return vals.copy().astype(int)
    idx  = np.argsort(vals)
    sv   = vals[idx].astype(float)
    gaps = np.diff(sv)
    threshold = max(float(np.median(gaps)) * 3.0, 5.0)
    out  = vals.copy().astype(float)
    gs   = 0
    for i in range(1, len(sv) + 1):
        if i == len(sv) or sv[i] - sv[gs] > threshold:
            mean_val = int(np.round(sv[gs:i].mean()))
            for j in range(gs, i):
                out[idx[j]] = mean_val
            gs = i
    return out.astype(int)

# ── DDPM cosine schedule ──────────────────────────────────────────────────────
def _cosine_alpha_bars(T, s=0.008):
    ts = torch.arange(T + 1, dtype=torch.float64)
    f  = torch.cos((ts / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)

alpha_bars = _cosine_alpha_bars(T).to(DEVICE)
alphas     = alpha_bars / torch.cat([torch.ones(1, device=DEVICE), alpha_bars[:-1]])
betas      = (1.0 - alphas).clamp(max=0.999)

# ── x0-prediction 后验系数（预计算） ─────────────────────────────────────────
# ᾱ_{t-1}，t=0 时定义为 1
alpha_bars_prev = torch.cat([torch.ones(1, device=DEVICE), alpha_bars[:-1]])

# μ̃_t = coef1 * x0_pred + coef2 * x_t
posterior_coef1 = alpha_bars_prev.sqrt() * betas / (1.0 - alpha_bars)
posterior_coef2 = alphas.sqrt() * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
posterior_var   = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

def q_sample(x0, t_idx, noise):
    ab = alpha_bars[t_idx].view(-1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

# ── 加载数据 ──────────────────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

N_MAX = len(records[0]["adj_matrix"])
N     = len(records)
print(f"Records={N}, N_MAX={N_MAX}, device={DEVICE}")

all_x = [c[0] for r in records for c in r["node_coords"][:r["n_nodes"]]]
all_y = [c[1] for r in records for c in r["node_coords"][:r["n_nodes"]]]
_norm_params["xmin"]   = min(all_x); _norm_params["xrange"] = max(all_x) - min(all_x)
_norm_params["ymin"]   = min(all_y); _norm_params["yrange"] = max(all_y) - min(all_y)
print(f"x: {min(all_x)}~{max(all_x)}, y: {min(all_y)}~{max(all_y)}")

coords_raw = torch.zeros(N, N_MAX, 2)
node_masks = torch.zeros(N, N_MAX)
for i, r in enumerate(records):
    n_nodes = r["n_nodes"]
    for k, (x, y) in enumerate(r["node_coords"][:n_nodes]):
        nx, ny = norm_xy(x, y)
        coords_raw[i, k, 0] = nx
        coords_raw[i, k, 1] = ny
    node_masks[i, :n_nodes] = 1.0

coords_raw = coords_raw.to(DEVICE)
node_masks = node_masks.to(DEVICE)

adj_tensor = torch.tensor(
    [r["adj_matrix"] for r in records], dtype=torch.float32
).to(DEVICE)

def row_normalize(adj, mask):
    deg    = adj.sum(dim=-1, keepdim=True).clamp(min=1)
    a_norm = adj / deg
    return a_norm * mask.unsqueeze(-1)

adj_norm1 = row_normalize(adj_tensor, node_masks)
adj_norm2 = row_normalize(torch.bmm(adj_norm1, adj_norm1), node_masks)

VIZ_MIN = min(all_x) - 5
VIZ_MAX = max(all_x) + 5

# ── BERT ─────────────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
bert      = BertModel.from_pretrained(str(BERT_PATH)).to(DEVICE)
for p in bert.parameters():
    p.requires_grad = False
bert.eval()

print("Encoding prompts...")
with torch.no_grad():
    cls_list = []
    for r in records:
        inp = tokenizer(r["prompt"], return_tensors="pt",
                        padding=True, truncation=True, max_length=32)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        cls_list.append(bert(**inp).last_hidden_state[0, 0])
    cls_encs = torch.stack(cls_list)
print(f"CLS: {cls_encs.shape}")

# ── 时间步嵌入 ────────────────────────────────────────────────────────────────
def sinusoidal_emb(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / max(half - 1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)

# ── 两跳 GCN 聚合层 ───────────────────────────────────────────────────────────
class TwoHopGCNProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, coords, a1, a2):
        agg = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)
        return self.proj(agg)

# ── 模型（与 ε 版本相同，只是输出语义变了：输出 x0 而非 ε）─────────────────
class NodeDiffusionGCN(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, n_max=N_MAX, use_adj=True):
        super().__init__()
        self.use_adj = use_adj
        self.n_max   = n_max
        self.n_heads = n_heads
        self.gcn_proj  = TwoHopGCNProj(d_model)
        self.pos_emb   = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.text_proj = nn.Linear(768, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True)
        self.tf  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, cls_enc, a1, a2, adj=None, node_mask=None):
        B = noisy_coords.size(0)
        x = (self.gcn_proj(noisy_coords, a1, a2)
             + self.pos_emb(torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0))
             + self.time_proj(sinusoidal_emb(t_idx, D_MODEL)).unsqueeze(1)
             + self.text_proj(cls_enc).unsqueeze(1))
        key_pad   = ((node_mask == 0).float() * -1e9) if node_mask is not None else None
        attn_mask = None
        if self.use_adj and adj is not None:
            real_row  = node_mask.unsqueeze(-1)
            attn_mask = ((1.0 - adj) * (-1e9) * real_row
                         ).unsqueeze(1).expand(B, self.n_heads, N_MAX, N_MAX
                         ).reshape(B * self.n_heads, N_MAX, N_MAX)
        x = self.tf(x, mask=attn_mask, src_key_padding_mask=key_pad)
        return self.out(x)   # 输出 pred_x0，形状 [B, N_MAX, 2]


# ── 训练 + 推理 + 可视化 ──────────────────────────────────────────────────────
def train_version(use_adj, label):
    print(f"\n{'='*60}")
    print(f"版本：{'有邻接掩码' if use_adj else '无邻接掩码'}  ({label})  [x0-prediction]")
    print(f"{'='*60}")

    model = NodeDiffusionGCN(use_adj=use_adj).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

    mask3 = node_masks.unsqueeze(-1)

    # ── 训练循环（loss 直接在坐标空间）──────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        t_idx  = torch.randint(0, T, (N,), device=DEVICE)
        noise  = torch.randn_like(coords_raw)
        x_t    = q_sample(coords_raw, t_idx, noise) * mask3

        pred_x0 = model(
            x_t, t_idx, cls_encs,
            a1=adj_norm1, a2=adj_norm2,
            adj=adj_tensor if use_adj else None,
            node_mask=node_masks,
        )
        # loss：直接预测真实坐标，只计算有效节点
        loss = ((pred_x0 - coords_raw) ** 2 * mask3).sum() / mask3.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}  loss={loss.item():.6f}")

    # ── 推理（x0-prediction 后验均值）────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x = torch.randn(N, N_MAX, 2, device=DEVICE) * mask3
        for step in reversed(range(T)):
            t_idx = torch.full((N,), step, dtype=torch.long, device=DEVICE)

            # 模型预测 x0
            pred_x0 = model(
                x, t_idx, cls_encs,
                a1=adj_norm1, a2=adj_norm2,
                adj=adj_tensor if use_adj else None,
                node_mask=node_masks,
            ) * mask3
            pred_x0 = pred_x0.clamp(-1.5, 1.5)   # 防止极端值

            if step == 0:
                x = pred_x0
            else:
                # 后验均值：μ̃_t = coef1 * x0_pred + coef2 * x_t
                c1 = posterior_coef1[step]
                c2 = posterior_coef2[step]
                mean = c1 * pred_x0 + c2 * x
                # 后验方差加噪
                var  = posterior_var[step]
                x = mean + var.sqrt() * torch.randn_like(x) * mask3

        pred_coords = x

    # ── MAE ───────────────────────────────────────────────────────────────────
    pred_np_tmp = pred_coords.cpu().numpy()
    gt_np_tmp   = coords_raw.cpu().numpy()
    mask_np_tmp = node_masks.cpu().numpy()
    total_err, total_cnt = 0.0, 0
    for i in range(N):
        n  = int(mask_np_tmp[i].sum())
        px = denorm_x(pred_np_tmp[i, :n, 0])
        py = denorm_y(pred_np_tmp[i, :n, 1])
        gx = denorm_x(gt_np_tmp[i, :n, 0])
        gy = denorm_y(gt_np_tmp[i, :n, 1])
        total_err += np.abs(px - gx).sum() + np.abs(py - gy).sum()
        total_cnt += n * 2
    mae_pixel = total_err / total_cnt
    print(f"\n整体 MAE（像素）={mae_pixel:.2f}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    pred_np = pred_coords.cpu().numpy()
    gt_np   = coords_raw.cpu().numpy()
    mask_np = node_masks.cpu().numpy()
    adj_np  = adj_tensor.cpu().numpy()

    def draw_graph(ax, xs, ys, adj, n, color, title):
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b] > 0.5:
                    ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                            color="steelblue", lw=1, alpha=0.6, zorder=1)
        ax.scatter(xs, ys, c=color, s=40, zorder=3)
        ax.set_title(title, fontsize=7)
        ax.set_xlim(VIZ_MIN, VIZ_MAX); ax.set_ylim(VIZ_MAX, VIZ_MIN)
        ax.set_aspect("equal")

    fig, axes = plt.subplots(N, 2, figsize=(8, N * 3.5))
    for i, r in enumerate(records):
        n   = int(mask_np[i].sum())
        g_x = denorm_x(gt_np[i,   :n, 0]); g_y = denorm_y(gt_np[i,   :n, 1])
        p_x = denorm_x(pred_np[i, :n, 0]); p_y = denorm_y(pred_np[i, :n, 1])
        per_mae = (np.abs(g_x - p_x).mean() + np.abs(g_y - p_y).mean()) / 2.0
        draw_graph(axes[i, 0], g_x, g_y, adj_np[i], n, "green",
                   f"[{i+1}] GT  |  {r['prompt'][:45]}")
        draw_graph(axes[i, 1], p_x, p_y, adj_np[i], n, "tomato",
                   f"Pred  MAE={per_mae:.1f}px")

    fig.suptitle(f"Node Diffusion x0-pred — {label}  整体MAE={mae_pixel:.2f}px", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / f"result_gcn_x0_{label}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"图像已保存 → {out_path}")

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    save_path = weights_dir / f"diffusion_gcn_x0_{label}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"权重已保存 → {save_path}")

    return mae_pixel


# ── 主程序 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--adj", choices=["yes", "no", "both"], default="both")
args = parser.parse_args()

results = {}
if args.adj in ("yes", "both"):
    results["with_adj_mask"] = train_version(use_adj=True,  label="with_adj_mask")
if args.adj in ("no", "both"):
    results["no_adj_mask"]   = train_version(use_adj=False, label="no_adj_mask")

print("\n" + "=" * 60)
for label, mae in results.items():
    print(f"  {label}  MAE = {mae:.2f} px")
print("=" * 60)
