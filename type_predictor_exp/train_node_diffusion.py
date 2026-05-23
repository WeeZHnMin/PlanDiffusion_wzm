"""
Node-based Coordinate Diffusion (两版本对比)
- 数据来源：node_data.jsonl（node_coords, node_mask, adj_matrix）
- 扩散目标：每个节点的 (x, y) 坐标，归一化到 [-1, 1]
- 文本条件：冻结 BERT CLS → 广播加到节点特征
- 版本A：邻接矩阵掩码作用于 Transformer 注意力
- 版本B：无邻接掩码（全注意力）
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── 超参 ──────────────────────────────────────────────────────────────────────
T        = 200
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
EPOCHS   = 6000
LR       = 1e-3

# 坐标归一化：基于数据实际 min/max，精确映射到 [-1, 1]
# 在加载数据后计算，先占位
_norm_params = {}   # 延迟填充

def norm_xy(x, y):
    nx = 2 * (x - _norm_params["xmin"]) / _norm_params["xrange"] - 1
    ny = 2 * (y - _norm_params["ymin"]) / _norm_params["yrange"] - 1
    return nx, ny

def denorm_x(v): return np.round((v + 1) / 2 * _norm_params["xrange"] + _norm_params["xmin"]).astype(int)
def denorm_y(v): return np.round((v + 1) / 2 * _norm_params["yrange"] + _norm_params["ymin"]).astype(int)

# ── DDPM 噪声表 ───────────────────────────────────────────────────────────────
betas      = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

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

# 计算归一化参数（只用真实节点坐标）
all_x = [c[0] for r in records for c in r["node_coords"][:r["n_nodes"]]]
all_y = [c[1] for r in records for c in r["node_coords"][:r["n_nodes"]]]
_norm_params["xmin"]   = min(all_x);  _norm_params["xrange"] = max(all_x) - min(all_x)
_norm_params["ymin"]   = min(all_y);  _norm_params["yrange"] = max(all_y) - min(all_y)
print(f"x: {min(all_x)}~{max(all_x)}, y: {min(all_y)}~{max(all_y)}")

# 坐标张量 [N, N_MAX, 2]，归一化到 [-1, 1]
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

# 邻接矩阵 [N, N_MAX, N_MAX]
adj_tensor = torch.tensor(
    [r["adj_matrix"] for r in records], dtype=torch.float32
).to(DEVICE)

# 可视化坐标范围（像素空间）
VIZ_MIN = min(all_x) - 5
VIZ_MAX = max(all_x) + 5

# ── BERT（冻结）──────────────────────────────────────────────────────────────
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
    cls_encs = torch.stack(cls_list)   # [N, 768]
print(f"CLS: {cls_encs.shape}")

# ── 时间步嵌入 ────────────────────────────────────────────────────────────────
def sinusoidal_emb(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / max(half - 1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]

# ── 模型 ──────────────────────────────────────────────────────────────────────
class NodeDiffusion(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, n_max=N_MAX, use_adj=True):
        super().__init__()
        self.use_adj = use_adj
        self.n_max   = n_max
        self.n_heads = n_heads

        self.coord_proj = nn.Linear(2, d_model)
        self.pos_emb    = nn.Embedding(n_max, d_model)   # 节点位置编码
        self.time_proj  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_proj = nn.Linear(768, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.tf  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, cls_enc, adj=None, node_mask=None):
        B = noisy_coords.size(0)

        t_emb   = sinusoidal_emb(t_idx, D_MODEL)           # [B, d]
        t_emb   = self.time_proj(t_emb).unsqueeze(1)       # [B, 1, d]
        txt_emb = self.text_proj(cls_enc).unsqueeze(1)     # [B, 1, d]
        pos_idx = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)  # [1, N_MAX]
        pos_emb = self.pos_emb(pos_idx)                    # [1, N_MAX, d]

        x = self.coord_proj(noisy_coords) + pos_emb + t_emb + txt_emb  # [B, N_MAX, d]

        # padding mask：用 float，与 attn_mask 类型一致（避免 deprecated 警告）
        key_pad = ((node_mask == 0).float() * -1e9) if node_mask is not None else None  # [B, N_MAX]

        # 邻接掩码：adj=0 → -inf（只对真实节点行生效，padding行置0防止NaN）
        attn_mask = None
        if self.use_adj and adj is not None:
            real_row = node_mask.unsqueeze(-1)              # [B, N_MAX, 1]，1=真实节点
            attn_mask = (1.0 - adj) * (-1e9) * real_row    # padding行全0，不会NaN
            attn_mask = attn_mask.unsqueeze(1) \
                                 .expand(B, self.n_heads, N_MAX, N_MAX) \
                                 .reshape(B * self.n_heads, N_MAX, N_MAX)

        x = self.tf(x, mask=attn_mask, src_key_padding_mask=key_pad)
        return self.out(x)   # [B, N_MAX, 2]


# ── 训练 + 推理 + 可视化 ──────────────────────────────────────────────────────
def train_version(use_adj, label):
    print(f"\n{'='*60}")
    print(f"版本：{'有邻接掩码' if use_adj else '无邻接掩码'}  ({label})")
    print(f"{'='*60}")

    model = NodeDiffusion(use_adj=use_adj).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        t_idx  = torch.randint(0, T, (N,), device=DEVICE)
        noise  = torch.randn_like(coords_raw)
        x_t    = q_sample(coords_raw, t_idx, noise)
        mask3  = node_masks.unsqueeze(-1)          # [N, N_MAX, 1]
        x_t    = x_t * mask3                       # padding 位置清零，与推理一致
        noise  = noise * mask3                     # loss 目标也只看真实节点的噪声

        pred_noise = model(
            x_t, t_idx, cls_encs,
            adj=adj_tensor if use_adj else None,
            node_mask=node_masks,
        )

        loss  = ((pred_noise - noise) ** 2 * mask3).sum() / mask3.sum()

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}  loss={loss.item():.4f}")

    # ── 推理（DDPM 逆扩散）────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x = torch.randn(N, N_MAX, 2, device=DEVICE) * node_masks.unsqueeze(-1)

        for step in reversed(range(T)):
            t_idx = torch.full((N,), step, dtype=torch.long, device=DEVICE)
            pred  = model(
                x, t_idx, cls_encs,
                adj=adj_tensor if use_adj else None,
                node_mask=node_masks,
            ) * node_masks.unsqueeze(-1)   # padding 位置置 0

            beta  = betas[step]
            alpha = alphas[step]
            ab    = alpha_bars[step]
            mean  = (x - beta / (1.0 - ab).sqrt() * pred) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x) * node_masks.unsqueeze(-1)
            else:
                x = mean

        pred_coords = x   # [N, N_MAX, 2]，归一化空间

    # ── MAE（像素空间直接计算）────────────────────────────────────────────────
    pred_np_tmp = pred_coords.cpu().numpy()
    gt_np_tmp   = coords_raw.cpu().numpy()
    mask_np_tmp = node_masks.cpu().numpy()
    total_err, total_cnt = 0.0, 0
    for i in range(N):
        n = int(mask_np_tmp[i].sum())
        px = denorm_x(pred_np_tmp[i, :n, 0]);  py = denorm_y(pred_np_tmp[i, :n, 1])
        gx = denorm_x(gt_np_tmp[i,   :n, 0]);  gy = denorm_y(gt_np_tmp[i,   :n, 1])
        total_err += np.abs(px - gx).sum() + np.abs(py - gy).sum()
        total_cnt += n * 2
    mae_pixel = total_err / total_cnt
    print(f"\n整体 MAE（像素）={mae_pixel:.2f}")

    # ── 可视化：每个样本左(GT)右(Pred)并排 ──────────────────────────────────────
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
        ax.set_xlim(VIZ_MIN, VIZ_MAX)
        ax.set_ylim(VIZ_MAX, VIZ_MIN)
        ax.set_aspect("equal")

    # 每行一个样本，每列两格(GT | Pred)
    fig, axes = plt.subplots(N, 2, figsize=(8, N * 3.5))

    for i, r in enumerate(records):
        n   = int(mask_np[i].sum())
        g_x = denorm_x(gt_np[i,   :n, 0]);  g_y = denorm_y(gt_np[i,   :n, 1])
        p_x = denorm_x(pred_np[i, :n, 0]);  p_y = denorm_y(pred_np[i, :n, 1])
        per_mae = (np.abs(g_x - p_x).mean() + np.abs(g_y - p_y).mean()) / 2.0

        draw_graph(axes[i, 0], g_x, g_y, adj_np[i], n,
                   color="green",
                   title=f"[{i+1}] GT  |  {r['prompt'][:45]}")
        draw_graph(axes[i, 1], p_x, p_y, adj_np[i], n,
                   color="tomato",
                   title=f"Pred  MAE={per_mae:.1f}px")

    fig.suptitle(f"Node Diffusion — {label}  整体MAE={mae_pixel:.2f}px", fontsize=11)
    fig.tight_layout()
    out_path = OUT_DIR / f"result_{label}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"图像已保存 → {out_path}")

    return mae_pixel


# ── 主程序 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--adj", choices=["yes", "no", "both"], default="both",
                    help="yes=有邻接掩码, no=无邻接掩码, both=两个都跑")
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
