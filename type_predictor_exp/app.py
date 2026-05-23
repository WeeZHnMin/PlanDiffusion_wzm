"""
Flask Demo: Text Prompt → GCN Diffusion 逐帧动画
- POST /generate  → SSE 流式推送每帧 base64 PNG
- 前端实时接收帧，推理完毕后循环播放动画，并单独展示最终结果
"""

import json, io, base64, math
from pathlib import Path
from flask import Flask, render_template, request, Response, stream_with_context

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from PIL import Image
from transformers import BertTokenizer, BertModel

# ── 路径 / 超参 ───────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
BERT_PATH = BASE.parent / "models" / "bert-base-uncased"
DATA_PATH = BASE / "node_data.jsonl"
WEIGHTS   = BASE / "weights"

T        = 400
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 加载数据 / 归一化 ─────────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

N_MAX  = len(records[0]["adj_matrix"])
all_x  = [c[0] for r in records for c in r["node_coords"][:r["n_nodes"]]]
all_y  = [c[1] for r in records for c in r["node_coords"][:r["n_nodes"]]]
_norm  = {
    "xmin": min(all_x), "xrange": max(all_x) - min(all_x),
    "ymin": min(all_y), "yrange": max(all_y) - min(all_y),
}
VIZ_MIN = min(all_x) - 5
VIZ_MAX = max(all_x) + 5

def denorm_x(v): return np.round((v+1)/2 * _norm["xrange"] + _norm["xmin"]).astype(int)
def denorm_y(v): return np.round((v+1)/2 * _norm["yrange"] + _norm["ymin"]).astype(int)

# 训练集 prompt → GT 查找表
_gt_lookup = {r["prompt"].strip(): r for r in records}

# ── DDPM (cosine schedule, Nichol & Dhariwal 2021) ───────────────────────────
def _cosine_alpha_bars(T, s=0.008):
    ts = torch.arange(T + 1, dtype=torch.float64)
    f  = torch.cos((ts / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)   # prevent div-by-zero at t=T

alpha_bars = _cosine_alpha_bars(T).to(DEVICE)
alphas     = alpha_bars / torch.cat([torch.ones(1, device=DEVICE), alpha_bars[:-1]])
betas      = (1.0 - alphas).clamp(max=0.999)

# ── BERT ─────────────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
bert      = BertModel.from_pretrained(str(BERT_PATH)).to(DEVICE)
for p in bert.parameters(): p.requires_grad = False
bert.eval()
print("BERT ready.")

# ── 模型定义 ──────────────────────────────────────────────────────────────────
class AdjPredictor(nn.Module):
    def __init__(self, d_in=768, d_hidden=512, n_max=N_MAX):
        super().__init__()
        self.n_max = n_max
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, n_max * n_max),
        )
    def forward(self, x):
        return self.mlp(x).view(-1, self.n_max, self.n_max)

def sinusoidal_emb(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / max(half-1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)

class TwoHopGCNProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
    def forward(self, coords, a1, a2):
        agg = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)
        return self.proj(agg)

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
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
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
        return self.out(x)

# ── 懒加载权重 ────────────────────────────────────────────────────────────────
_adj_model  = None
_diff_model = None
_diff_mode  = None

def ensure_models(use_adj):
    global _adj_model, _diff_model, _diff_mode
    adj_ckpt  = WEIGHTS / "adj_predictor.pt"
    diff_name = f"diffusion_gcn_{'with' if use_adj else 'no'}_adj_mask.pt"
    diff_ckpt = WEIGHTS / diff_name
    missing   = [p for p in [adj_ckpt, diff_ckpt] if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少权重: " + ", ".join(p.name for p in missing))
    if _adj_model is None:
        m = AdjPredictor().to(DEVICE)
        m.load_state_dict(torch.load(adj_ckpt, map_location=DEVICE, weights_only=True))
        m.eval(); _adj_model = m
    if _diff_model is None or _diff_mode != use_adj:
        m = NodeDiffusionGCN(use_adj=use_adj).to(DEVICE)
        m.load_state_dict(torch.load(diff_ckpt, map_location=DEVICE, weights_only=True))
        m.eval(); _diff_model = m; _diff_mode = use_adj

def row_normalize(adj, mask):
    a = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1)
    return a * mask.unsqueeze(-1)

def _snap_1d(vals):
    """Adaptive snap: split at gaps > 3× median gap, replace each cluster with rounded mean."""
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

# ── 渲染单帧 → base64 PNG ─────────────────────────────────────────────────────
def render_frame_b64(coords_np, adj_np, n_nodes, title, label, snap=False):
    xs = denorm_x(coords_np[:n_nodes, 0])
    ys = denorm_y(coords_np[:n_nodes, 1])
    if snap:
        xs = _snap_1d(xs)
        ys = _snap_1d(ys)
    fig, ax = plt.subplots(figsize=(5, 5))
    for a in range(n_nodes):
        for b in range(a+1, n_nodes):
            if adj_np[a, b] > 0.5:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                        color="steelblue", lw=1.5, alpha=0.75, zorder=1)
    ax.scatter(xs, ys, c=np.arange(n_nodes), cmap="tab20",
               s=60, zorder=3, edgecolors="white", linewidths=0.8)
    for i in range(n_nodes):
        ax.annotate(str(i), (xs[i], ys[i]), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xlabel(title[:55], fontsize=7)
    ax.set_xlim(VIZ_MIN, VIZ_MAX); ax.set_ylim(VIZ_MAX, VIZ_MIN)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    fig.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    examples = [r["prompt"] for r in records]
    return render_template("index.html", examples=examples)

@app.route("/generate")
def generate():
    prompt  = request.args.get("prompt", "").strip()
    use_adj = request.args.get("use_adj", "1") == "1"

    def stream():
        if not prompt:
            yield f"data: {json.dumps({'error': '请输入提示词'})}\n\n"
            return
        try:
            ensure_models(use_adj)
        except FileNotFoundError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        with torch.no_grad():
            # BERT 编码
            inp = tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True, max_length=32)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            cls_enc = bert(**inp).last_hidden_state[0, 0].unsqueeze(0)

            # 邻接矩阵 / 节点数
            gt_rec = _gt_lookup.get(prompt)
            if gt_rec is not None:
                n_nodes = gt_rec["n_nodes"]
                adj_use = torch.tensor(gt_rec["adj_matrix"],
                                       dtype=torch.float32, device=DEVICE).unsqueeze(0)
                src = "GT"
            else:
                logits  = _adj_model(cls_enc)
                adj_sym = ((logits.sigmoid() > 0.5).float() +
                           (logits.sigmoid() > 0.5).float().transpose(1,2) > 0).float()
                n_nodes = max(min(int(adj_sym[0].diagonal().sum().item()), N_MAX), 3)
                adj_use = adj_sym
                src = "AdjPredictor"

            node_mask = torch.zeros(1, N_MAX, device=DEVICE)
            node_mask[0, :n_nodes] = 1.0
            adj_use = adj_use * node_mask.unsqueeze(-1) * node_mask.unsqueeze(1)
            a1 = row_normalize(adj_use, node_mask)
            a2 = row_normalize(torch.bmm(a1, a1), node_mask)
            adj_np  = adj_use[0].cpu().numpy()
            n_edges = int((adj_np[:n_nodes, :n_nodes].sum() - n_nodes) / 2)

            # 推送元信息
            yield f"data: {json.dumps({'meta': {'n_nodes': n_nodes, 'n_edges': n_edges, 'src': src}})}\n\n"

            # 逆扩散，每 20 步推送一帧
            x = torch.randn(1, N_MAX, 2, device=DEVICE) * node_mask.unsqueeze(-1)
            frame_idx = 0
            for step in reversed(range(T)):
                t_idx = torch.tensor([step], dtype=torch.long, device=DEVICE)
                pred  = _diff_model(
                    x, t_idx, cls_enc, a1=a1, a2=a2,
                    adj=adj_use if use_adj else None, node_mask=node_mask,
                ) * node_mask.unsqueeze(-1)

                beta  = betas[step]; alpha = alphas[step]; ab = alpha_bars[step]
                mean  = (x - beta / (1.0 - ab).sqrt() * pred) / alpha.sqrt()
                x     = (mean + beta.sqrt() * torch.randn_like(x) * node_mask.unsqueeze(-1)
                         if step > 0 else mean)

                elapsed = T - step
                if elapsed % 20 == 0:
                    frame_idx += 1
                    label  = ("✅ 最终结果 (t=0)" if step == 0
                              else f"去噪 {elapsed}/{T} 步  (t={step})")
                    is_final = (step == 0)
                    b64    = render_frame_b64(x[0].cpu().numpy(), adj_np, n_nodes, prompt, label, snap=is_final)
                    yield f"data: {json.dumps({'frame': b64, 'idx': frame_idx, 'label': label, 'final': is_final})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(stream()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    print(f"N_MAX={N_MAX}, T={T}, device={DEVICE}")
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=False)
