"""
注意力消融可视化脚本（独立，不重跑完整评估）

从本地已下载的 checkpoint 加载三个变体，
对 n_viz 条样本做 DDPM 采样，生成对比图。

用法：
  python -u visualize_ablation.py
  python -u visualize_ablation.py --n_viz 8 --seed 123
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 模型定义（与 eval_node_diffusion_ablation.py 保持一致）────────────────────

def timestep_embedding(timesteps, dim):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                         device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, -1e4)
    scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads; self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out      = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q  = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k  = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v  = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        out = attention(q, k, v, self.d_k, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer_Adj(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm_cross = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout); self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x); x = x + self.dropout(self.adj_attn(x2, x2, x2, adj_mask))
        x2 = self.norm_cross(x); x = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x); return x + self.dropout(self.ff(x2))


class EncoderLayer_Global(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm_cross = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout); self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x); x = x + self.dropout(self.global_attn(x2, x2, x2, None))
        x2 = self.norm_cross(x); x = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x); return x + self.dropout(self.ff(x2))


class EncoderLayer_Dual(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm_cross = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout); self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x)
        x  = x + self.dropout(self.adj_attn(x2, x2, x2, adj_mask) + self.global_attn(x2, x2, x2, None))
        x2 = self.norm_cross(x); x = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x); return x + self.dropout(self.ff(x2))


LAYER_MAP = {"adj_only": EncoderLayer_Adj, "global_only": EncoderLayer_Global, "dual_stream": EncoderLayer_Dual}
DISPLAY_NAME = {"adj_only": "AdjAttn Only", "global_only": "GlobalAttn Only", "dual_stream": "Dual-stream (Ours)"}


class NodeDiffusionTransformer(nn.Module):
    def __init__(self, layer_cls, model_channels=384, num_layers=6, num_heads=6,
                 dropout=0.1, bert_name="bert-base-uncased"):
        super().__init__()
        from transformers import BertModel
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, model_channels))
        self.input_emb  = nn.Linear(2, model_channels)
        self.bert       = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters(): p.requires_grad = False
        self.text_proj  = nn.Linear(self.bert.config.hidden_size, model_channels)
        self.layers     = nn.ModuleList([layer_cls(model_channels, num_heads, dropout) for _ in range(num_layers)])
        self.coord_head = nn.Sequential(nn.Linear(model_channels, model_channels), nn.ReLU(),
                                        nn.Linear(model_channels, model_channels // 2), nn.Linear(model_channels // 2, 2))

    def _adj_mask(self, adj, mask):
        return torch.clamp((1 - adj) + (1 - mask).unsqueeze(1), 0, 1)

    def forward(self, x, timesteps, adj_matrix, node_mask, prompt_tokens=None, prompt_mask=None, **kw):
        B, _, N = x.shape
        x  = x.permute(0, 2, 1).float()
        te = self.time_embed(timestep_embedding(timesteps, self.model_channels)).unsqueeze(1)
        h  = self.input_emb(x) + te
        am = self._adj_mask(adj_matrix.float(), node_mask.float())
        if prompt_tokens is not None:
            ba = prompt_mask if prompt_mask is not None else (prompt_tokens != 0).long()
            with torch.no_grad():
                th = self.bert(input_ids=prompt_tokens, attention_mask=ba).last_hidden_state
            tf = self.text_proj(th); tm = (1 - ba.float()).unsqueeze(1)
        else:
            tf = torch.zeros(B, 1, self.model_channels, device=h.device, dtype=h.dtype); tm = None
        for layer in self.layers:
            h = layer(h, am, tf, tm)
        return self.coord_head(h).permute(0, 2, 1)


class GaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.T = timesteps
        t  = torch.arange(timesteps + 1) / timesteps
        f  = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = f / f[0]; b = (1 - ab[1:] / ab[:-1]).clamp(max=0.999); ab = ab[1:]
        a  = 1.0 - b; ap = torch.cat([torch.tensor([1.0]), ab[:-1]])
        self.betas = b; self.alphas = a; self.alphas_bar = ab
        self.alphas_bar_prev = ap; self.post_var = (b * (1 - ap) / (1 - ab)).clamp(min=1e-20)

    def _to(self, dev):
        for attr in ["betas", "alphas", "alphas_bar", "alphas_bar_prev", "post_var"]:
            setattr(self, attr, getattr(self, attr).to(dev))
        return self


@torch.no_grad()
def ddpm_sample_batch(model, diff, cond_batch, device):
    diff._to(device)
    B = cond_batch["adj_matrix"].shape[0]
    x = torch.randn(B, 2, 40, device=device)
    for t in reversed(range(diff.T)):
        tb  = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(x, tb, **cond_batch).float()
        ab  = diff.alphas_bar[t]; ap = diff.alphas_bar_prev[t]
        a   = diff.alphas[t];     b  = diff.betas[t]
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        mu  = (ap.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        x   = mu + diff.post_var[t].sqrt() * torch.randn_like(x) if t > 0 else mu
    return x.permute(0, 2, 1).cpu().numpy()


# ── 可视化 ────────────────────────────────────────────────────────────────────

NODE_PALETTE = [
    "#4E8CC2","#7BB9E0","#B0D4F0","#1A5F8A",
    "#4DA863","#82C78A","#B5E0B8","#1A6B2A",
    "#D4A520","#E8C060","#F5DC99","#9A7010",
    "#9B5DB5","#C090D8","#E0C0F0","#6A2E8A",
    "#D04040","#E88080","#F5B5B5","#8A1A1A",
    "#808080","#A8A8A8","#C8C8C8","#585858",
    "#D07830","#E8A870","#F5CCA8","#8A4A10",
    "#30A0A0","#70C8C8","#A8E0E0","#107070",
]

def node_color(tid):
    idx = (int(tid) - 1) % len(NODE_PALETTE) if 1 <= int(tid) <= 32 else -1
    return NODE_PALETTE[idx] if idx >= 0 else "#CCCCCC"

def draw_cell(ax, xy, types, adj, n, xlim, ylim, rmse=None):
    for ii in range(n):
        for jj in range(ii + 1, n):
            if adj[ii, jj] > 0.5:
                ax.plot([xy[ii,0], xy[jj,0]], [xy[ii,1], xy[jj,1]],
                        color="#BBBBBB", lw=0.7, zorder=1, solid_capstyle="round")
    for k in range(n):
        ax.scatter(xy[k,0], xy[k,1], color=node_color(types[k]),
                   s=80, zorder=3, edgecolors="#444444", linewidths=0.6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#AAAAAA")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",   default="checkpoints/ablation_eval")
    p.add_argument("--hf_token",   default="", help="HF token，权重不存在时自动拉取")
    p.add_argument("--data_path",  default="data/processed/node_diffusion_cross_att/graph_dataset_6k.npz")
    p.add_argument("--bert",       default="bert-base-uncased")
    p.add_argument("--out_dir",    default="ablation_out")
    p.add_argument("--n_viz",      type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--timesteps",  type=int, default=1000)
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    # 直连 huggingface.co（私有仓库需要 token，镜像不支持私有仓库认证）
    os.environ.pop("HF_ENDPOINT", None)
    HF_REPOS = {
        "adj_only":    "wzmmmm/plandiff-adj-cross-6k",
        "global_only": "wzmmmm/plandiff-global-cross-6k",
        "dual_stream": "wzmmmm/plandiff-double-cross-6k",
    }

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    models = {}
    for name, layer_cls in LAYER_MAP.items():
        ckpt_path = os.path.join(args.ckpt_dir, name, "latest.pt")
        if not os.path.exists(ckpt_path):
            if not hf_token:
                print(f"[skip] {name}: 本地不存在且未提供 --hf_token")
                continue
            print(f"本地无权重，从 HF 拉取 {name} ...", flush=True)
            from huggingface_hub import hf_hub_download
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            try:
                ckpt_path = hf_hub_download(
                    repo_id=HF_REPOS[name], filename="latest.pt", token=hf_token,
                    local_dir=os.path.join(args.ckpt_dir, name), force_download=False,
                )
            except Exception as e:
                print(f"  [skip] 拉取失败: {e}")
                continue
        print(f"加载 {name} ...", flush=True)
        m = NodeDiffusionTransformer(layer_cls=layer_cls, model_channels=args.model_channels,
                                     num_layers=args.num_layers, num_heads=args.num_heads,
                                     bert_name=args.bert).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"]
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        result = m.load_state_dict(state, strict=False)
        if result.missing_keys:
            print(f"  [WARN] missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"  [WARN] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"{name}: 模型代码与 checkpoint 不匹配，请检查架构定义")
        m.eval()
        models[name] = m
        print(f"  step={ckpt.get('step','?')}  OK", flush=True)

    if not models:
        raise RuntimeError("没有加载到任何模型，请先运行 eval_node_diffusion_ablation.py 下载 checkpoint")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    data  = np.load(args.data_path, allow_pickle=True)
    total = len(data["node_coords"])
    rng   = np.random.default_rng(args.seed)
    idxs  = sorted(rng.choice(total, size=args.n_viz, replace=False).tolist())
    print(f"可视化 {args.n_viz} 条样本 (idx={idxs})", flush=True)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    gt_list, adj_list, mask_list, type_list = [], [], [], []
    for idx in idxs:
        gt_list.append(data["node_coords"][idx].astype("float32"))
        adj_list.append(data["adj_matrix"][idx].astype("float32"))
        mask_list.append(data["node_mask"][idx].astype("float32"))
        type_list.append(data["node_combo_ids"][idx].astype("int64"))

    cond_batch = {
        "adj_matrix":    torch.from_numpy(np.stack([data["adj_matrix"][i]   for i in idxs], 0).astype("float32")).to(device),
        "node_mask":     torch.from_numpy(np.stack([data["node_mask"][i]    for i in idxs], 0).astype("float32")).to(device),
        "prompt_tokens": torch.from_numpy(np.stack([data["prompt_tokens"][i] for i in idxs], 0).astype("int64")).to(device),
        "prompt_mask":   torch.from_numpy(np.stack([data["prompt_mask"][i]  for i in idxs], 0).astype("float32")).to(device),
    }

    pred_dict = {}
    rmse_dict = {}
    for name, model in models.items():
        print(f"采样 {name} ...", flush=True)
        preds = ddpm_sample_batch(model, diffusion, cond_batch, device)  # [n_viz, 40, 2]
        pred_dict[name] = preds
        rmses = []
        for i in range(args.n_viz):
            valid = mask_list[i] > 0.5
            rmses.append(float(np.sqrt(np.mean((preds[i][valid] - gt_list[i][valid]) ** 2))))
        rmse_dict[name] = rmses
        print(f"  mean RMSE = {np.mean(rmses):.2f}", flush=True)

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 16,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.linewidth": 0.8, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

    # 转置布局：列=变体(GT+3), 行=样本
    var_keys  = list(models.keys())
    col_keys  = ["gt"] + var_keys
    COL_LABELS = {"gt": "Ground Truth", **{k: DISPLAY_NAME[k] for k in var_keys}}

    n_viz   = args.n_viz          # 行数 = 样本数
    n_cols  = len(col_keys)       # 列数 = 变体数(4)
    cell_w, cell_h = 4.0, 4.0
    label_h = 0.7                 # 列标题行高
    fig_w   = cell_w * n_cols
    fig_h   = label_h + cell_h * n_viz
    margin  = 0.01

    fig = plt.figure(figsize=(fig_w, fig_h))
    col_starts = [cell_w * c / fig_w for c in range(n_cols)]
    col_width  = cell_w / fig_w
    # 内容行从顶部 label_h 以下开始
    row_starts = [1.0 - (label_h + cell_h * (r + 1)) / fig_h for r in range(n_viz)]
    row_height = cell_h / fig_h

    axes = {}
    for ri in range(n_viz):
        for ci in range(n_cols):
            ax = fig.add_axes([col_starts[ci] + margin, row_starts[ri] + margin,
                               col_width - 2*margin, row_height - 2*margin])
            axes[(ri, ci)] = ax

    # 列标题（变体名）写在顶部
    for ci, ck in enumerate(col_keys):
        fig.text(col_starts[ci] + col_width * 0.5, 1.0 - label_h * 0.5 / fig_h,
                 COL_LABELS[ck], ha="center", va="center",
                 fontsize=16, fontweight="bold", color="#111111")

    for ri in range(n_viz):
        n_node = int(mask_list[ri].sum())
        gt_xy  = gt_list[ri]
        adj_np = adj_list[ri]
        types  = type_list[ri]
        all_pts = [gt_xy[:n_node]] + [pred_dict[rk][ri][:n_node] for rk in var_keys]
        all_pts = np.concatenate(all_pts, axis=0)
        pad   = max(12.0, 0.08 * float(np.ptp(all_pts, axis=0).max()))
        xlim  = (all_pts[:,0].min() - pad, all_pts[:,0].max() + pad)
        ylim  = (all_pts[:,1].min() - pad, all_pts[:,1].max() + pad)
        for ci, ck in enumerate(col_keys):
            ax = axes[(ri, ci)]
            if ck == "gt":
                draw_cell(ax, gt_xy, types, adj_np, n_node, xlim, ylim)
            else:
                draw_cell(ax, pred_dict[ck][ri], types, adj_np, n_node, xlim, ylim)

    for ext in ("pdf", "png"):
        out_path = os.path.join(args.out_dir, f"ablation_viz.{ext}")
        fig.savefig(out_path)
        print(f"保存: {out_path}", flush=True)
    plt.close(fig)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
