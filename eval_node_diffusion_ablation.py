"""
NodeDiffusion 注意力结构消融评估脚本（服务器版）

对三个变体各跑 DDPM 1000 步采样，计算 Coord-RMSE：
  - adj_only    : wzmmmm/plandiff-adj-cross-6k
  - global_only : wzmmmm/plandiff-global-cross-6k
  - dual_stream : wzmmmm/plandiff-double-cross-6k （或本文完整模型）

用法：
  python eval_node_diffusion_ablation.py --hf_token YOUR_TOKEN
  python eval_node_diffusion_ablation.py --hf_token YOUR_TOKEN --n_eval 200
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  default="", help="HF token（也可用 HF_TOKEN 环境变量）")
    p.add_argument("--data_path", default="data/processed/node_diffusion_cross_att/graph_dataset_6k.npz")
    p.add_argument("--ckpt_dir",  default="checkpoints/ablation_eval")
    p.add_argument("--out",       default="ablation_attn_results.json")
    p.add_argument("--out_dir",   default="ablation_out")
    p.add_argument("--bert",      default="bert-base-uncased")
    p.add_argument("--n_eval",    type=int, default=1000)
    p.add_argument("--batch_size",type=int, default=256)
    p.add_argument("--n_viz",     type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    return p.parse_args()


HF_REPOS = {
    "adj_only":    "wzmmmm/plandiff-adj-cross-6k",
    "global_only": "wzmmmm/plandiff-global-cross-6k",
    "dual_stream": "wzmmmm/plandiff-double-cross-6k",
}

DISPLAY_NAME = {
    "adj_only":    "AdjAttn only",
    "global_only": "GlobalAttn only",
    "dual_stream": "Dual-stream (AdjAttn + GlobalAttn)",
}


# ── 模型定义 ──────────────────────────────────────────────────────────────────

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
        self.d_k = d_model // heads
        self.h   = heads
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
        self.norm1      = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.adj_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff         = FeedForward(d_model, dropout)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x)
        x  = x + self.dropout(self.adj_attn(x2, x2, x2, adj_mask))
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class EncoderLayer_Global(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x)
        x  = x + self.dropout(self.global_attn(x2, x2, x2, None))
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class EncoderLayer_Dual(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, adj_mask, tf, tm):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn(x2, x2, x2, adj_mask) +
            self.global_attn(x2, x2, x2, None)
        )
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, tf, tf, tm))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


LAYER_MAP = {
    "adj_only":    EncoderLayer_Adj,
    "global_only": EncoderLayer_Global,
    "dual_stream": EncoderLayer_Dual,
}


class NodeDiffusionTransformer(nn.Module):
    def __init__(self, layer_cls, model_channels=384, num_layers=6,
                 num_heads=6, dropout=0.1, bert_name="bert-base-uncased"):
        super().__init__()
        from transformers import BertModel
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels), nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.bert.config.hidden_size, model_channels)
        self.layers = nn.ModuleList([
            layer_cls(model_channels, num_heads, dropout) for _ in range(num_layers)
        ])
        self.coord_head = nn.Sequential(
            nn.Linear(model_channels, model_channels), nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )

    def _adj_mask(self, adj, mask):
        return torch.clamp((1 - adj) + (1 - mask).unsqueeze(1), 0, 1)

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kw):
        B, _, N = x.shape
        x  = x.permute(0, 2, 1).float()
        te = self.time_embed(timestep_embedding(timesteps, self.model_channels)).unsqueeze(1)
        h  = self.input_emb(x) + te
        am = self._adj_mask(adj_matrix.float(), node_mask.float())
        if prompt_tokens is not None:
            ba = prompt_mask if prompt_mask is not None else (prompt_tokens != 0).long()
            with torch.no_grad():
                th = self.bert(input_ids=prompt_tokens, attention_mask=ba).last_hidden_state
            tf = self.text_proj(th)
            tm = (1 - ba.float()).unsqueeze(1)
        else:
            tf = torch.zeros(B, 1, self.model_channels, device=h.device, dtype=h.dtype)
            tm = None
        for layer in self.layers:
            h = layer(h, am, tf, tm)
        return self.coord_head(h).permute(0, 2, 1)


# ── DDPM 采样 ──────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.T = timesteps
        t  = torch.arange(timesteps + 1) / timesteps
        f  = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = f / f[0]
        b  = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        ab = ab[1:]
        a  = 1.0 - b
        ap = torch.cat([torch.tensor([1.0]), ab[:-1]])
        self.betas = b; self.alphas = a
        self.alphas_bar = ab; self.alphas_bar_prev = ap
        self.post_var = (b * (1 - ap) / (1 - ab)).clamp(min=1e-20)

    def _to(self, dev):
        for attr in ["betas", "alphas", "alphas_bar", "alphas_bar_prev", "post_var"]:
            setattr(self, attr, getattr(self, attr).to(dev))
        return self


@torch.no_grad()
def ddpm_sample_batch(model, diff, cond_batch, device):
    """批量 DDPM 采样，cond_batch 中每个 tensor 已在 device 上，shape [B, ...]"""
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
    return x.permute(0, 2, 1).cpu().numpy()  # [B, 40, 2]


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_path):
        print(f"本地数据集不存在，从 HF 拉取 ...")
        from huggingface_hub import hf_hub_download
        _dir = os.path.dirname(args.data_path) or "."
        os.makedirs(_dir, exist_ok=True)
        local = hf_hub_download(
            repo_id="wzmmmm/node_diffusion_150k",
            filename=os.path.basename(args.data_path),
            repo_type="dataset",
            token=hf_token,
            local_dir=_dir,
        )
        args.data_path = local
        print(f"数据集已下载: {args.data_path}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    models = {}
    for name, repo_id in HF_REPOS.items():
        local_path = os.path.join(args.ckpt_dir, name, "latest.pt")
        if os.path.exists(local_path):
            print(f"加载 {name} <- 本地 {local_path} ...")
            path = local_path
        else:
            print(f"加载 {name} <- HF {repo_id} ...")
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id=repo_id, filename="latest.pt", token=hf_token,
                    local_dir=os.path.join(args.ckpt_dir, name),
                )
            except Exception as e:
                print(f"  [skip] {name}: {e}")
                continue
        m = NodeDiffusionTransformer(
            layer_cls=LAYER_MAP[name],
            model_channels=args.model_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            bert_name=args.bert,
        ).to(device)
        ckpt  = torch.load(path, map_location=device)
        state = ckpt["model"]
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        result = m.load_state_dict(state, strict=False)
        if result.missing_keys:
            print(f"  [WARN] missing keys ({len(result.missing_keys)}): {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"  [WARN] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:3]}")
        m.eval()
        models[name] = m
        print(f"  step={ckpt.get('step', '?')}  OK")

    if not models:
        raise RuntimeError("没有加载到任何模型")

    # ── 评估 ──────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    data  = np.load(args.data_path, allow_pickle=True)
    total = len(data["node_coords"])
    rng   = np.random.default_rng(args.seed)
    idxs  = sorted(rng.choice(total, size=min(args.n_eval, total), replace=False).tolist())
    print(f"\n数据集: {total}  评估: {len(idxs)} 条")

    rmse_list  = {n: [] for n in models}
    nrmse_list = {n: [] for n in models}   # neighbor relative RMSE
    pred_cache = {n: [] for n in models}
    gt_cache, mask_cache, adj_cache, type_cache = [], [], [], []

    # 预先加载所有样本到 CPU
    gt_all   = data["node_coords"][idxs].astype("float32")
    adj_all  = data["adj_matrix"][idxs].astype("float32")
    mask_all = data["node_mask"][idxs].astype("float32")
    typ_all  = data["node_combo_ids"][idxs].astype("int64")
    ptok_all = data["prompt_tokens"][idxs].astype("int64")
    pmsk_all = data["prompt_mask"][idxs].astype("float32")

    for i in range(len(idxs)):
        gt_cache.append(gt_all[i])
        adj_cache.append(adj_all[i])
        mask_cache.append(mask_all[i] > 0.5)
        type_cache.append(typ_all[i])

    BS = args.batch_size
    n_batches = (len(idxs) + BS - 1) // BS

    for name, model in models.items():
        print(f"\n评估 {name} ...", flush=True)
        all_preds = []
        for bi in range(n_batches):
            s, e = bi * BS, min((bi + 1) * BS, len(idxs))
            cond_batch = {
                "adj_matrix":    torch.from_numpy(adj_all[s:e]).to(device),
                "node_mask":     torch.from_numpy(mask_all[s:e]).to(device),
                "prompt_tokens": torch.from_numpy(ptok_all[s:e]).to(device),
                "prompt_mask":   torch.from_numpy(pmsk_all[s:e]).to(device),
            }
            preds = ddpm_sample_batch(model, diffusion, cond_batch, device)  # [B, 40, 2]
            all_preds.append(preds)
            if (bi + 1) % 5 == 0 or bi == n_batches - 1:
                done = min(e, len(idxs))
                print(f"  [{done}/{len(idxs)}]", flush=True)

        all_preds = np.concatenate(all_preds, axis=0)  # [N, 40, 2]
        pred_cache[name] = list(all_preds)
        for i in range(len(idxs)):
            valid = mask_all[i] > 0.5
            rmse  = float(np.sqrt(np.mean((all_preds[i][valid] - gt_all[i][valid]) ** 2)))
            rmse_list[name].append(rmse)
            # ── Neighbor-RMSE：遍历所有1跳邻居对 (i,j)，计算相对坐标误差 ──
            adj_i = adj_all[i]   # [40, 40]
            errs  = []
            for u in range(40):
                if not valid[u]:
                    continue
                for v in range(40):
                    if not valid[v] or adj_i[u, v] < 0.5:
                        continue
                    gt_rel   = gt_all[i][v]   - gt_all[i][u]
                    pred_rel = all_preds[i][v] - all_preds[i][u]
                    errs.append(np.sum((pred_rel - gt_rel) ** 2))
            nrmse_list[name].append(float(np.sqrt(np.mean(errs))) if errs else 0.0)
        print(f"  mean RMSE = {np.mean(rmse_list[name]):.4f}  "
              f"Neighbor-RMSE = {np.mean(nrmse_list[name]):.4f}", flush=True)

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"{'变体':<35} {'RMSE':>8} {'Neighbor-RMSE':>15}")
    print(f"{'─'*75}")
    results = {}
    for name in models:
        r  = np.array(rmse_list[name])
        nr = np.array(nrmse_list[name])
        results[name] = {
            "display": DISPLAY_NAME[name],
            "mean_rmse":         round(float(r.mean()),  4),
            "std_rmse":          round(float(r.std()),   4),
            "median_rmse":       round(float(np.median(r)), 4),
            "mean_neighbor_rmse": round(float(nr.mean()), 4),
            "std_neighbor_rmse":  round(float(nr.std()),  4),
            "n": len(r),
        }
        print(f"  {DISPLAY_NAME[name]:<33} {r.mean():>8.2f} {nr.mean():>15.2f}")
    print(f"{'─'*75}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至 {args.out}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
                           s=28, zorder=3, edgecolors="#444444", linewidths=0.4)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("#AAAAAA")
            if rmse is not None:
                ax.text(0.5, -0.06, f"RMSE = {rmse:.1f} px",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6, color="#333333")

        plt.rcParams.update({
            "font.family": "DejaVu Serif", "font.size": 7,
            "axes.titlesize": 7, "axes.titleweight": "bold",
            "axes.linewidth": 0.6, "figure.dpi": 300,
            "savefig.dpi": 300, "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        })

        n_viz = min(args.n_viz, len(idxs))
        rng_viz  = np.random.default_rng(args.seed + 99)
        viz_idxs = sorted(rng_viz.choice(len(idxs), size=n_viz, replace=False).tolist())

        var_keys = list(models.keys())
        ROW_LABELS = {"gt": "Ground Truth", **{k: DISPLAY_NAME[k] for k in var_keys}}
        row_keys = ["gt"] + var_keys

        cell_w, cell_h = 2.4, 2.4
        label_w = 1.0
        fig_w = label_w + cell_w * n_viz
        fig_h = cell_h * len(row_keys)
        margin = 0.012

        fig = plt.figure(figsize=(fig_w, fig_h))
        col_starts = [(label_w + cell_w * c) / fig_w for c in range(n_viz)]
        col_width  = cell_w / fig_w
        row_starts = [1.0 - cell_h * (r + 1) / fig_h for r in range(len(row_keys))]
        row_height = cell_h / fig_h

        axes = {}
        for ri in range(len(row_keys)):
            for ci in range(n_viz):
                ax = fig.add_axes([
                    col_starts[ci] + margin, row_starts[ri] + margin,
                    col_width - 2*margin, row_height - 2*margin,
                ])
                axes[(ri, ci)] = ax

        for ci, si in enumerate(viz_idxs):
            n_node = int(mask_cache[si].sum())
            gt_xy  = gt_cache[si]
            adj_np = adj_cache[si]
            types  = type_cache[si]
            pts    = gt_xy[:n_node]
            pad    = max(12.0, 0.08 * float(np.ptp(pts, axis=0).max()))
            xlim   = (pts[:,0].min() - pad, pts[:,0].max() + pad)
            ylim   = (pts[:,1].min() - pad, pts[:,1].max() + pad)
            axes[(0, ci)].set_title(f"Sample {ci+1}", pad=3, fontsize=7,
                                    fontweight="bold", color="#222222")
            for ri, rk in enumerate(row_keys):
                ax = axes[(ri, ci)]
                if rk == "gt":
                    draw_cell(ax, gt_xy, types, adj_np, n_node, xlim, ylim)
                else:
                    draw_cell(ax, pred_cache[rk][si], types, adj_np, n_node,
                              xlim, ylim, rmse=rmse_list[rk][si])

        for ri, rk in enumerate(row_keys):
            fig.text((label_w * 0.5) / fig_w, row_starts[ri] + row_height * 0.5,
                     ROW_LABELS.get(rk, rk), ha="center", va="center",
                     fontsize=7, fontweight="bold", color="#111111", rotation=90)

        out_pdf = os.path.join(args.out_dir, "ablation_viz.pdf")
        out_png = os.path.join(args.out_dir, "ablation_viz.png")
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"可视化保存至 {out_pdf}")
        print(f"可视化保存至 {out_png}")
    except Exception as e:
        print(f"[WARN] 可视化失败: {e}")


if __name__ == "__main__":
    main()
