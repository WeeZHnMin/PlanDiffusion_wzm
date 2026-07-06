"""
用测试集 GT 邻接矩阵 + 文本条件，跳过 θ₁ 直接运行 θ₂→render 并可视化。

只测试 node_diffusion_room（坐标扩散模型），不需要 θ₃ 类型分类器。
节点类型直接从 JSONL 的 node_types 字段读取，用于渲染着色。

输出 N 行 × 5 列图：
  Col1  输入文本描述
  Col2  GT 邻接图（spring layout）
  Col3  GT 真实平面图（GT 坐标渲染）
  Col4  θ₂ 预测坐标图
  Col5  预测平面图（预测坐标渲染）

用法（项目根目录）：
    python -m node_diffusion_room_tri.visualize_gt_adj \\
        --ckpt   checkpoints/node_diffusion_room_tri/latest.pt \\
        --data   data/jsonl/test_graph_dataset_10k.jsonl \\
        --n      5 \\
        --out    outputs/visualize_gt_adj_room/result.png
"""

import argparse
import json
import math
import os
import textwrap
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize':   7,
    'font.size':        7,
})

import numpy as np
import torch
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer, _assign_room_membership_single
from .diffusion import GaussianDiffusion
from .eval_iou import TextCondGNN, load_vocab
from text_graph_align.model import TextGraphAlign

# ── 渲染常量（来自 render.py，内联以消除跨包依赖）─────────────────────────────
ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]
ROOM_COLORS = {
    "bathroom":    "#AED6F1",
    "bedroom":     "#D7BDE2",
    "living_room": "#FAD7A0",
    "kitchen":     "#A9DFBF",
    "corridor":    "#CCD1D1",
    "dining_room": "#F9E79F",
    "other":       "#EAEDED",
}
ROOM_LABELS = {
    "bathroom":    "Bath",
    "bedroom":     "Bed",
    "living_room": "Living",
    "kitchen":     "Kitchen",
    "corridor":    "Corridor",
    "dining_room": "Dining",
    "other":       "Other",
}

def _build_sorted_neighbors(coords, adj, n):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(nbrs[i],
            key=lambda w: math.atan2(coords[w][1] - coords[i][1],
                                     coords[w][0] - coords[i][0]))
    return nbrs

def _next_half_edge(u, v, sorted_nbrs):
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    return nbrs[(nbrs.index(u) - 1) % len(nbrs)]

def _signed_area(face, coords):
    pts = [coords[i] for i in face]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0

def find_faces(coords, adj):
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited, faces = set(), []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face, cu, cv, steps = [], u, v, 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv)); face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw; steps += 1
            if len(face) >= 3:
                faces.append(face)
    if not faces:
        return []
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]

def vote_room_type(face, node_types, all_nbrs):
    from collections import Counter
    face_set = set(face)
    face_counts = Counter(t for node in face for t in node_types[node])
    if not face_counts:
        return "other"
    ext_counts = Counter(t for node in face
                         for w in all_nbrs[node] if w not in face_set
                         for t in node_types[w])
    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1) for t in face_counts}
    best = max(scores.values())
    winners = [t for t, s in scores.items() if s == best]
    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))

MAX_BERT_LEN = 224
N_NODES      = 40


# ── Spring layout ─────────────────────────────────────────────────────────────

def spring_layout(adj: np.ndarray, n: int, iters: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k   = 1.0 / math.sqrt(n) if n > 1 else 1.0
    t   = 1.0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist  = np.linalg.norm(delta, axis=2) + 1e-6
        rep   = (delta / dist[:, :, None] ** 2) * (k ** 2)
        f     = rep.sum(axis=1)
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0.5:
                    d = dist[i, j]
                    f[i] -= delta[i, j] * d / k
        pos = np.clip(pos + np.clip(f, -t, t), -2, 2)
        t  *= 0.95
    lo, hi = pos.min(0), pos.max(0)
    span   = np.maximum(hi - lo, 1e-6)
    return (pos - lo) / span * 1.8 - 0.9


# ── 绘图辅助 ──────────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#FAFAFA')
    for sp in ax.spines.values():
        sp.set_edgecolor('#DDDDDD'); sp.set_linewidth(0.5)


def draw_col1_text(ax, text: str):
    ax.axis('off')
    ax.text(0.5, 0.5, textwrap.fill(text, width=36),
            ha='center', va='center', fontsize=14,
            transform=ax.transAxes, multialignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', linewidth=0.8))


def draw_col2_adj(ax, adj_np: np.ndarray, n: int, seed: int = 0):
    _ax_style(ax)
    if n == 0:
        return
    pos = spring_layout(adj_np[:n, :n], n, seed=seed)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_np[i, j] > 0.5:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color='#AAAAAA', lw=0.9, alpha=0.7, zorder=1)
    for i in range(n):
        ax.add_patch(plt.Circle((pos[i, 0], pos[i, 1]), 0.11,
                                color='#4E8CC2', ec='#333333', lw=0.6, zorder=3))
        ax.text(pos[i, 0], pos[i, 1], str(i),
                ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', zorder=4)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')


def draw_col3_coords(ax, coords: np.ndarray, adj_np: np.ndarray, mask_np: np.ndarray):
    _ax_style(ax)
    valid = np.where(mask_np > 0.5)[0]
    if len(valid) == 0:
        return
    pts = coords[valid]
    c   = pts.mean(0)
    s   = max(np.abs(pts - c).max(), 1.0)
    d   = (pts - c) / s * 2.5

    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj_np[ni, nj] > 0.5:
                ax.plot([d[ii, 0], d[jj, 0]], [d[ii, 1], d[jj, 1]],
                        color='#AAAAAA', lw=0.8, alpha=0.7, zorder=1)
    for i in range(len(valid)):
        ax.add_patch(plt.Circle((d[i, 0], d[i, 1]), 0.16,
                                color='#4E8CC2', ec='#444444', lw=0.5, zorder=3))
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')


def draw_col4_render(ax, coords: np.ndarray, adj_np: np.ndarray,
                     mask_np: np.ndarray, node_types: List[List[str]]):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#DDDDDD'); sp.set_linewidth(0.5)

    n = int((mask_np > 0.5).sum())
    if n < 3:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Too few nodes',
                ha='center', va='center', fontsize=6, transform=ax.transAxes)
        return

    raw_coords = [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
    adj        = [[int(adj_np[i, j]) for j in range(n)] for i in range(n)]
    all_nbrs   = _build_sorted_neighbors(raw_coords, adj, n)

    try:
        faces      = find_faces(raw_coords, adj)
        face_types = [vote_room_type(f, node_types[:n], all_nbrs) for f in faces]
    except Exception:
        faces, face_types = [], []

    xs, ys = [c[0] for c in raw_coords], [c[1] for c in raw_coords]
    span   = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = span * 0.12
    mn_x, mn_y = min(xs), min(ys)
    def norm(x, y):
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for face, room_type in zip(faces, face_types):
        pts = [norm(*raw_coords[i]) for i in face]
        ax.add_patch(MplPolygon(pts, closed=True,
                                facecolor=ROOM_COLORS.get(room_type, '#EAEDED'),
                                edgecolor='#555555', linewidth=0.8,
                                alpha=0.88, zorder=1))
        try:
            rp = ShapelyPolygon(pts).representative_point()
            cx, cy = rp.x, rp.y
        except Exception:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, ROOM_LABELS.get(room_type, room_type),
                ha='center', va='center', fontsize=10,
                color='#111111', zorder=3)

    for i in range(n):
        for j in all_nbrs[i]:
            if j > i:
                x0, y0 = norm(*raw_coords[i])
                x1, y1 = norm(*raw_coords[j])
                ax.plot([x0, x1], [y0, y1], color='#888888', lw=0.7, zorder=2)

    for i in range(n):
        x, y = norm(*raw_coords[i])
        ax.plot(x, y, 'o', color='#333333', markersize=2.5, zorder=4)


# ── DDPM 采样 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_coords(model, diffusion, room_mb_np: np.ndarray, adj_np: np.ndarray,
                  mask_np: np.ndarray, ptok_np: np.ndarray, pmsk_np: np.ndarray,
                  device, seed: int = 0, no_text: bool = False) -> np.ndarray:
    """DDPM 1000步逆采样，三流条件：room_membership + adj_matrix + text。返回 [N_NODES, 2]。"""
    room_mb = torch.from_numpy(room_mb_np[None]).float().to(device)   # [1, N, MAX_ROOMS]
    adj     = torch.from_numpy(adj_np[None]).float().to(device)        # [1, N, N]
    mask    = torch.from_numpy(mask_np[None]).float().to(device)       # [1, N]
    ptok    = torch.from_numpy(ptok_np[None]).to(device)               # [1, T]
    pmsk    = torch.from_numpy(pmsk_np[None]).long().to(device)        # [1, T]

    diffusion._to(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, 2, N_NODES, device=device, generator=g)

    for t in reversed(range(diffusion.T)):
        tb  = torch.full((1,), t, device=device, dtype=torch.long)
        _ptok = None if no_text else ptok
        _pmsk = None if no_text else pmsk
        eps = model(x, tb, mask,
                    prompt_tokens=_ptok, prompt_mask=_pmsk,
                    room_membership=room_mb, adj_matrix=adj)
        ab  = diffusion.alphas_bar[t]
        ap  = diffusion.alphas_bar_prev[t]
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        a   = diffusion.alphas[t]
        b_  = diffusion.betas[t]
        mu  = (ap.sqrt() * b_ / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        if t > 0:
            x = mu + diffusion.posterior_variance[t].sqrt() * \
                torch.randn(x.shape, device=device, generator=g)
        else:
            x = mu

    return x.squeeze(0).permute(1, 0).cpu().numpy()  # [N_NODES, 2]


# ── DDIM 采样 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_coords_ddim(model, diffusion, room_mb_np, adj_np, mask_np,
                       ptok_np, pmsk_np, device, seed=0,
                       no_text=False, ddim_steps=200) -> np.ndarray:
    """DDIM 逆采样，返回 [N_NODES, 2]。"""
    room_mb = torch.from_numpy(room_mb_np[None]).float().to(device)
    adj     = torch.from_numpy(adj_np[None]).float().to(device)
    mask    = torch.from_numpy(mask_np[None]).float().to(device)
    ptok    = torch.from_numpy(ptok_np[None]).to(device)
    pmsk    = torch.from_numpy(pmsk_np[None]).long().to(device)

    diffusion._to(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, 2, N_NODES, device=device, generator=g)

    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()
    _ptok = None if no_text else ptok
    _pmsk = None if no_text else pmsk

    # 预计算文本特征
    text_feat, text_mask = model.encode_text(ptok, pmsk)
    extra = dict(node_mask=mask, room_membership=room_mb, adj_matrix=adj,
                 text_feat=text_feat, text_mask=text_mask)

    for i, t in enumerate(ts):
        tb  = torch.full((1,), t, device=device, dtype=torch.long)
        eps = model(x, tb, **extra)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)
        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0

    return x.squeeze(0).permute(1, 0).cpu().numpy()  # [N_NODES, 2]


# ── DDIM 采样（CLIP 梯度引导）────────────────────────────────────────────────

def sample_coords_ddim_clip_guided(
        model, diffusion, align_model,
        room_mb_np, adj_np, mask_np, ptok_np, pmsk_np,
        device, seed=0, no_text=False,
        ddim_steps=200, guidance_scale=1.0) -> np.ndarray:
    """DDIM + CLIP 梯度引导，返回 [N_NODES, 2]。"""
    import torch.nn.functional as F

    room_mb = torch.from_numpy(room_mb_np[None]).float().to(device)
    adj     = torch.from_numpy(adj_np[None]).float().to(device)
    mask    = torch.from_numpy(mask_np[None]).float().to(device)
    ptok    = torch.from_numpy(ptok_np[None]).to(device)
    pmsk    = torch.from_numpy(pmsk_np[None]).long().to(device)

    diffusion._to(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, 2, N_NODES, device=device, generator=g)

    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()

    with torch.no_grad():
        text_feat, text_mask = model.encode_text(ptok, pmsk)
        t_emb = align_model.text_enc(ptok, pmsk)

    extra = dict(node_mask=mask, room_membership=room_mb, adj_matrix=adj,
                 text_feat=text_feat, text_mask=text_mask)

    for i, t in enumerate(ts):
        tb = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, tb, **extra)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)

        if guidance_scale > 0:
            x0_g = x0.detach().requires_grad_(True)
            g_emb = align_model.graph_enc(x0_g.permute(0, 2, 1), adj, mask, room_mb)
            loss  = 1.0 - F.cosine_similarity(g_emb, t_emb).mean()
            loss.backward()
            x0 = (x0 - guidance_scale * x0_g.grad.detach()).clamp(-5, 5)

        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0

    return x.detach().squeeze(0).permute(1, 0).cpu().numpy()  # [N_NODES, 2]


# ── DDPM 采样（CLIP 梯度引导）────────────────────────────────────────────────

def sample_coords_clip_guided(
        model, diffusion,
        align_model,           # TextGraphAlign，已冻结
        room_mb_np, adj_np, mask_np, ptok_np, pmsk_np,
        device, seed=0, no_text=False,
        guidance_scale=1.0) -> np.ndarray:
    """
    DDPM 1000步逆采样 + 推理时 CLIP 梯度引导。
    每步在预测出 x̂₀ 后，用 text_graph_align 计算余弦距离梯度，
    叠加到采样均值，把坐标往更符合文本描述的方向推。
    返回 [N_NODES, 2]，归一化坐标。
    """
    import torch.nn.functional as F

    room_mb = torch.from_numpy(room_mb_np[None]).float().to(device)
    adj     = torch.from_numpy(adj_np[None]).float().to(device)
    mask    = torch.from_numpy(mask_np[None]).float().to(device)
    ptok    = torch.from_numpy(ptok_np[None]).to(device)
    pmsk    = torch.from_numpy(pmsk_np[None]).long().to(device)

    diffusion._to(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, 2, N_NODES, device=device, generator=g)

    # 提前编码文本（只跑一次）
    with torch.no_grad():
        _ptok = None if no_text else ptok
        _pmsk = None if no_text else pmsk
        t_emb = align_model.text_enc(ptok, pmsk)   # [1, d_embed]，L2归一化

    for t in reversed(range(diffusion.T)):
        tb = torch.full((1,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model(x, tb, mask,
                        prompt_tokens=_ptok, prompt_mask=_pmsk,
                        room_membership=room_mb, adj_matrix=adj)

        ab = diffusion.alphas_bar[t]
        ap = diffusion.alphas_bar_prev[t]

        # 预测 x̂₀，开启梯度以便对其求 CLIP 梯度
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-5, 5)

        if guidance_scale > 0:
            x0_g = x0.detach().requires_grad_(True)
            # x0_g: [1, 2, N] → permute → [1, N, 2] 喂 GraphEncoder
            g_emb = align_model.graph_enc(
                x0_g.permute(0, 2, 1), adj, mask, room_mb)  # [1, d_embed]
            clip_loss = 1.0 - F.cosine_similarity(g_emb, t_emb).mean()
            clip_loss.backward()
            grad = x0_g.grad.detach()                        # [1, 2, N]
            x0 = x0 - guidance_scale * grad
            x0 = x0.detach().clamp(-5, 5)

        a  = diffusion.alphas[t]
        b_ = diffusion.betas[t]
        mu = (ap.sqrt() * b_ / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        if t > 0:
            x = mu + diffusion.posterior_variance[t].sqrt() * \
                torch.randn(x.shape, device=device, generator=g)
        else:
            x = mu

    return x.squeeze(0).permute(1, 0).cpu().numpy()  # [N_NODES, 2]


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default='checkpoints/node_diffusion_room_tri/latest.pt')
    p.add_argument('--data',    default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--bert',    default='models/bert-base-uncased')
    p.add_argument('--n',       type=int, default=5)
    p.add_argument('--indices', type=int, nargs='+', default=None,
                   help='手动指定行号，如 --indices 0 42 100 200 500')
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--gpu',     type=int, default=None)
    p.add_argument('--out',     default='outputs/visualize_gt_adj_room/result.png')
    p.add_argument('--align_bert', default='',
                   help='text_graph_align 训练好的 BERT 权重（bert_aligned_best.pt），留空则跳过')
    p.add_argument('--align_ckpt', default='',
                   help='完整 text_graph_align 模型路径（align_best.pt），用于 CLIP 梯度引导推理')
    p.add_argument('--guidance_scale', type=float, default=1.0,
                   help='CLIP 梯度引导强度，0=关闭，越大引导越强')
    p.add_argument('--sampler',     default='ddim', choices=['ddpm', 'ddim'],
                   help='采样器：ddpm=1000步DDPM，ddim=DDIM（步数由--ddim_steps指定）')
    p.add_argument('--ddim_steps', type=int, default=200,
                   help='DDIM 步数（--sampler ddim 时生效）')
    p.add_argument('--no_text',      action='store_true',
                   help='推理时不传文本条件（text_feat 置零），纯图结构生成')
    p.add_argument('--use_type_model', action='store_true',
                   help='用 TextCondGNN 预测节点类型（否则 Col5 沿用 GT 类型）')
    p.add_argument('--type_ckpt', default='checkpoints/node_type/model_latest.pt')
    p.add_argument('--vocab',     default="node_diffusion_room_tri/type_combo_vocab_v3.json")
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    bert_tok = BertTokenizer.from_pretrained(args.bert)

    # ── 读取测试集 ────────────────────────────────────────────────────────────
    print(f'读取: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()

    total = len(all_lines)
    if args.indices is not None:
        indices = args.indices
    else:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(total, size=args.n, replace=False).tolist()
    print(f'选取索引: {indices}')

    records = []
    for idx in indices:
        d      = json.loads(all_lines[idx])
        n      = int(d['n_nodes'])
        adj_np = np.array(d['adj_matrix'], dtype=np.float32)[:N_NODES, :N_NODES]
        # 确保是方阵 N_NODES×N_NODES
        pad_a  = np.zeros((N_NODES, N_NODES), dtype=np.float32)
        pad_a[:adj_np.shape[0], :adj_np.shape[1]] = adj_np
        adj_np = pad_a

        mask_np        = np.zeros(N_NODES, dtype=np.float32)
        mask_np[:min(n, N_NODES)] = 1.0

        # node_types: list of list of str
        raw_types  = d.get('node_types', [])
        node_types = []
        for k in range(N_NODES):
            if k < len(raw_types):
                t = raw_types[k]
                node_types.append(t if isinstance(t, list) else [t])
            else:
                node_types.append(['other'])

        # room_membership from adj
        n_eff       = int(mask_np.sum())
        adj_bool    = adj_np[:n_eff, :n_eff].astype(bool)
        room_mb_eff = _assign_room_membership_single(adj_bool, n_eff)  # [n_eff, MAX_ROOMS]
        MAX_ROOMS   = room_mb_eff.shape[1]
        room_mb_np  = np.zeros((N_NODES, MAX_ROOMS), dtype=np.float32)
        room_mb_np[:n_eff] = room_mb_eff

        text = d['prompt']
        enc  = bert_tok(text, max_length=MAX_BERT_LEN,
                        padding='max_length', truncation=True)
        ptok_np = np.array(enc['input_ids'],      dtype=np.int64)
        pmsk_np = np.array(enc['attention_mask'], dtype=np.float32)

        # GT 坐标（padded 到 N_NODES）
        raw_coords   = d.get('node_coords', [])
        gt_coords_np = np.zeros((N_NODES, 2), dtype=np.float32)
        for k in range(min(n_eff, len(raw_coords))):
            gt_coords_np[k] = raw_coords[k]

        print(f'  [idx={idx}] n={n_eff}  text={text[:60]}...')
        records.append(dict(
            idx=idx, text=text, n_nodes=n_eff,
            adj_np=adj_np, mask_np=mask_np,
            node_types=node_types,
            room_mb_np=room_mb_np,
            ptok_np=ptok_np, pmsk_np=pmsk_np,
            gt_coords_np=gt_coords_np,
        ))

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f'\n加载模型: {args.ckpt}')
    model = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=False)
    print(f'  step={ckpt.get("step", "?")}')
    del ckpt

    if args.align_bert:
        bert_sd = torch.load(args.align_bert, map_location=device)
        missing, unexpected = model.bert.load_state_dict(bert_sd, strict=False)
        print(f'[align_bert] {args.align_bert}  missing={len(missing)}  unexpected={len(unexpected)}')

    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── 加载 text_graph_align 模型（可选）────────────────────────────────────
    align_model = None
    if args.align_ckpt:
        print(f'\n加载 text_graph_align: {args.align_ckpt}')
        align_model = TextGraphAlign(bert_name=args.bert).to(device)
        align_sd    = torch.load(args.align_ckpt, map_location=device)
        raw_sd      = align_sd['model']
        if any(k.startswith('module.') for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        align_model.load_state_dict(raw_sd, strict=False)
        align_model.eval()
        for p in align_model.parameters():
            p.requires_grad_(False)
        print(f'  step={align_sd.get("step", "?")}  guidance_scale={args.guidance_scale}')

    # ── 采样 ──────────────────────────────────────────────────────────────────
    use_ddim  = args.sampler == 'ddim'
    step_desc = f'DDIM {args.ddim_steps}步' if use_ddim else 'DDPM 1000步'
    for rec in records:
        print(f'  {step_desc}  idx={rec["idx"]}...', flush=True)
        if use_ddim:
            rec['pred_coords'] = sample_coords_ddim(
                model, diffusion,
                rec['room_mb_np'], rec['adj_np'],
                rec['mask_np'], rec['ptok_np'], rec['pmsk_np'],
                device, seed=rec['idx'], no_text=args.no_text,
                ddim_steps=args.ddim_steps,
            )
        else:
            rec['pred_coords'] = sample_coords(
                model, diffusion,
                rec['room_mb_np'], rec['adj_np'],
                rec['mask_np'], rec['ptok_np'], rec['pmsk_np'],
                device, seed=rec['idx'], no_text=args.no_text,
            )

    # ── 采样（CLIP 引导）──────────────────────────────────────────────────────
    if align_model is not None:
        for rec in records:
            print(f'  {step_desc}+CLIP引导  idx={rec["idx"]}...', flush=True)
            if use_ddim:
                rec['pred_coords_clip'] = sample_coords_ddim_clip_guided(
                    model, diffusion, align_model,
                    rec['room_mb_np'], rec['adj_np'],
                    rec['mask_np'], rec['ptok_np'], rec['pmsk_np'],
                    device, seed=rec['idx'], no_text=args.no_text,
                    ddim_steps=args.ddim_steps,
                    guidance_scale=args.guidance_scale,
                )
            else:
                rec['pred_coords_clip'] = sample_coords_clip_guided(
                    model, diffusion, align_model,
                    rec['room_mb_np'], rec['adj_np'],
                    rec['mask_np'], rec['ptok_np'], rec['pmsk_np'],
                    device, seed=rec['idx'], no_text=args.no_text,
                    guidance_scale=args.guidance_scale,
                )

    del model, align_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 类型模型推理（可选）──────────────────────────────────────────────────────
    if args.use_type_model:
        print(f'\n加载类型模型: {args.type_ckpt}')
        id_to_combo = load_vocab(args.vocab)
        type_model  = TextCondGNN(bert_name=args.bert).to(device)
        type_ckpt   = torch.load(args.type_ckpt, map_location=device)
        type_sd     = type_ckpt.get('model', type_ckpt)
        if any(k.startswith('module.') for k in type_sd):
            type_sd = {k[7:]: v for k, v in type_sd.items()}
        type_model.load_state_dict(type_sd, strict=False)
        type_model.eval()

        with torch.no_grad():
            for rec in records:
                pred_xy = torch.from_numpy(
                    rec['pred_coords'].T[None]).float().to(device)       # [1, 2, N_NODES]
                adj_t  = torch.from_numpy(rec['adj_np'][None]).to(device)
                mask_t = torch.from_numpy(rec['mask_np'][None]).to(device)
                ptok_t = torch.from_numpy(rec['ptok_np'][None]).to(device)
                pmsk_t = torch.from_numpy(rec['pmsk_np'][None]).to(device)
                logits = type_model(pred_xy, adj_matrix=adj_t, node_mask=mask_t,
                                    prompt_tokens=ptok_t, prompt_mask=pmsk_t)
                combo_ids = logits.argmax(dim=-1).cpu().numpy()[0]       # [N_NODES]
                n_eff = rec['n_nodes']
                rec['pred_node_types'] = [
                    id_to_combo.get(int(combo_ids[k]), ['other'])
                    for k in range(N_NODES)
                ]
                print(f'  类型推理完成  idx={rec["idx"]}  '
                      f'前{n_eff}个: {[rec["pred_node_types"][k] for k in range(n_eff)]}')

        del type_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    has_clip  = 'pred_coords_clip' in records[0]
    n_cols    = 7 if has_clip else 5
    B         = len(records)
    print(f'\n绘制 {B} × {n_cols} 图...')

    COL_W = [5.0, 3.2, 3.4, 3.2, 3.4] + ([3.2, 3.4] if has_clip else [])
    ROW_H = 3.4
    fig, axes = plt.subplots(
        B, n_cols,
        figsize=(sum(COL_W) + 0.4, B * ROW_H + 0.8),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        pred_types = rec.get('pred_node_types', rec['node_types'])
        draw_col1_text   (axs[0], rec['text'])
        draw_col2_adj    (axs[1], rec['adj_np'], rec['n_nodes'], seed=args.seed)
        draw_col4_render (axs[2], rec['gt_coords_np'], rec['adj_np'],
                          rec['mask_np'], rec['node_types'])
        draw_col3_coords (axs[3], rec['pred_coords'], rec['adj_np'], rec['mask_np'])
        draw_col4_render (axs[4], rec['pred_coords'], rec['adj_np'],
                          rec['mask_np'], pred_types)
        if has_clip:
            draw_col3_coords (axs[5], rec['pred_coords_clip'], rec['adj_np'], rec['mask_np'])
            draw_col4_render (axs[6], rec['pred_coords_clip'], rec['adj_np'],
                              rec['mask_np'], pred_types)

    col_titles = ['Text', 'GT Adj', 'GT Floor Plan',
                  r'$\theta_2$ Pred', 'Pred Floor Plan']
    if has_clip:
        col_titles += [f'CLIP(s={args.guidance_scale}) Pred',
                       'CLIP Floor Plan']
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=11, fontweight='bold', pad=5)

    for row_i, rec in enumerate(records):
        axes[row_i][0].set_ylabel(f"#{rec['idx']}", fontsize=6, labelpad=2)

    fig.get_layout_engine().set(hspace=0.03, wspace=0.03,
                                h_pad=0.02, w_pad=0.02)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    pdf_out = os.path.splitext(args.out)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {args.out}')
    print(f'已保存: {pdf_out}')


if __name__ == '__main__':
    main()
