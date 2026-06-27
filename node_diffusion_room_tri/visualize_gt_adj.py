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
                  device, seed: int = 0) -> np.ndarray:
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
        eps = model(x, tb, mask,
                    prompt_tokens=ptok, prompt_mask=pmsk,
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
    model.eval()
    print(f'  step={ckpt.get("step", "?")}')
    del ckpt

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── DDPM 采样 ─────────────────────────────────────────────────────────────
    for rec in records:
        print(f'  DDPM 1000步  idx={rec["idx"]}...', flush=True)
        rec['pred_coords'] = sample_coords(
            model, diffusion,
            rec['room_mb_np'], rec['adj_np'],
            rec['mask_np'], rec['ptok_np'], rec['pmsk_np'],
            device, seed=rec['idx'],
        )   # [40, 2]

    del model
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
    B = len(records)
    print(f'\n绘制 {B} × 5 图...')
    COL_W = [4.2, 2.6, 2.8, 2.6, 2.8]
    ROW_H = 2.7
    fig, axes = plt.subplots(
        B, 5,
        figsize=(sum(COL_W) + 0.2, B * ROW_H + 0.55),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        draw_col1_text   (axs[0], rec['text'])
        draw_col2_adj    (axs[1], rec['adj_np'], rec['n_nodes'], seed=args.seed)
        draw_col4_render (axs[2], rec['gt_coords_np'], rec['adj_np'],
                          rec['mask_np'], rec['node_types'])
        draw_col3_coords (axs[3], rec['pred_coords'], rec['adj_np'], rec['mask_np'])
        pred_types_for_render = rec.get('pred_node_types', rec['node_types'])
        draw_col4_render (axs[4], rec['pred_coords'], rec['adj_np'],
                          rec['mask_np'], pred_types_for_render)

    col_titles = ['Text Description',
                  'GT Adjacency Graph',
                  'GT Floor Plan',
                  r'$\theta_2$: Predicted Coords',
                  'Predicted Floor Plan']
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=13, fontweight='bold', pad=5)

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
