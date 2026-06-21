"""
端到端推理可视化：从测试集随机取 N 条文本，从头走完 θ₁→θ₂→θ₃→render 全流程。
θ₂/θ₃ 批量并行推理（batch_size = N）。

输出 N 行 × 5 列图：
  Col1  输入文本描述
  Col2  θ₁ 邻接图（spring layout，节点无坐标）
  Col3  θ₂ 顶点坐标图（节点按类型着色）
  Col4  θ₃ 类型预测（节点按预测类型着色）
  Col5  渲染平面图

用法（项目根目录）：
    python -m node_diffusion_cross_att.visualize_e2e \\
        --ckpt1  checkpoints/llm_graph/stage2/20260614_155601/latest.pt \\
        --ckpt2  checkpoints/node_diffusion_cross_att/latest.pt \\
        --ckpt3  checkpoints/node_type/20260616_223156/model_latest.pt \\
        --n      5 --seed 42 \\
        --out    outputs/visualize_e2e/result.png
"""

import argparse
import math
import os
import textwrap
from pathlib import Path
from typing import Dict, List

from shapely.geometry import Polygon as ShapelyPolygon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon as MplPolygon

# ── 全局字体设置（Roman / serif） ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize':   7,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'font.size':        7,
})

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import BertTokenizer

from llm_graph.infer_stage1 import (
    load_model as load_llm,
    load_dataset,
    get_prefix_and_gt,
    parse_sequence,
)
from llm_graph.infer_batch import generate_batch, decode_bpe_text, MAX_BERT_LEN
from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .type_model import NodeTypeClassifier
from .render import (
    load_vocab,
    find_faces,
    vote_room_type,
    _build_sorted_neighbors,
    ROOM_COLORS,
    ROOM_LABELS,
)

# ── 颜色表 ────────────────────────────────────────────────────────────────────

COMBO_COLORS = [
    '#A9CDE8', '#7BB9E0', '#4D9FD5', '#1A6FA6',
    '#A8D5A2', '#6EBC68', '#3A9E35', '#1E7B19',
    '#F5D48B', '#F0BE45', '#E89E10', '#B87A00',
    '#D4A8D4', '#B87BB8', '#8F4F8F', '#6B2D6B',
    '#F5A8A8', '#E86060', '#D02020', '#A00000',
    '#BBBBBB', '#999999', '#777777', '#555555',
    '#FFDDB0', '#FFB86C', '#E88C30', '#C06010',
    '#B0E0E0', '#70C0C0', '#30A0A0', '#008080',
]


def type_color(type_id: int) -> str:
    if type_id <= 0 or type_id > 32:
        return '#DDDDDD'
    return COMBO_COLORS[(type_id - 1) % len(COMBO_COLORS)]


# ── spring layout ─────────────────────────────────────────────────────────────

def spring_layout(adj: np.ndarray, n: int, iters: int = 200,
                  seed: int = 0) -> np.ndarray:
    """Fruchterman-Reingold，返回 [n, 2] 坐标（范围约 [-0.9, 0.9]）。"""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k   = 1.0 / math.sqrt(n) if n > 1 else 1.0
    t   = 1.0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]       # [n, n, 2]
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


# ── θ₂ 批量采样 ───────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_coords_batch(model2, diffusion,
                        adj_batch: np.ndarray, mask_batch: np.ndarray,
                        ptok_batch: np.ndarray, pmsk_batch: np.ndarray,
                        device,
                        sample_indices: List[int] = None) -> np.ndarray:
    """
    DDPM 1000步批量逆采样。
    输入均为 numpy，shape [B, ...]；返回 pred_coords [B, 40, 2]。
    """
    B    = adj_batch.shape[0]
    adj  = torch.from_numpy(adj_batch ).to(device)          # [B, 40, 40]
    mask = torch.from_numpy(mask_batch).to(device)          # [B, 40]
    ptok = torch.from_numpy(ptok_batch).to(device)          # [B, T]
    pmsk = torch.from_numpy(pmsk_batch).long().to(device)   # [B, T]

    # 预计算 BERT 特征（整个 DDPM 过程共享）
    text_hidden = model2.bert(input_ids=ptok, attention_mask=pmsk).last_hidden_state
    text_feat   = model2.text_proj(text_hidden)              # [B, T, d]
    text_mask   = (1 - pmsk.float()).unsqueeze(1)            # [B, 1, T]
    am          = model2._build_adj_mask(adj.float(), mask.float())  # [B, 40, 40]

    from .model import timestep_embedding

    def fwd(x_t: torch.Tensor, t_val: int) -> torch.Tensor:
        tb    = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_in  = x_t.permute(0, 2, 1).float()                # [B, 40, 2]
        t_emb = model2.time_embed(
            timestep_embedding(tb, model2.model_channels)
        ).unsqueeze(1)                                       # [B, 1, d]
        h = model2.input_emb(x_in) + t_emb                  # [B, 40, d]
        for layer in model2.layers:
            h = layer(h, am, text_feat, text_mask)
        return model2.coord_head(h).permute(0, 2, 1).float() # [B, 2, 40]

    diffusion._to(device)
    # 每条样本用自身索引独立 seed，与全局 RNG 状态解耦，
    # 保证同一样本无论与哪些样本同批次结果都一致。
    if sample_indices is not None:
        x_slices = []
        for idx in sample_indices:
            g = torch.Generator(device=device)
            g.manual_seed(int(idx) + 123456)
            x_slices.append(torch.randn(1, 2, 40, device=device, generator=g))
        x = torch.cat(x_slices, dim=0)
    else:
        x = torch.randn(B, 2, 40, device=device)
    for t in reversed(range(diffusion.T)):
        eps = fwd(x, t)
        ab  = diffusion.alphas_bar[t]
        ap  = diffusion.alphas_bar_prev[t]
        a   = diffusion.alphas[t]
        b_  = diffusion.betas[t]
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        mu  = (ap.sqrt() * b_ / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        x   = mu + diffusion.posterior_variance[t].sqrt() * torch.randn_like(x) if t > 0 else mu

    return x.permute(0, 2, 1).cpu().numpy()   # [B, 40, 2]


# ── 各列绘图函数 ──────────────────────────────────────────────────────────────

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
        ax.add_patch(plt.Circle((pos[i, 0], pos[i, 1]), 0.07,
                                color='#4E8CC2', ec='#333333', lw=0.6, zorder=3))
        ax.text(pos[i, 0], pos[i, 1], str(i),
                ha='center', va='center', fontsize=7,
                color='white', fontweight='bold', zorder=4)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')


def _draw_graph(ax, coords: np.ndarray, adj_np: np.ndarray,
                mask_np: np.ndarray, type_ids: np.ndarray,
                r: float = 0.12):
    """通用图绘制：节点按类型着色，节点半径 r。"""
    _ax_style(ax)
    valid = np.where(mask_np > 0.5)[0]
    if len(valid) == 0:
        return
    pts = coords[valid]
    c   = pts.mean(0)
    s   = max(np.abs(pts - c).max(), 1.0)
    d   = (pts - c) / s * 2.5                   # 归一化到 [-2.5, 2.5]

    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj_np[ni, nj] > 0.5:
                ax.plot([d[ii, 0], d[jj, 0]], [d[ii, 1], d[jj, 1]],
                        color='#AAAAAA', lw=0.8, alpha=0.7, zorder=1)
    for i, vi in enumerate(valid):
        ax.add_patch(plt.Circle((d[i, 0], d[i, 1]), r,
                                color=type_color(int(type_ids[vi])),
                                ec='#444444', lw=0.5, zorder=3))
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')


def draw_col3_coords(ax, coords, adj_np, mask_np, type_ids):
    _draw_graph(ax, coords, adj_np, mask_np, type_ids, r=0.16)


def draw_col4_types(ax, coords, adj_np, mask_np, type_ids):
    _draw_graph(ax, coords, adj_np, mask_np, type_ids, r=0.20)


def draw_col5_render(ax, coords: np.ndarray, adj_np: np.ndarray,
                     mask_np: np.ndarray, type_ids: np.ndarray,
                     id_to_combo: Dict[int, List[str]]):
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
    combo_ids  = [int(type_ids[i]) for i in range(n)]
    adj        = [[int(adj_np[i, j]) for j in range(n)] for i in range(n)]
    node_types = [id_to_combo.get(cid, ['other']) for cid in combo_ids]
    all_nbrs   = _build_sorted_neighbors(raw_coords, adj, n)

    try:
        faces      = find_faces(raw_coords, adj)
        face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]
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


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt1', default='checkpoints/llm_graph/stage2/20260614_155601/latest.pt')
    p.add_argument('--ckpt2', default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--ckpt3', default='checkpoints/node_type/20260616_223156/model_latest.pt')
    p.add_argument('--data',  default='data/processed/graph_tree/text_graph_tree_test_10k.npz')
    p.add_argument('--vocab', default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='data/processed/type_combo_vocab.json')
    p.add_argument('--n',       type=int,   default=5)
    p.add_argument('--indices', type=int,   nargs='+', default=None,
                   help='手动指定测试集索引，例如 --indices 0 42 100 200 500；指定后忽略 --n 和 --seed')
    p.add_argument('--seed',    type=int,   default=42)
    p.add_argument('--out',   default='outputs/visualize_e2e/result.png')
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    id_to_combo = load_vocab(Path(args.combo_vocab))

    # 分词器（CPU，整个过程保留）
    bpe_tok  = Tokenizer.from_file(args.vocab)
    bert_tok = BertTokenizer.from_pretrained(args.bert)

    # ── 取样本索引 ────────────────────────────────────────────────────────────
    all_tokens, all_lengths, all_textlens = load_dataset(args.data)
    if args.indices is not None:
        indices = args.indices
        print(f'手动指定索引: {indices}')
    else:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(len(all_tokens), size=args.n, replace=False).tolist()
        print(f'随机索引: {indices}')

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 1  θ₁：加载 → 批量推理 → 卸载
    # ════════════════════════════════════════════════════════════════════════
    print('\n[θ₁] 加载模型...')
    model1 = load_llm(args.ckpt1, device)

    prefixes = [get_prefix_and_gt(i, all_tokens, all_lengths, all_textlens)[0]
                for i in indices]
    print(f'[θ₁] 批量自回归生成（batch={len(prefixes)}）...')
    gen_seqs = generate_batch(model1, prefixes, device, max_new_tokens=200)

    # 卸载 θ₁
    del model1
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₁] 模型已卸载')

    # 解析 + BERT 重编码
    records = []
    for idx, prefix, gen_seq in zip(indices, prefixes, gen_seqs):
        parsed = parse_sequence(gen_seq)
        print(f'  idx={idx}  n_nodes={parsed["n_nodes"]}  valid={parsed["valid"]}')
        if not parsed['valid']:
            continue

        N      = parsed['n_nodes']
        adj_np = np.zeros((40, 40), dtype=np.float32)
        adj_np[:N, :N] = np.array(parsed['adj'], dtype=np.float32)
        mask_np        = np.zeros(40, dtype=np.float32)
        mask_np[:N]    = 1.0

        text    = decode_bpe_text(prefix[:-1], bpe_tok)
        enc     = bert_tok(text, max_length=MAX_BERT_LEN,
                           padding='max_length', truncation=True)
        ptok_np = np.array(enc['input_ids'],      dtype=np.int64)
        pmsk_np = np.array(enc['attention_mask'], dtype=np.float32)
        print(f'    文本: {text[:70]}...' if len(text) > 70 else f'    文本: {text}')

        records.append(dict(idx=idx, text=text, n_nodes=N,
                            adj_np=adj_np, mask_np=mask_np,
                            ptok_np=ptok_np, pmsk_np=pmsk_np))

    if not records:
        print('所有样本均无效，退出。')
        return

    B          = len(records)
    adj_batch  = np.stack([r['adj_np']  for r in records])   # [B, 40, 40]
    mask_batch = np.stack([r['mask_np'] for r in records])   # [B, 40]
    ptok_batch = np.stack([r['ptok_np'] for r in records])   # [B, T]
    pmsk_batch = np.stack([r['pmsk_np'] for r in records])   # [B, T]

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 2  θ₂：加载 → 批量 DDPM 采样 → 卸载
    # ════════════════════════════════════════════════════════════════════════
    print(f'\n[θ₂] 加载模型...')
    model2 = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2  = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    diffusion = GaussianDiffusion(timesteps=1000)
    print(f'  step={ckpt2.get("step","?")}')
    del ckpt2

    print(f'[θ₂] 批量 DDPM 1000步（batch={B}）...')
    with torch.no_grad():
        pred_coords_batch = sample_coords_batch(
            model2, diffusion,
            adj_batch, mask_batch, ptok_batch, pmsk_batch, device,
            sample_indices=[r['idx'] for r in records])   # [B, 40, 2]

    del model2, diffusion
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₂] 模型已卸载')

    # ════════════════════════════════════════════════════════════════════════
    # 阶段 3  θ₃：加载 → 批量类型预测 → 卸载
    # ════════════════════════════════════════════════════════════════════════
    print(f'\n[θ₃] 加载模型...')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step","?")}')
    del ckpt3

    print(f'[θ₃] 批量类型预测（batch={B}）...')
    with torch.no_grad():
        x_in  = torch.from_numpy(pred_coords_batch.transpose(0, 2, 1)).to(device)
        adj_t = torch.from_numpy(adj_batch ).to(device)
        msk_t = torch.from_numpy(mask_batch).to(device)
        ptk_t = torch.from_numpy(ptok_batch).to(device)
        pmk_t = torch.from_numpy(pmsk_batch).long().to(device)
        logits       = model3(x_in, adj_matrix=adj_t, node_mask=msk_t,
                              prompt_tokens=ptk_t, prompt_mask=pmk_t)
        type_ids_all = logits.argmax(dim=-1).cpu().numpy()   # [B, 40]

    del model3, x_in, adj_t, msk_t, ptk_t, pmk_t, logits
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₃] 模型已卸载')

    # 写回 records
    for k, rec in enumerate(records):
        rec['pred_coords'] = pred_coords_batch[k]
        rec['type_ids']    = type_ids_all[k]

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    print(f'\n绘制 {B} × 5 图...')
    COL_W = [4.2, 2.6, 2.6, 2.6, 2.8]
    ROW_H = 2.7   # ≈ graph column width，让 equal-aspect 图刚好填满格子
    fig, axes = plt.subplots(
        B, 5,
        figsize=(sum(COL_W) + 0.2, B * ROW_H + 0.55),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    # 先画内容
    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        draw_col1_text  (axs[0], rec['text'])
        draw_col2_adj   (axs[1], rec['adj_np'],      rec['n_nodes'], seed=args.seed)
        draw_col3_coords(axs[2], rec['pred_coords'], rec['adj_np'], rec['mask_np'], rec['type_ids'])
        draw_col4_types (axs[3], rec['pred_coords'], rec['adj_np'], rec['mask_np'], rec['type_ids'])
        draw_col5_render(axs[4], rec['pred_coords'], rec['adj_np'], rec['mask_np'],
                         rec['type_ids'], id_to_combo)

    # 列标题在 draw 之后设（避免被 draw 内的 set_title 覆盖）
    col_titles = ['Text Description',
                  r'$\theta_1$: Adjacency Graph',
                  r'$\theta_2$: Coordinate Graph',
                  r'$\theta_3$: Type Prediction',
                  'Rendered Floor Plan']
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=10, fontweight='bold', pad=4)

    # 行标签
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
