"""
逆扩散过程 + 类型预测 + 渲染 全流程可视化（真实模型推理版）：
  6 帧 DDPM 逆采样（每 200 步）→ θ₃ 类型预测 → 渲染平面图，共 8 个面板。

用法（从项目根目录运行）：
  python paper_work/figures/draw_diffusion_process_v2.py \
      --ckpt  checkpoints/node_diffusion_cross_att/latest.pt \
      --ckpt3 checkpoints/node_type/20260616_223156/model_latest.pt \
      --bert  models/bert-base-uncased \
      --data  data/processed/node_diffusion_cross_att/graph_dataset_6k.npz \
      --idx   20
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import torch

sys.path.insert(0, '.')
from node_diffusion_cross_att.model import NodeDiffusionTransformer
from node_diffusion_cross_att.type_model import NodeTypeClassifier
from node_diffusion_cross_att.render import (
    load_vocab, find_faces, vote_room_type,
    _build_sorted_neighbors, ROOM_COLORS, ROOM_LABELS,
)


# ── DDPM schedule（cosine，与训练一致）────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, T=1000):
        self.T = T
        t  = torch.arange(T + 1) / T
        f  = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = f / f[0]
        b  = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        ab = ab[1:]
        a  = 1.0 - b
        ap = torch.cat([torch.tensor([1.0]), ab[:-1]])
        self.betas           = b
        self.alphas          = a
        self.alphas_bar      = ab
        self.alphas_bar_prev = ap
        self.post_var        = (b * (1 - ap) / (1 - ab)).clamp(min=1e-20)

    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_bar', 'alphas_bar_prev', 'post_var']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


NODE_COLOR = '#4E8CC2'

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

def type_color(type_id):
    if type_id <= 0 or type_id > 32:
        return '#DDDDDD'
    return COMBO_COLORS[(type_id - 1) % len(COMBO_COLORS)]


# ── 逆扩散采样，保存中间帧 ────────────────────────────────────────────────────

def _fwd_precomputed(model, x, tb, adj, mask, text_feat, text_mask):
    from node_diffusion_cross_att.model import timestep_embedding
    x_in  = x.permute(0, 2, 1).float()
    t_emb = model.time_embed(timestep_embedding(tb, model.model_channels)).unsqueeze(1)
    h     = model.input_emb(x_in) + t_emb
    am    = model._build_adj_mask(adj.float(), mask.float())
    for layer in model.layers:
        h = layer(h, am, text_feat, text_mask)
    return model.coord_head(h).permute(0, 2, 1).float()


@torch.no_grad()
def ddpm_sample_with_snapshots(model, diff, cond, device, save_at, seed=42):
    diff.to(device)
    adj  = cond['adj_matrix'].to(device)
    mask = cond['node_mask'].to(device)
    ptok = cond['prompt_tokens'].to(device)
    pmsk = cond['prompt_mask'].to(device).long()

    text_hidden = model.bert(input_ids=ptok, attention_mask=pmsk).last_hidden_state
    text_feat   = model.text_proj(text_hidden)
    text_mask   = (1 - pmsk.float()).unsqueeze(1)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(1, 2, 40, device=device, generator=g)
    snapshots = {}

    for t in reversed(range(diff.T)):
        if t + 1 in save_at:
            snapshots[t + 1] = x.clone()
        tb  = torch.full((1,), t, device=device, dtype=torch.long)
        eps = _fwd_precomputed(model, x, tb, adj, mask, text_feat, text_mask)
        ab  = diff.alphas_bar[t]
        ap  = diff.alphas_bar_prev[t]
        a   = diff.alphas[t]
        b   = diff.betas[t]
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        mu  = (ap.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        x   = mu + diff.post_var[t].sqrt() * torch.randn_like(x) if t > 0 else mu

    snapshots[0] = x.clone()
    return snapshots


# ── 各面板绘制 ────────────────────────────────────────────────────────────────

def draw_diffusion_panel(ax, coords, adj, valid_mask, display=3.0, node_color=None):
    """统一蓝色节点（扩散帧）"""
    valid = np.where(valid_mask)[0]
    pts   = coords[valid]
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj[ni, nj] > 0.5:
                ax.plot([pts[ii, 0], pts[jj, 0]], [pts[ii, 1], pts[jj, 1]],
                        color='#BBBBBB', lw=0.6, alpha=0.7,
                        solid_capstyle='round', zorder=1)
    color = node_color or NODE_COLOR
    for i in range(len(valid)):
        ax.add_patch(plt.Circle((pts[i, 0], pts[i, 1]), 0.11,
                                color=color, ec='#333333', lw=0.5, zorder=3))
    ax.set_xlim(-display, display); ax.set_ylim(-display, display)
    ax.set_aspect('equal'); ax.axis('off')


def draw_type_panel(ax, coords, adj, valid_mask, type_ids, display=3.0):
    """节点按 θ₃ 预测类型着色"""
    valid = np.where(valid_mask)[0]
    pts   = coords[valid]
    # 归一化到 display 范围
    c = pts.mean(0)
    s = max(np.abs(pts - c).max(), 1.0)
    d = (pts - c) / s * (display * 0.85)

    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj[ni, nj] > 0.5:
                ax.plot([d[ii, 0], d[jj, 0]], [d[ii, 1], d[jj, 1]],
                        color='#BBBBBB', lw=0.6, alpha=0.7,
                        solid_capstyle='round', zorder=1)
    for i, vi in enumerate(valid):
        ax.add_patch(plt.Circle((d[i, 0], d[i, 1]), 0.14,
                                color=type_color(int(type_ids[vi])),
                                ec='#444444', lw=0.5, zorder=3))
    ax.set_xlim(-display, display); ax.set_ylim(-display, display)
    ax.set_aspect('equal'); ax.axis('off')


def draw_render_panel(ax, coords, adj_np, valid_mask, type_ids, id_to_combo):
    """投票渲染平面图"""
    n = int(valid_mask.sum())
    if n < 3:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Too few nodes', ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
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

    xs = [c[0] for c in raw_coords]
    ys = [c[1] for c in raw_coords]
    span   = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = span * 0.12
    mn_x, mn_y = min(xs), min(ys)

    def norm(x, y):
        return ((x - mn_x + margin) / (span + 2 * margin),
                (y - mn_y + margin) / (span + 2 * margin))

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
                ha='center', va='center', fontsize=7,
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


def add_arrow(fig, ax_left, ax_right, scale=9):
    xr  = ax_left.get_position().x1
    ymd = (ax_left.get_position().y0 + ax_left.get_position().y1) / 2
    xl  = ax_right.get_position().x0
    fig.add_artist(FancyArrowPatch(
        (xr + 0.002, ymd), (xl - 0.002, ymd),
        transform=fig.transFigure,
        arrowstyle='->', color='#888888',
        mutation_scale=scale, lw=0.9))


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--ckpt3',       default='checkpoints/node_type/20260616_223156/model_latest.pt')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--data',        default='data/processed/node_diffusion_cross_att/graph_dataset_6k.npz')
    p.add_argument('--combo_vocab', default='data/processed/type_combo_vocab.json')
    p.add_argument('--idx',         type=int, default=20)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--out',         default='paper_work/figures/diffusion_process.pdf')
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    id_to_combo = load_vocab(Path(args.combo_vocab))

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    data    = np.load(args.data, allow_pickle=True)
    idx     = args.idx
    adj_np  = data['adj_matrix'][idx].astype('float32')     # [40, 40]
    mask_np = data['node_mask'][idx].astype('float32')      # [40]
    ptok_np = data['prompt_tokens'][idx].astype('int64')
    pmsk_np = data['prompt_mask'][idx].astype('float32')
    valid_mask = mask_np > 0.5
    print(f'样本 #{idx}  n_nodes={valid_mask.sum()}')

    cond = {
        'adj_matrix':    torch.from_numpy(adj_np  ).unsqueeze(0),
        'node_mask':     torch.from_numpy(mask_np ).unsqueeze(0),
        'prompt_tokens': torch.from_numpy(ptok_np ).unsqueeze(0),
        'prompt_mask':   torch.from_numpy(pmsk_np ).unsqueeze(0),
    }

    # ════════════════════════════════════════════════════════════════════════
    # θ₂：DDPM 逆采样，保存每 200 步快照
    # ════════════════════════════════════════════════════════════════════════
    TIMESTEPS = [1000, 800, 600, 400, 200, 0]
    SAVE_AT   = set(TIMESTEPS)

    print('[θ₂] 加载模型...')
    model2 = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2  = torch.load(args.ckpt, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    print(f'  step={ckpt2.get("step", "?")}')

    diff  = GaussianDiffusion(T=1000)
    print('开始 DDPM 逆采样（1000步）...')
    snaps = ddpm_sample_with_snapshots(model2, diff, cond, device, SAVE_AT, seed=args.seed)
    print('采样完成')

    del model2
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₂] 模型已卸载')

    # 归一化参考：以 t=0 最终坐标为基准
    final_coords_raw = snaps[0][0].permute(1, 0).cpu().numpy()   # [40, 2]
    final_valid      = final_coords_raw[valid_mask]
    ref_center       = final_valid.mean(0)
    ref_scale        = max(np.abs(final_valid - ref_center).max(), 1.0)

    def to_display(snap_tensor, self_norm=False):
        coords = snap_tensor[0].permute(1, 0).cpu().numpy()
        if self_norm:
            v = coords[valid_mask]; c = v.mean(0)
            s = max(np.abs(v - c).max(), 1e-3)
            return (coords - c) / s * 2.5
        return (coords - ref_center) / ref_scale * 2.5

    # ════════════════════════════════════════════════════════════════════════
    # θ₃：节点类型预测
    # ════════════════════════════════════════════════════════════════════════
    print('\n[θ₃] 加载模型...')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step", "?")}')

    final_coords_t0 = snaps[0].cpu()   # [1, 2, 40]
    with torch.no_grad():
        x_in  = final_coords_t0.to(device)                              # [1, 2, 40]
        adj_t = torch.from_numpy(adj_np ).unsqueeze(0).to(device)
        msk_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        ptk_t = torch.from_numpy(ptok_np).unsqueeze(0).to(device)
        pmk_t = torch.from_numpy(pmsk_np).unsqueeze(0).long().to(device)
        logits    = model3(x_in, adj_matrix=adj_t, node_mask=msk_t,
                           prompt_tokens=ptk_t, prompt_mask=pmk_t)
        type_ids  = logits[0].argmax(dim=-1).cpu().numpy()   # [40]

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₃] 模型已卸载')

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
    })

    # ── 蛇形 2×4 布局 ─────────────────────────────────────────────────────────
    # Row 0 (左→右): t=1000, t=800, t=600, t=400
    # Row 1 (右→左): t=200,  t=0,   θ₃,   Rendered
    #                col 3    col 2  col 1  col 0
    PANEL_W  = 1.8
    PANEL_H  = 1.8
    GAP_X    = 0.15   # 水平间隔（留给箭头）
    GAP_Y    = 0.45   # 垂直间隔（留给竖向箭头）
    PAD_T    = 0.18   # 顶部留给标题
    PAD_B    = 0.22   # 底部留给第二行标题

    N_COLS   = 4
    FIG_W    = PANEL_W * N_COLS + GAP_X * (N_COLS - 1)
    FIG_H    = PANEL_H * 2 + GAP_Y + PAD_T + PAD_B

    DISPLAY  = 3.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def make_ax(row, col):
        left   = (col * (PANEL_W + GAP_X)) / FIG_W
        width  = PANEL_W / FIG_W
        if row == 0:
            bottom = (PAD_B + PANEL_H + GAP_Y) / FIG_H
        else:
            bottom = PAD_B / FIG_H
        height = PANEL_H / FIG_H
        return fig.add_axes([left, bottom, width, height])

    # 创建 2×4 axes 网格
    axs = [[make_ax(r, c) for c in range(N_COLS)] for r in range(2)]

    def style_ax(ax):
        ax.set_facecolor('#FAFAFA')
        for sp in ax.spines.values():
            sp.set_edgecolor('#DDDDDD'); sp.set_linewidth(0.5)

    # Row 0: t=1000, t=800, t=600, t=400
    row0_data = [
        (1000, '$t=1000$', True),
        (800,  '$t=800$',  False),
        (600,  '$t=600$',  False),
        (400,  '$t=400$',  False),
    ]
    for col, (t_val, label, self_norm) in enumerate(row0_data):
        ax = axs[0][col]
        style_ax(ax)
        coords = to_display(snaps[t_val], self_norm=self_norm)
        draw_diffusion_panel(ax, coords, adj_np, valid_mask, display=DISPLAY)
        ax.set_title(label, fontsize=8, pad=2, fontfamily='serif', fontstyle='italic')

    def below_title(ax, label, italic=False):
        ax.text(0.5, -0.06, label,
                ha='center', va='top', fontsize=8,
                fontfamily='serif',
                fontstyle='italic' if italic else 'normal',
                transform=ax.transAxes)

    # Row 1 (右→左): col3=t=200, col2=t=0, col1=Type Prediction, col0=Rendered
    ax = axs[1][3]; style_ax(ax)
    draw_diffusion_panel(ax, to_display(snaps[200]), adj_np, valid_mask, display=DISPLAY)
    below_title(ax, '$t=200$', italic=True)

    ax = axs[1][2]; style_ax(ax)
    draw_diffusion_panel(ax, to_display(snaps[0]), adj_np, valid_mask, display=DISPLAY)
    below_title(ax, '$t=0$', italic=True)

    ax = axs[1][1]; style_ax(ax)
    draw_type_panel(ax, to_display(snaps[0]), adj_np, valid_mask, type_ids, display=DISPLAY)
    below_title(ax, 'Type Prediction')

    ax = axs[1][0]
    draw_render_panel(ax, final_coords_raw, adj_np, valid_mask, type_ids, id_to_combo)
    below_title(ax, 'Rendered Floor Plan')

    # ── 箭头 ──────────────────────────────────────────────────────────────────
    def arrow(x0, y0, x1, y1):
        fig.add_artist(FancyArrowPatch(
            (x0, y0), (x1, y1),
            transform=fig.transFigure,
            arrowstyle='->', color='#888888',
            mutation_scale=9, lw=0.9))

    # Row 0: 左→右水平箭头
    for col in range(N_COLS - 1):
        xr = axs[0][col].get_position().x1
        xl = axs[0][col + 1].get_position().x0
        ym = (axs[0][col].get_position().y0 + axs[0][col].get_position().y1) / 2
        arrow(xr + 0.005, ym, xl - 0.005, ym)

    # 竖向箭头：row0 col3 → row1 col3
    xm  = (axs[0][3].get_position().x0 + axs[0][3].get_position().x1) / 2
    yb0 = axs[0][3].get_position().y0
    yt1 = axs[1][3].get_position().y1
    arrow(xm, yb0 - 0.01, xm, yt1 + 0.01)

    # Row 1: 右→左水平箭头（col3→col2→col1→col0）
    for col in range(N_COLS - 1, 0, -1):
        xl = axs[1][col].get_position().x0
        xr = axs[1][col - 1].get_position().x1
        ym = (axs[1][col].get_position().y0 + axs[1][col].get_position().y1) / 2
        arrow(xl - 0.005, ym, xr + 0.005, ym)

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_pdf = args.out
    out_png = args.out.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight', dpi=200)
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'保存: {out_pdf}')
    print(f'保存: {out_png}')


if __name__ == '__main__':
    main()
