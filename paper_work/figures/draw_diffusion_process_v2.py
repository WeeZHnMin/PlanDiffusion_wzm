"""
逆扩散过程可视化（真实模型推理版）：
用训练好的 NodeDiffusionTransformer 从噪声还原坐标，每 200 步保存一帧。
6 个面板：t=1000（纯噪声）→ t=800 → t=600 → t=400 → t=200 → t=0（清晰）

用法（从项目根目录运行）：
  python paper_work/figures/draw_diffusion_process_v2.py \
      --ckpt checkpoints/node_diffusion_cross_att/latest.pt \
      --bert models/bert-base-uncased \
      --data data/processed/node_diffusion_cross_att/graph_dataset_6k.npz \
      --idx  20
"""

import argparse
import math
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch

sys.path.insert(0, '.')
from node_diffusion_cross_att.model import NodeDiffusionTransformer


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


# ── 逆扩散采样，保存中间帧 ────────────────────────────────────────────────────

def _forward_with_precomputed_text(model, x, tb, adj, mask, text_feat, text_mask):
    from node_diffusion_cross_att.model import timestep_embedding
    x_in     = x.permute(0, 2, 1).float()
    t_emb    = model.time_embed(timestep_embedding(tb, model.model_channels)).unsqueeze(1)
    h        = model.input_emb(x_in) + t_emb
    adj_mask = model._build_adj_mask(adj.float(), mask.float())
    for layer in model.layers:
        h = layer(h, adj_mask, text_feat, text_mask)
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
        eps = _forward_with_precomputed_text(model, x, tb, adj, mask, text_feat, text_mask)

        ab = diff.alphas_bar[t]
        ap = diff.alphas_bar_prev[t]
        a  = diff.alphas[t]
        b  = diff.betas[t]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        mu = (ap.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        x  = mu + diff.post_var[t].sqrt() * torch.randn_like(x) if t > 0 else mu

    snapshots[0] = x.clone()
    return snapshots


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def draw_panel(ax, coords, adj, valid_mask, xlim, ylim):
    valid = np.where(valid_mask)[0]
    pts   = coords[valid]
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj[ni, nj] > 0.5:
                ax.plot([pts[ii, 0], pts[jj, 0]],
                        [pts[ii, 1], pts[jj, 1]],
                        color='#BBBBBB', lw=0.6, alpha=0.7,
                        solid_capstyle='round', zorder=1)
    for i in range(len(valid)):
        ax.add_patch(plt.Circle((pts[i, 0], pts[i, 1]), 0.11,
                                color=NODE_COLOR, ec='#333333', lw=0.5, zorder=3))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal'); ax.axis('off')


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--bert', default='models/bert-base-uncased')
    p.add_argument('--data', default='data/processed/node_diffusion_cross_att/graph_dataset_6k.npz')
    p.add_argument('--idx',  type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out',  default='paper_work/figures/diffusion_process.pdf')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'模型加载完成  step={ckpt.get("step", "?")}')

    # ── 加载样本 ──────────────────────────────────────────────────────────────
    data    = np.load(args.data, allow_pickle=True)
    idx     = args.idx
    adj_np  = data['adj_matrix'][idx].astype('float32')
    mask_np = data['node_mask'][idx].astype('float32')
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

    # ── 逆扩散采样（每 200 步保存一帧）──────────────────────────────────────
    TIMESTEPS = [1000, 800, 600, 400, 200, 0]
    LABELS    = ['$t=1000$', '$t=800$', '$t=600$', '$t=400$', '$t=200$', '$t=0$']
    SAVE_AT   = set(TIMESTEPS)

    diff  = GaussianDiffusion(T=1000)
    print('开始 DDPM 逆采样（1000步）...')
    snaps = ddpm_sample_with_snapshots(model, diff, cond, device, SAVE_AT, seed=args.seed)
    print('采样完成')

    # 以 t=0 最终坐标为归一化参考
    final_coords = snaps[0][0].permute(1, 0).cpu().numpy()
    final_valid  = final_coords[valid_mask]
    ref_center   = final_valid.mean(0)
    ref_scale    = max(np.abs(final_valid - ref_center).max(), 1.0)

    def to_display(snap_tensor, self_norm=False):
        coords = snap_tensor[0].permute(1, 0).cpu().numpy()
        if self_norm:
            v = coords[valid_mask]
            c = v.mean(0)
            s = max(np.abs(v - c).max(), 1e-3)
            return (coords - c) / s * 2.5
        return (coords - ref_center) / ref_scale * 2.5

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
    })

    N        = len(TIMESTEPS)
    PANEL_W  = 1.4
    ARROW_W  = 0.08
    FIG_H    = 1.65
    FIG_W    = PANEL_W * N + ARROW_W * (N - 1)
    DISPLAY  = 3.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_bottom = 0.03
    panel_height = 1.0 - panel_bottom - 0.07

    axes = []
    for col in range(N):
        left  = col * (PANEL_W + ARROW_W) / FIG_W
        width = PANEL_W / FIG_W
        ax    = fig.add_axes([left, panel_bottom, width, panel_height])
        axes.append(ax)

    for col, (t_val, label) in enumerate(zip(TIMESTEPS, LABELS)):
        ax     = axes[col]
        coords = to_display(snaps[t_val], self_norm=(t_val == 1000))

        ax.set_facecolor('#FAFAFA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#DDDDDD')
            spine.set_linewidth(0.5)

        draw_panel(ax, coords, adj_np, valid_mask,
                   xlim=(-DISPLAY, DISPLAY), ylim=(-DISPLAY, DISPLAY))
        ax.set_title(label, fontsize=8, pad=2,
                     fontfamily='serif', fontstyle='italic')

        if col < N - 1:
            xr  = ax.get_position().x1
            ymd = (ax.get_position().y0 + ax.get_position().y1) / 2
            fig.add_artist(FancyArrowPatch(
                (xr + 0.002, ymd), (xr + ARROW_W / FIG_W - 0.004, ymd),
                transform=fig.transFigure,
                arrowstyle='->', color='#888888',
                mutation_scale=9, lw=0.9))

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
