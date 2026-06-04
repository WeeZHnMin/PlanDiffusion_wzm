"""
Forward diffusion process on vertex graph node coordinates.
6 panels: t=0, 200, 400, 600, 800, 1000  (left=clean → right=noisy)
Coords normalized to [-1,1] before DDPM so noise scale is correct.
Run from project root: python paper_work/figures/draw_diffusion_process.py
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── DDPM schedule ─────────────────────────────────────────────────────────────
T_MAX  = 1000
betas  = np.linspace(1e-4, 0.02, T_MAX)
ab     = np.cumprod(1.0 - betas)          # ᾱ_t

def q_sample(x0_norm, t_idx, seed=0):
    rng = np.random.default_rng(seed)      # 每帧独立噪声
    eps = rng.standard_normal(x0_norm.shape)
    return math.sqrt(ab[t_idx]) * x0_norm + math.sqrt(1 - ab[t_idx]) * eps

# ── Combo type colors (same as dataset_fig) ───────────────────────────────────
DATA_PATH  = 'data/jsonl/final_graph_dataset_v2.jsonl'
VOCAB_PATH = 'data/processed/type_combo_vocab_old.json'

with open(VOCAB_PATH, encoding='utf-8') as f:
    vocab = json.load(f)
id_to_bases = {v: [int(x) for x in k.strip('[]').split(', ') if x.strip()]
               for k, v in vocab['combo_to_id'].items()}
BASE_RGB = {
    1: (177, 148, 211),
    2: (123, 191, 234),
    3: (245, 194, 107),
    4: (116, 196, 118),
    5: (187, 187, 187),
    6: (255, 200, 150),
    7: (220, 220, 220),
}
def combo_fc(cid):
    bases = id_to_bases.get(cid, [])
    if not bases: return '#DDDDDD'
    rgbs = [BASE_RGB.get(b, (200,200,200)) for b in bases]
    avg  = tuple(sum(c[i] for c in rgbs)/len(rgbs) for i in range(3))
    return tuple(v/255 for v in avg)

def combo_ec(cid, f=0.55):
    fc = combo_fc(cid)
    return tuple(v*f for v in fc)

BASE_NAMES = {int(k): v for k, v in vocab['base_type_names'].items()}
SHORT = {
    'bathroom':    'Bath',
    'bedroom':     'Bed',
    'living_room': 'Living',
    'kitchen':     'Kit',
    'corridor':    'Corr',
    'dining_room': 'Dining',
    'other':       'Other',
}
def combo_label(cid):
    bases = id_to_bases.get(cid, [])
    if not bases: return f'#{cid}'
    return '+'.join(SHORT.get(BASE_NAMES.get(b,''), str(b)) for b in bases)

# ── Load record ───────────────────────────────────────────────────────────────
with open(DATA_PATH, encoding='utf-8') as f:
    rec = json.loads(f.readlines()[2])

n      = rec['n_nodes']
coords = np.array(rec['node_coords'][:n], dtype=float)
adj    = np.array(rec['adj_matrix'])[:n, :n]
mask   = rec['node_mask'][:n]
cids   = rec['node_combo_ids'][:n]
valid  = [i for i in range(n) if mask[i] == 1]

pts0 = coords[valid]              # [N, 2] clean coords

# Normalize: center and scale so t=0 fills ~70% of display panel
pts0 = pts0 - pts0.mean(axis=0)
scale = np.abs(pts0).max()
pts_norm = pts0 / scale * 2.7     # t=0 spans ~±2.7 in display range ±3.2

# ── Figure ────────────────────────────────────────────────────────────────────
TIMESTEPS = [999, 749, 499, 249, 0]
LABELS    = ['$t=1000$', '$t=750$', '$t=500$', '$t=250$', '$t=0$']
N_panels  = len(TIMESTEPS)

PANEL_W  = 1.4
ARROW_W  = 0.18
FIG_H    = 1.95
FIG_W    = PANEL_W * N_panels + ARROW_W * (N_panels - 1) + 0.1
LEGEND_H = 0.32

fig = plt.figure(figsize=(FIG_W, FIG_H))
panel_bottom = LEGEND_H / FIG_H + 0.03
panel_height = 1.0 - panel_bottom - 0.03
axes = []
for col in range(N_panels):
    left  = col * (PANEL_W + ARROW_W) / FIG_W + 0.008
    width = PANEL_W / FIG_W - 0.008
    ax = fig.add_axes([left, panel_bottom, width, panel_height])
    axes.append(ax)

DISPLAY_RANGE = 3.5
NODE_R = 0.11

for col, (t_idx, label) in enumerate(zip(TIMESTEPS, LABELS)):
    ax = axes[col]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)

    # Noisy coords
    if t_idx == 0:
        pts_t = pts_norm.copy()
    else:
        pts_t = q_sample(pts_norm, t_idx, seed=col * 7 + 13)  # 每帧独立噪声

    # Draw edges
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            ni, nj = valid[i], valid[j]
            if adj[ni][nj] == 1:
                ax.plot([pts_t[i,0], pts_t[j,0]],
                        [pts_t[i,1], pts_t[j,1]],
                        color='#BBBBBB', lw=0.6, alpha=0.7,
                        solid_capstyle='round', zorder=1)

    # Draw nodes
    for i, ni in enumerate(valid):
        cid = cids[ni]
        fc  = combo_fc(cid)
        ec  = combo_ec(cid)
        ax.add_patch(plt.Circle((pts_t[i,0], pts_t[i,1]), NODE_R,
                                color=fc, ec=ec, lw=0.6, zorder=3))

    # Light background panel
    ax.add_patch(mpatches.FancyBboxPatch(
        (-DISPLAY_RANGE*0.97, -DISPLAY_RANGE*0.97),
        DISPLAY_RANGE*1.94, DISPLAY_RANGE*1.94,
        boxstyle='round,pad=0.05',
        linewidth=0.5, edgecolor='#DDDDDD', facecolor='#FAFAFA',
        transform=ax.transData, zorder=0))

    # Timestep label
    ax.set_title(label, fontsize=8.5, pad=3,
                 fontfamily='serif', fontstyle='italic')

    # Arrow between panels (except after last)
    if col < N_panels - 1:
        x_arrow = ax.get_position().x1
        y_mid   = (ax.get_position().y0 + ax.get_position().y1) / 2
        fig.add_artist(
            FancyArrowPatch(
                (x_arrow + 0.002, y_mid),
                (x_arrow + 0.028, y_mid),
                transform=fig.transFigure,
                arrowstyle='->', color='#888888',
                mutation_scale=10, lw=1.0
            )
        )

# ── Shared legend at bottom ───────────────────────────────────────────────────
present = sorted({cids[ni] for ni in valid})
handles = [
    mpatches.Patch(facecolor=combo_fc(cid), edgecolor=combo_ec(cid),
                   label=combo_label(cid), linewidth=0.8)
    for cid in present
]
ax_leg = fig.add_axes([0.01, 0.0, 0.98, LEGEND_H / FIG_H])
ax_leg.axis('off')
ax_leg.legend(handles=handles,
              loc='center', ncol=len(present),
              fontsize=6.5, frameon=True,
              framealpha=0.95, edgecolor='#CCCCCC',
              title='Node Type', title_fontsize=7.0,
              columnspacing=0.5, handlelength=0.9, handletextpad=0.35,
              borderpad=0.4)

plt.savefig('paper_work/figures/diffusion_process.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('paper_work/figures/diffusion_process.png',
            bbox_inches='tight', dpi=200)
print('Saved diffusion_process.pdf / .png')
