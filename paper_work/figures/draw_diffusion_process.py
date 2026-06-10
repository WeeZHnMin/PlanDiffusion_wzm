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
from matplotlib.path import Path as MPath
import matplotlib.patches as mpl_patches

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
TIMESTEPS    = [999, 749, 499, 249, 0, None]
LABELS       = ['$t=1000$', '$t=750$', '$t=500$', '$t=250$', '$t=0$', 'Node Types']
# snake order: row0 left→right, row1 right→left
PANEL_POS    = [(0,0), (0,1), (0,2), (1,2), (1,1), (1,0)]
ARROW_LABELS = ['Diffusion', 'Diffusion', 'Diffusion', 'Diffusion', 'Type\nPrediction']
N_COLS, N_ROWS = 3, 2

PANEL_W  = 1.8
PANEL_H  = 2.0
ARROW_W  = 0.75
GAP_ROW  = 0.1
LEGEND_H = 0.42
FIG_W    = PANEL_W * N_COLS + ARROW_W * (N_COLS - 1) + 0.2
FIG_H    = PANEL_H * N_ROWS + GAP_ROW + LEGEND_H + 0.3

L, R, T  = 0.02, 0.02, 0.03
B        = LEGEND_H / FIG_H
arrow_w_frac = ARROW_W / FIG_W
panel_w_frac = (1 - L - R - arrow_w_frac * (N_COLS - 1)) / N_COLS
gap_row_frac = GAP_ROW / FIG_H
panel_h_frac = (1 - T - B - gap_row_frac) / N_ROWS

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Build axes at each (row, col) grid position
grid_axes = {}
for r, c in PANEL_POS:
    if (r, c) in grid_axes:
        continue
    x0 = L + c * (panel_w_frac + arrow_w_frac)
    y0 = B + (N_ROWS - 1 - r) * (panel_h_frac + gap_row_frac)
    grid_axes[(r, c)] = fig.add_axes([x0, y0, panel_w_frac, panel_h_frac])

DISPLAY_RANGE = 3.5
NODE_R = 0.24

for idx, (t_idx, label) in enumerate(zip(TIMESTEPS, LABELS)):
    r, c = PANEL_POS[idx]
    ax   = grid_axes[(r, c)]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-DISPLAY_RANGE, DISPLAY_RANGE)
    ax.set_ylim(-DISPLAY_RANGE, DISPLAY_RANGE)

    # Noisy coords
    if t_idx is None or t_idx == 0:
        pts_t = pts_norm.copy()
    else:
        pts_t = q_sample(pts_norm, t_idx, seed=idx * 7 + 13)

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
        if t_idx is None:
            fc = combo_fc(cids[ni])
            ec = combo_ec(cids[ni])
        else:
            fc = '#d8d8d8'
            ec = '#aaaaaa'
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
    ax.set_title(label, fontsize=16, pad=6,
                 fontfamily='serif', fontstyle='italic')

    # Arrow + label to next panel
    if idx < len(TIMESTEPS) - 1:
        r_next, c_next = PANEL_POS[idx + 1]
        pos_cur  = grid_axes[(r, c)].get_position()
        pos_next = grid_axes[(r_next, c_next)].get_position()
        alabel   = ARROW_LABELS[idx]

        if r == r_next:
            # horizontal arrow
            if c_next > c:   # rightward
                xa0 = pos_cur.x1  + 0.005
                xa1 = pos_next.x0 - 0.005
            else:             # leftward
                xa0 = pos_cur.x0  - 0.005
                xa1 = pos_next.x1 + 0.005
            ya = (pos_cur.y0 + pos_cur.y1) / 2
            fig.add_artist(FancyArrowPatch(
                (xa0, ya), (xa1, ya),
                transform=fig.transFigure,
                arrowstyle='->', color='#555555',
                mutation_scale=22, lw=2.0))
            # row 0 (rightward): label above; row 1 (leftward): label below
            if c_next > c:
                fig.text((xa0 + xa1) / 2, ya + 0.032, alabel,
                         ha='center', va='bottom', fontsize=9.5,
                         color='#333333', transform=fig.transFigure)
            else:
                fig.text((xa0 + xa1) / 2, ya - 0.018, alabel,
                         ha='center', va='top', fontsize=9.5,
                         color='#333333', transform=fig.transFigure)
        else:
            # bent arrow: right → down → left  (wraps around right side)
            x_start = pos_cur.x1
            y_start = (pos_cur.y0 + pos_cur.y1) / 2
            x_right = pos_cur.x1 + 0.055
            y_end   = (pos_next.y0 + pos_next.y1) / 2
            x_end   = pos_next.x1

            # draw three-segment line (no arrowhead yet)
            verts = [(x_start, y_start),
                     (x_right, y_start),
                     (x_right, y_end),
                     (x_end + 0.008, y_end)]
            codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]
            seg = mpl_patches.PathPatch(
                MPath(verts, codes),
                facecolor='none', edgecolor='#555555', lw=2.0,
                transform=fig.transFigure, clip_on=False, zorder=10)
            fig.add_artist(seg)

            # arrowhead at the end (pointing left onto right edge of next panel)
            fig.add_artist(FancyArrowPatch(
                (x_end + 0.008, y_end), (x_end, y_end),
                transform=fig.transFigure,
                arrowstyle='->', color='#555555',
                mutation_scale=22, lw=1.0))

            # label to the right of the vertical segment
            fig.text(x_right + 0.012, (y_start + y_end) / 2, alabel,
                     ha='left', va='center', fontsize=9.5,
                     color='#333333', transform=fig.transFigure)


# ── Legend at bottom ─────────────────────────────────────────────────────────
present = sorted({cids[ni] for ni in valid})
handles = [
    mpatches.Patch(facecolor=combo_fc(cid), edgecolor=combo_ec(cid),
                   label=combo_label(cid), linewidth=0.8)
    for cid in present
]
leg_y0 = B - LEGEND_H / FIG_H
ax_leg = fig.add_axes([L, leg_y0, 1 - L - R, LEGEND_H / FIG_H])
ax_leg.axis('off')
ax_leg.legend(handles=handles,
              loc='center', ncol=len(present),
              fontsize=9.5, frameon=False,
              title='Node Type', title_fontsize=10.5,
              columnspacing=0.5, handlelength=0.9, handletextpad=0.4,
              borderpad=0.4)

plt.savefig('paper_work/figures/diffusion_process.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('paper_work/figures/diffusion_process.png',
            bbox_inches='tight', dpi=200)
print('Saved diffusion_process.pdf / .png')
