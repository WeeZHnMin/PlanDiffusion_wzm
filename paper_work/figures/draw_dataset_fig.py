"""
Dataset figure: 3 rows × 3 columns
  Col 0: floor plan redraw from source polygons
  Col 1: vertex graph  — nodes colored by combo type, legend below
  Col 2: text description (wrapped)
Run from project root: python paper_work/figures/draw_dataset_fig.py
"""
import json, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/jsonl/final_graph_dataset_v3.jsonl'
VOCAB_PATH = 'data/processed/type_combo_vocab_old.json'
SRC_DIR    = Path('data/Architext_v1/train_jsonl')
OUT_PDF    = 'paper_work/figures/dataset_fig.pdf'
OUT_PNG    = 'paper_work/figures/dataset_fig.png'

SAMPLE_INDICES = [2, 56, 10]

# ── Load combo vocab ─────────────────────────────────────────────────────────
with open(VOCAB_PATH, encoding='utf-8') as f:
    vocab = json.load(f)

# combo_id → list of base type ints, e.g. 8 → [1, 2]
id_to_bases = {}
for k, v in vocab['combo_to_id'].items():
    bases = [int(x) for x in k.strip('[]').split(', ') if x.strip()]
    id_to_bases[v] = bases

BASE_NAMES = {int(k): v for k, v in vocab['base_type_names'].items()}
# short display labels
SHORT = {
    'bathroom':    'Bath',
    'bedroom':     'Bed',
    'living_room': 'Living',
    'kitchen':     'Kit',
    'corridor':    'Corr',
    'dining_room': 'Dining',
    'other':       'Other',
}
OTHER_BASE_ID = next((k for k, v in BASE_NAMES.items() if v == 'other'), 7)

# Base RGB per type (for blending)
BASE_RGB = {
    1: (177, 148, 211),  # bathroom  – purple
    2: (123, 191, 234),  # bedroom   – blue
    3: (245, 194, 107),  # living    – yellow-orange
    4: (116, 196, 118),  # kitchen   – green
    5: (187, 187, 187),  # corridor  – grey
    6: (255, 200, 150),  # dining    – peach
    7: (220, 220, 220),  # other     – light grey
}
ROOM_FACE = {
    'bathroom': '#AED6F1',
    'bedroom': '#A9DFBF',
    'living_room': '#F9E79F',
    'kitchen': '#F1948A',
    'corridor': '#D7BDE2',
    'dining_room': '#FAD7A0',
    'other': '#DDDDDD',
}
ROOM_LABEL = {
    'bathroom': 'Bath',
    'bedroom': 'Bed',
    'living_room': 'Living',
    'kitchen': 'Kitchen',
    'corridor': 'Corridor',
    'dining_room': 'Dining',
    'other': 'Other',
}

def _hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*[int(x) for x in rgb])

def display_bases(cid):
    return id_to_bases.get(cid, [])

def combo_fc(cid):
    bases = display_bases(cid)
    if not bases:
        return '#DDDDDD'
    rgbs = [BASE_RGB.get(b, (200, 200, 200)) for b in bases]
    avg  = tuple(sum(c[i] for c in rgbs) / len(rgbs) for i in range(3))
    return _hex(avg)

def combo_ec(cid, factor=0.55):
    fc = combo_fc(cid)
    r, g, b = int(fc[1:3],16), int(fc[3:5],16), int(fc[5:7],16)
    return _hex((r*factor, g*factor, b*factor))

def combo_label(cid):
    bases = display_bases(cid)
    if not bases:
        return f'#{cid}'
    return '+'.join(SHORT.get(BASE_NAMES.get(b, ''), str(b)) for b in bases)

_SOURCE_CACHE = {}

def load_source_record(source_file, source_line):
    key = (source_file, int(source_line))
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]

    src_path = SRC_DIR / source_file
    with src_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if line_no == int(source_line):
                rec = json.loads(line)
                _SOURCE_CACHE[key] = rec
                return rec
    raise ValueError(f'Could not find {source_file}:{source_line}')

def polygon_centroid(points):
    n = len(points)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for k in range(n):
        x0, y0 = points[k]
        x1, y1 = points[(k + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    return cx / (6 * area), cy / (6 * area)

def draw_floor_plan(ax, source_rec):
    rooms = source_rec.get('rooms', [])
    if not rooms:
        ax.text(0.5, 0.5, 'no rooms', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888888')
        ax.axis('off')
        return

    all_x = [c[0] for room in rooms for c in room['coords']]
    all_y = [c[1] for room in rooms for c in room['coords']]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    span = max(xmax - xmin, ymax - ymin, 1)
    pad = span * 0.10 + 3

    ax.set_aspect('equal')
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axis('off')

    for room in rooms:
        pts = np.array(room['coords'], dtype=float)
        rtype = room['type']
        ax.add_patch(mpatches.Polygon(
            pts,
            closed=True,
            facecolor=ROOM_FACE.get(rtype, ROOM_FACE['other']),
            edgecolor='#666666',
            linewidth=0.8,
            joinstyle='round',
        ))

        cx, cy = polygon_centroid(room['coords'])
        area = abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
        ) / 2.0
        fs = 7.5 if area < 700 else 9.0
        ax.text(cx, cy, ROOM_LABEL.get(rtype, rtype),
                ha='center', va='center', fontsize=fs,
                color='#2A2A2A', clip_on=True)

# ── Load JSONL records ───────────────────────────────────────────────────────
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
records = [json.loads(all_lines[i]) for i in SAMPLE_INDICES]

# ── Figure layout ────────────────────────────────────────────────────────────
N_ROWS  = len(records)
ROW_H   = 2.6      # inches per row
FIG_W   = 7.2
LEGEND_H = 0.34    # inches reserved at bottom for shared legend
fig = plt.figure(figsize=(FIG_W, ROW_H * N_ROWS + LEGEND_H))

# Collect all combo ids across all rows for the shared legend
all_combo_ids = sorted({
    rec['node_combo_ids'][i]
    for rec in records
    for i in range(rec['n_nodes'])
    if rec['node_mask'][i] == 1
})

# Proportional column widths: image | graph | text
COL_RATIOS = [1.0, 1.1, 1.5]
total_r    = sum(COL_RATIOS)
fig_h_in   = ROW_H * N_ROWS + LEGEND_H
leg_frac   = LEGEND_H / fig_h_in      # fraction of figure height for legend
L, R, T    = 0.02, 0.02, 0.02
B          = leg_frac + 0.02           # leave room for legend at bottom
GAP_ROW    = 0.01
GAP_COL    = 0.005

# row 0: short (少文字), row 1: medium, row 2: tall (多文字)
ROW_H_RATIOS = [1.0, 1.0, 1.0]
avail_h      = 1 - T - B - GAP_ROW * (N_ROWS - 1)
row_h_fracs  = [r / sum(ROW_H_RATIOS) * avail_h for r in ROW_H_RATIOS]

def make_ax(row, col):
    y0  = B + sum(row_h_fracs[r] + GAP_ROW for r in range(N_ROWS - 1, row, -1))
    x0  = L + sum(COL_RATIOS[:col]) / total_r * (1 - L - R) + GAP_COL * col
    w   = COL_RATIOS[col] / total_r * (1 - L - R) - GAP_COL
    return fig.add_axes([x0, y0, w, row_h_fracs[row]])

axes = [[make_ax(r, c) for c in range(3)] for r in range(N_ROWS)]

# ── Draw each row ────────────────────────────────────────────────────────────
for row, rec in enumerate(records):
    n      = rec['n_nodes']
    coords = np.array(rec['node_coords'][:n], dtype=float)
    adj    = np.array(rec['adj_matrix'])[:n, :n]
    mask   = rec['node_mask'][:n]
    cids   = rec['node_combo_ids'][:n]
    valid  = [i for i in range(n) if mask[i] == 1]
    prompt = rec['prompt']

    # ── Col 0: floor plan redraw ─────────────────────────────────────────────
    ax_img = axes[row][0]
    try:
        source_rec = load_source_record(rec['source_file'], rec['source_line'])
        draw_floor_plan(ax_img, source_rec)
    except (FileNotFoundError, KeyError, ValueError):
        ax_img.text(0.5, 0.5, 'not found', ha='center', va='center',
                    transform=ax_img.transAxes, fontsize=10, color='#888888')
        ax_img.axis('off')
    if row == 0:
        ax_img.set_title('Floor Plan', fontsize=11, fontweight='bold', pad=4)

    # ── Col 1: vertex graph ──────────────────────────────────────────────────
    ax_g = axes[row][1]
    ax_g.set_aspect('equal')
    ax_g.axis('off')

    pts = coords[valid]
    pad_g = 8
    ax_g.set_xlim(pts[:,0].min()-pad_g, pts[:,0].max()+pad_g)
    ax_g.set_ylim(pts[:,1].min()-pad_g, pts[:,1].max()+pad_g)

    # edges
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            ni, nj = valid[i], valid[j]
            if adj[ni][nj] == 1:
                ax_g.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                          color='#AAAAAA', lw=0.75, alpha=0.8,
                          solid_capstyle='round', zorder=1)

    # nodes — smaller radius
    span = max(pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), 1)
    node_r = span * 0.030

    for i, ni in enumerate(valid):
        cid = cids[ni]
        fc  = combo_fc(cid)
        ec  = combo_ec(cid)
        ax_g.add_patch(plt.Circle((pts[i,0], pts[i,1]), node_r,
                                   color=fc, ec=ec, lw=0.8, zorder=3))


    if row == 0:
        ax_g.set_title('Vertex Graph', fontsize=11, fontweight='bold', pad=4)

    # ── Col 2: text description ──────────────────────────────────────────────
    ax_t = axes[row][2]
    ax_t.axis('off')

    wrapped = textwrap.fill(prompt, width=32)
    ax_t.text(0.05, 0.97, wrapped,
              transform=ax_t.transAxes,
              fontsize=10.5, va='top', ha='left',
              linespacing=1.22, color='#1A1A1A',
              clip_on=False)

    ax_t.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.01',
        linewidth=0.6, edgecolor='#CCCCCC', facecolor='#FAFAFA',
        transform=ax_t.transAxes, zorder=0))

    if row == 0:
        ax_t.set_title('Text Description', fontsize=11, fontweight='bold', pad=4)

# ── Single shared legend at bottom ──────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=combo_fc(cid), edgecolor=combo_ec(cid),
                   label=combo_label(cid), linewidth=0.8)
    for cid in all_combo_ids
]
ax_leg = fig.add_axes([L, 0.005, 1 - L - R, leg_frac - 0.01])
ax_leg.axis('off')
ax_leg.legend(handles=legend_handles,
              loc='center', ncol=6,
              fontsize=9.0, frameon=False,
              columnspacing=0.55, handlelength=0.9, handletextpad=0.35,
              borderpad=0.35, labelspacing=0.35,
              title='Node Type', title_fontsize=9.5)

plt.savefig(OUT_PDF, bbox_inches='tight', dpi=150)
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
print(f'Saved {OUT_PDF}')
print(f'Saved {OUT_PNG}')
