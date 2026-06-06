"""
v3 数据集可视化检查：随机抽10个样本
左列：原始平面图  右列：顶点图（节点按 combo 类型着色，显示类型标签）
Run from project root: python paper_work/figures/draw_v3_check.py
"""
import json, random, textwrap
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 路径 ────────────────────────────────────────────────────────────────────
V3_PATH   = 'data/jsonl/final_graph_dataset_v3.jsonl'
VOCAB_PATH = 'data/processed/type_combo_vocab_new.json'
SRC_DIR   = Path('data/Architext_v1/train_jsonl')
OUT_PNG   = 'paper_work/figures/v3_check.png'
SEED      = 42
N_SAMPLES = 10

# ── 加载 vocab ───────────────────────────────────────────────────────────────
with open(VOCAB_PATH, encoding='utf-8') as f:
    vocab = json.load(f)

id_to_combo = {int(k): v for k, v in vocab['id_to_combo'].items()}

ROOM_COLORS = {
    'bathroom':    '#AED6F1',
    'bedroom':     '#A9DFBF',
    'living_room': '#F9E79F',
    'kitchen':     '#F1948A',
    'corridor':    '#D7BDE2',
    'dining_room': '#FAD7A0',
    'other':       '#DDDDDD',
}

# combo 节点颜色：混合所有成员颜色
BASE_RGB = {
    'bathroom':    (174, 214, 241),
    'bedroom':     (169, 223, 155),
    'living_room': (249, 231, 159),
    'kitchen':     (241, 148, 138),
    'corridor':    (215, 189, 226),
    'dining_room': (250, 215, 160),
    'other':       (221, 221, 221),
}

def blend_color(combo):
    if not combo:
        return '#DDDDDD'
    rgbs = [BASE_RGB.get(t, (200,200,200)) for t in combo]
    avg = tuple(int(sum(c[i] for c in rgbs)/len(rgbs)) for i in range(3))
    return '#{:02x}{:02x}{:02x}'.format(*avg)

def darken(hex_color, f=0.55):
    r,g,b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    return '#{:02x}{:02x}{:02x}'.format(int(r*f),int(g*f),int(b*f))

def combo_label(combo):
    shorts = {'bathroom':'Bath','bedroom':'Bed','living_room':'Liv',
              'kitchen':'Kit','corridor':'Cor','dining_room':'Din','other':'Oth'}
    return '+'.join(shorts.get(t,t[:3]) for t in combo)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
with open(V3_PATH, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

rng = random.Random(SEED)
indices = rng.sample(range(len(all_lines)), N_SAMPLES)
records = [json.loads(all_lines[i]) for i in indices]

# ── 画图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(12, N_SAMPLES * 2.8))
fig.patch.set_facecolor('#F8F8F8')

_src_cache = {}
def load_source(source_file, source_line):
    key = (source_file, int(source_line))
    if key not in _src_cache:
        with (SRC_DIR / source_file).open(encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                if ln == int(source_line):
                    _src_cache[key] = json.loads(line)
                    break
    return _src_cache.get(key)

for row, rec in enumerate(records):
    n     = rec['n_nodes']
    coords = np.array(rec['node_coords'][:n], dtype=float)
    cids   = rec['node_combo_ids'][:n]
    mask   = rec['node_mask'][:n]
    valid  = [i for i in range(n) if mask[i] == 1]
    idx    = indices[row]

    # ── 左：原始平面图 ───────────────────────────────────────────────────────
    ax_fp = axes[row][0]
    ax_fp.set_aspect('equal'); ax_fp.axis('off')

    src = load_source(rec['source_file'], rec['source_line'])
    if src:
        rooms = src['rooms']
        all_x = [c[0] for r in rooms for c in r['coords']]
        all_y = [c[1] for r in rooms for c in r['coords']]
        pad = 8
        ax_fp.set_xlim(min(all_x)-pad, max(all_x)+pad)
        ax_fp.set_ylim(min(all_y)-pad, max(all_y)+pad)
        for room in rooms:
            pts = np.array(room['coords'], dtype=float)
            fc = ROOM_COLORS.get(room['type'], '#EEE')
            ax_fp.add_patch(mpatches.Polygon(pts, closed=True,
                facecolor=fc, edgecolor='#555', lw=1.0))
            cx, cy = pts.mean(0)
            ax_fp.text(cx, cy, room['type'][:3], ha='center', va='center',
                      fontsize=7, color='#222')

    if row == 0:
        ax_fp.set_title('Floor Plan', fontsize=11, fontweight='bold', pad=4)
    ax_fp.text(0.01, 0.99, f'#{idx}', transform=ax_fp.transAxes,
               fontsize=7, va='top', color='#888')

    # ── 右：顶点图 ──────────────────────────────────────────────────────────
    ax_g = axes[row][1]
    ax_g.set_aspect('equal'); ax_g.axis('off')

    pts = coords[valid]
    pad_g = 12
    ax_g.set_xlim(pts[:,0].min()-pad_g, pts[:,0].max()+pad_g)
    ax_g.set_ylim(pts[:,1].min()-pad_g, pts[:,1].max()+pad_g)

    # 画边（来自 adj_matrix）
    adj = np.array(rec['adj_matrix'])[:n, :n]
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            ni, nj = valid[i], valid[j]
            if adj[ni][nj] == 1:
                ax_g.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                          color='#BBBBBB', lw=0.6, zorder=1)

    # 画节点
    span = max(pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), 1)
    node_r = span * 0.035

    for i, ni in enumerate(valid):
        cid   = cids[ni]
        combo = id_to_combo.get(cid, ['other'])
        fc    = blend_color(combo)
        ec    = darken(fc)
        ax_g.add_patch(plt.Circle((pts[i,0], pts[i,1]), node_r,
                                   color=fc, ec=ec, lw=0.8, zorder=3))
        label = combo_label(combo)
        # 标签：只显示在节点旁边
        ax_g.text(pts[i,0], pts[i,1]+node_r+1.2, label,
                  ha='center', va='bottom', fontsize=5.5,
                  color='#333', zorder=4)

    if row == 0:
        ax_g.set_title('Vertex Graph (v3 combo types)', fontsize=11,
                       fontweight='bold', pad=4)

plt.tight_layout(pad=0.8, h_pad=0.5)
plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
print(f'Saved {OUT_PNG}')
