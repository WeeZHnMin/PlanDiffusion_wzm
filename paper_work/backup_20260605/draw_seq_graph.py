"""
图序列化过程示意图
4个快照展示自回归生成过程中图结构的逐步构建：
  ① 部分父节点序列 → 前4个节点+树边
  ② 完整生成树（SEP之前）
  ③ 生成树 + 前2条补边
  ④ 完整图（EOS_G之前）

真实数据来自 data/jsonl/final_graph_dataset_v2.jsonl idx=44
Run: python drafts/figures/draw_seq_graph.py
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 真实图数据 (idx=44, n=10) ─────────────────────────────────────────────────
N = 10
# BFS order: [0,1,3,4,2,5,8,6,9,7]  parent_seq: [0,0,0,1,3,3,2,5,8]
TREE_EDGES = [(0,1),(0,3),(0,4),(1,2),(2,6),(3,5),(3,8),(5,9),(7,8)]
EXTRA_EDGES = [(2,3),(4,5),(6,7),(8,9)]
BFS_ORDER  = [0,1,3,4,2,5,8,6,9,7]
PARENT_SEQ = [0,0,0,1,3,3,2,5,8]   # parent of BFS_ORDER[1..9]

# 节点布局（手工调整使图形美观）
POS = {
    0: (0.50, 0.80),
    1: (0.20, 0.55),
    2: (0.20, 0.30),
    3: (0.50, 0.55),
    4: (0.80, 0.55),
    5: (0.50, 0.30),
    6: (0.05, 0.08),
    7: (0.35, 0.08),
    8: (0.65, 0.30),
    9: (0.65, 0.08),
}

# 节点类型颜色（来自真实数据 node_types: [8,1,11,7,2,9,3,3,4,4]）
NODE_COLORS = [
    '#F5D48B',  # 0  type 8
    '#A9CDE8',  # 1  type 1
    '#D4A8D4',  # 2  type 11
    '#F5A8A8',  # 3  type 7
    '#A8D5A2',  # 4  type 2
    '#FFDDB0',  # 5  type 9
    '#B0E0E0',  # 6  type 3
    '#B0E0E0',  # 7  type 3
    '#F5D4D4',  # 8  type 4
    '#F5D4D4',  # 9  type 4
]

# ── 4个快照定义 ───────────────────────────────────────────────────────────────
# 每个快照: (显示的节点集合, 树边集合, 补边集合, 标题token序列)
SNAPSHOTS = [
    # ① 生成前4个父节点token后：节点 0,1,3,4,2 出现，4条树边
    (
        {0,1,3,4,2},
        {(0,1),(0,3),(0,4),(1,2)},
        set(),
        r'BOS_G $N_{10}$ NODE$_0$ NODE$_0$ NODE$_0$ NODE$_1$ ...',
    ),
    # ② 完整生成树（SEP前）
    (
        set(range(10)),
        set(TREE_EDGES),
        set(),
        r'... NODE$_5$ NODE$_8$ SEP',
    ),
    # ③ 树 + 前2条补边
    (
        set(range(10)),
        set(TREE_EDGES),
        {(2,3),(4,5)},
        r'SEP NODE$_2$ NODE$_3$ NODE$_4$ NODE$_5$ ...',
    ),
    # ④ 完整图
    (
        set(range(10)),
        set(TREE_EDGES),
        set(EXTRA_EDGES),
        r'... NODE$_6$ NODE$_7$ NODE$_8$ NODE$_9$ EOS_G',
    ),
]

SNAPSHOT_LABELS = ['(1) Partial spanning tree', '(2) Full spanning tree', '(3) Extra edges (partial)', '(4) Complete graph']

# ── 绘图 ──────────────────────────────────────────────────────────────────────
ARROW_W  = 0.04
FIG_W    = 10.0
FIG_H    = 3.8
PANEL_W  = (FIG_W - ARROW_W * 3) / 4

fig = plt.figure(figsize=(FIG_W, FIG_H))

def panel_ax(col):
    left  = col * (PANEL_W + ARROW_W) / FIG_W
    ax = fig.add_axes([left + 0.01/FIG_W, 0.18, PANEL_W/FIG_W - 0.01/FIG_W, 0.68])
    return ax

NODE_R   = 0.06
FONT     = 7

for col, (nodes, tree, extra, token_str) in enumerate(SNAPSHOTS):
    ax = panel_ax(col)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # 背景
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.06, -0.03), 1.12, 1.06,
        boxstyle='round,pad=0.02',
        linewidth=0.6, edgecolor='#DDDDDD', facecolor='#FAFAFA',
        transform=ax.transData, zorder=0))

    # 树边（实线）
    for (u,v) in tree:
        if u in nodes and v in nodes:
            x0,y0 = POS[u]; x1,y1 = POS[v]
            ax.plot([x0,x1],[y0,y1], color='#4A90D9', lw=1.2,
                    solid_capstyle='round', zorder=1)

    # 补边（虚线，红色）
    for (u,v) in extra:
        if u in nodes and v in nodes:
            x0,y0 = POS[u]; x1,y1 = POS[v]
            ax.plot([x0,x1],[y0,y1], color='#E05C5C', lw=1.2,
                    linestyle='--', solid_capstyle='round', zorder=1)

    # 节点（只绘制已出现的节点）
    for i in nodes:
        x, y = POS[i]
        circle = plt.Circle((x, y), NODE_R, color=NODE_COLORS[i],
                             ec='#555555', lw=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=FONT, fontweight='bold', color='#333333', zorder=4)

    # 小标题
    ax.set_title(SNAPSHOT_LABELS[col], fontsize=8, pad=4, fontweight='bold')

    # token 序列说明（底部）
    ax.text(0.5, -0.15, token_str, ha='center', va='top',
            fontsize=7.5, transform=ax.transAxes,
            color='#222222', family='serif')

    # 箭头（面板之间）
    if col < 3:
        x_arrow = (col + 1) * (PANEL_W + ARROW_W) / FIG_W - ARROW_W / FIG_W
        y_mid   = 0.18 + 0.68 / 2
        fig.add_artist(
            matplotlib.patches.FancyArrowPatch(
                (x_arrow + 0.002, y_mid),
                (x_arrow + ARROW_W/FIG_W - 0.002, y_mid),
                transform=fig.transFigure,
                arrowstyle='->', color='#888888',
                mutation_scale=10, lw=1.0
            )
        )

# 图例
legend_elements = [
    mpatches.Patch(facecolor='white', edgecolor='#4A90D9', linewidth=1.2, label='Spanning tree edge'),
    mpatches.Patch(facecolor='white', edgecolor='#E05C5C', linewidth=1.2,
                   linestyle='--', label='Extra edge'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=8, frameon=True, framealpha=0.9,
           edgecolor='#CCCCCC', bbox_to_anchor=(0.5, -0.04),
           handlelength=1.5, columnspacing=1.5)

plt.savefig('drafts/figures/seq_graph.pdf', bbox_inches='tight', dpi=200)
plt.savefig('drafts/figures/seq_graph.png', bbox_inches='tight', dpi=200)
print('Saved drafts/figures/seq_graph.pdf / .png')
