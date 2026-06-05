"""
图序列化过程示意图 - 生成4个独立PNG，只有图，无文字
真实数据: data/jsonl/final_graph_dataset_v2.jsonl idx=44, n=10
Run: python drafts/figures/draw_seq_graph.py
输出: drafts/figures/seq_graph_1.png ~ seq_graph_4.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 图数据 (idx=44, n=10) ─────────────────────────────────────────────────────
TREE_EDGES  = [(0,1),(0,3),(0,4),(1,2),(2,6),(3,5),(3,8),(5,9),(7,8)]
EXTRA_EDGES = [(2,3),(4,5),(6,7),(8,9)]

POS = {
    0: (0.50, 0.82),
    1: (0.20, 0.57),
    2: (0.20, 0.32),
    3: (0.50, 0.57),
    4: (0.80, 0.57),
    5: (0.50, 0.32),
    6: (0.05, 0.10),
    7: (0.35, 0.10),
    8: (0.65, 0.32),
    9: (0.65, 0.10),
}

NODE_COLORS = {
    0:'#F5D48B', 1:'#A9CDE8', 2:'#D4A8D4', 3:'#F5A8A8', 4:'#A8D5A2',
    5:'#FFDDB0', 6:'#B0E0E0', 7:'#B0E0E0', 8:'#F5D4D4', 9:'#F5D4D4',
}

SNAPSHOTS = [
    ({0,1,3,4,2}, {(0,1),(0,3),(0,4),(1,2)}, set()),
    (set(range(10)), set(TREE_EDGES), set()),
    (set(range(10)), set(TREE_EDGES), {(2,3),(4,5)}),
    (set(range(10)), set(TREE_EDGES), set(EXTRA_EDGES)),
]

NODE_R   = 0.065
NODE_FONT= 12

for idx, (nodes, tree, extra) in enumerate(SNAPSHOTS, start=1):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # 背景框
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.08, -0.03), 1.16, 1.06,
        boxstyle='round,pad=0.02',
        linewidth=0.8, edgecolor='#CCCCCC', facecolor='#F8F8F8',
        transform=ax.transData, zorder=0))

    # 树边
    for (u, v) in tree:
        if u in nodes and v in nodes:
            x0,y0 = POS[u]; x1,y1 = POS[v]
            ax.plot([x0,x1],[y0,y1], color='#3A7EC6', lw=1.8,
                    solid_capstyle='round', zorder=1)

    # 补边
    for (u, v) in extra:
        if u in nodes and v in nodes:
            x0,y0 = POS[u]; x1,y1 = POS[v]
            ax.plot([x0,x1],[y0,y1], color='#D94040', lw=1.8,
                    linestyle='--', dash_capstyle='round', zorder=2)

    # 节点
    for i in nodes:
        x, y = POS[i]
        ax.add_patch(plt.Circle((x,y), NODE_R, color=NODE_COLORS[i],
                                 ec='#444444', lw=1.0, zorder=3))
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=NODE_FONT, fontweight='bold',
                color='#222222', zorder=4)

    plt.tight_layout(pad=0.1)
    out = f'drafts/figures/seq_graph_{idx}.png'
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}')
