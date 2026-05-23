"""
从 type_data.jsonl 提取节点数据，生成 node_data.jsonl
每条记录：
  prompt      : str
  node_coords : [[x,y], ...] 长度 N_MAX，padding 位置为 [0,0]（原始整数坐标）
  node_mask   : [1,...,0,...] 长度 N_MAX，1=真实节点
  adj_matrix  : N_MAX×N_MAX 的 01 矩阵（含自环，从 type_data.jsonl 直接读取）
  n_nodes     : int，真实节点数

同时输出可视化对比图（原始多边形 vs 节点图）用于验证正确性。
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Windows 中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SRC = Path(__file__).parent / "type_data.jsonl"
DST = Path(__file__).parent / "node_data.jsonl"
VIZ = Path(__file__).parent / "node_viz"
VIZ.mkdir(exist_ok=True)

ROOM_COLORS = {
    "bathroom":    "#AED6F1",
    "bedroom":     "#A9DFBF",
    "living_room": "#F9E79F",
    "kitchen":     "#F1948A",
    "corridor":    "#D7BDE2",
    "dining_room": "#FAD7A0",
}

# ── 读取原始数据 ──────────────────────────────────────────────────────────────
records = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

N_MAX = len(records[0]["adj_matrix"])
print(f"Records={len(records)}, N_MAX={N_MAX}")

# ── 坐标范围（从数据中读取）──────────────────────────────────────────────────
all_coords = [c for r in records for c in r["vertices"]]
X_MIN = min(c[0] for c in all_coords) - 5
X_MAX = max(c[0] for c in all_coords) + 5
Y_MIN = min(c[1] for c in all_coords) - 5
Y_MAX = max(c[1] for c in all_coords) + 5

def setup_ax(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MAX, Y_MIN)   # y 轴向下（y=0 在上方）
    ax.set_xlabel("x"); ax.set_ylabel("y")

# ── 构建 node_data 并可视化 ───────────────────────────────────────────────────
node_records = []

for idx, r in enumerate(records):
    n      = len(r["vertices"])
    coords = r["vertices"]   # list of [x, y]

    # padding 到 N_MAX
    node_coords = coords + [[0, 0]] * (N_MAX - n)
    node_mask   = [1] * n + [0] * (N_MAX - n)

    node_records.append({
        "prompt":      r["prompt"],
        "n_nodes":     n,
        "node_coords": node_coords,
        "node_mask":   node_mask,
        "adj_matrix":  r["adj_matrix"],
    })

    # ── 可视化 ────────────────────────────────────────────────────────────────
    fig, (ax_poly, ax_node) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：原始多边形
    for room in r["rooms"]:
        rtype = room["type"]
        color = ROOM_COLORS.get(rtype, "#CCCCCC")
        pts   = room["coords"]
        xs    = [c[0] for c in pts] + [pts[0][0]]
        ys    = [c[1] for c in pts] + [pts[0][1]]
        ax_poly.fill(xs, ys, color=color, alpha=0.6, zorder=1)
        ax_poly.plot(xs, ys, "k-", lw=1.2, zorder=2)
        cx = sum(c[0] for c in pts) / len(pts)
        cy = sum(c[1] for c in pts) / len(pts)
        ax_poly.text(cx, cy, rtype, ha="center", va="center",
                     fontsize=6, zorder=3)
    setup_ax(ax_poly, "原始多边形")

    # 右图：节点图（节点 + 邻接边）
    adj  = r["adj_matrix"]
    xs_n = [coords[k][0] for k in range(n)]
    ys_n = [coords[k][1] for k in range(n)]

    # 画边（蓝色）
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a][b]:
                ax_node.plot([xs_n[a], xs_n[b]], [ys_n[a], ys_n[b]],
                             color="steelblue", lw=1.2, alpha=0.7, zorder=1)

    # 画节点（红点）
    ax_node.scatter(xs_n, ys_n, c="tomato", s=40, zorder=3)

    # 标节点编号
    for k in range(n):
        ax_node.text(xs_n[k] + 0.4, ys_n[k] - 0.4, str(k),
                     fontsize=6, color="black", zorder=4)

    setup_ax(ax_node, f"节点图 (n={n})")

    fig.suptitle(f"[{idx+1}] {r['prompt']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(VIZ / f"sample_{idx+1:02d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

print(f"可视化图像已保存到 {VIZ}/")

# ── 保存 node_data.jsonl ──────────────────────────────────────────────────────
with open(DST, "w", encoding="utf-8") as f:
    for rec in node_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"已保存 {len(node_records)} 条记录 → {DST}")
for i, rec in enumerate(node_records):
    print(f"  [{i+1:2d}] n_nodes={rec['n_nodes']:2d}  {rec['prompt'][:60]}")
