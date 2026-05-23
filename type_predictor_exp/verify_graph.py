"""
逐样本验证节点提取 + 邻接矩阵是否正确。
对每个样本：
  1. 打印原始房间多边形顶点
  2. 打印去重后的节点列表（编号 + 坐标）
  3. 打印每条边的连接情况（是否有中间节点被正确切入）
  4. 输出可视化：左=原始多边形（顶点标坐标），右=节点图（节点标编号）
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

SRC = Path(__file__).parent / "type_data.jsonl"
VIZ = Path(__file__).parent / "verify_viz"
VIZ.mkdir(exist_ok=True)

records = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

ROOM_COLORS = {
    "bathroom": "#AED6F1", "bedroom": "#A9DFBF", "living_room": "#F9E79F",
    "kitchen": "#F1948A", "corridor": "#D7BDE2", "dining_room": "#FAD7A0",
}

for idx, r in enumerate(records):
    verts = r["vertices"]       # list of [x,y]
    adj   = r["vertex_adj"]     # list of lists
    rooms = r["rooms"]
    n     = len(verts)

    print(f"\n{'='*70}")
    print(f"[{idx+1}] {r['prompt']}")
    print(f"  节点数: {n}")

    # ── 1. 打印原始房间顶点 ────────────────────────────────────────────────
    print("  原始房间顶点:")
    for room in rooms:
        print(f"    {room['type']}: {room['coords']}")

    # ── 2. 打印节点列表 ────────────────────────────────────────────────────
    print("  去重节点 (编号: [x,y]):")
    for i, v in enumerate(verts):
        print(f"    {i:2d}: {v}  邻居→ {adj[i]}")

    # ── 3. 验证：是否有节点共坐标（重叠节点）─────────────────────────────
    coord_to_idx = {}
    for i, v in enumerate(verts):
        key = tuple(v)
        if key in coord_to_idx:
            print(f"  !! 节点 {i} 与节点 {coord_to_idx[key]} 坐标相同: {v}")
        else:
            coord_to_idx[key] = i

    # ── 4. 验证：邻接矩阵是否对称 ─────────────────────────────────────────
    asym = []
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                asym.append((i, j))
    if asym:
        print(f"  !! 非对称边: {asym}")
    else:
        print("  邻接矩阵对称: OK")

    # ── 5. 验证：是否存在"跨节点"的长边（即 A-B 存在但 A-B 之间有中间节点未切入）
    def point_on_segment(p, u, v):
        px,py=p; ux,uy=u; vx,vy=v
        cross=(vx-ux)*(py-uy)-(vy-uy)*(px-ux)
        if cross!=0: return False
        t_num=(px-ux) if ux!=vx else (py-uy)
        t_den=(vx-ux) if ux!=vx else (vy-uy)
        if t_den==0: return False
        return (0<t_num<t_den) if t_den>0 else (t_den<t_num<0)

    long_edges = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            u = verts[i]; v = verts[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if point_on_segment(verts[k], u, v):
                    long_edges.append((i, j, k))
    if long_edges:
        for (i, j, k) in long_edges:
            print(f"  !! 边 {i}-{j} 内部有节点 {k} 未被切入！")
    else:
        print("  所有边均已正确切分: OK")

    # ── 6. 可视化 ──────────────────────────────────────────────────────────
    all_x = [v[0] for v in verts]; all_y = [v[1] for v in verts]
    pad = 10
    xlim = (min(all_x)-pad, max(all_x)+pad)
    ylim = (max(all_y)+pad, min(all_y)-pad)

    fig, (ax_poly, ax_node) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：原始多边形，顶点标坐标
    for room in rooms:
        color = ROOM_COLORS.get(room["type"], "#CCC")
        pts   = room["coords"]
        xs    = [c[0] for c in pts] + [pts[0][0]]
        ys    = [c[1] for c in pts] + [pts[0][1]]
        ax_poly.fill(xs, ys, color=color, alpha=0.5)
        ax_poly.plot(xs, ys, "k-", lw=1.2)
        for c in pts:
            ax_poly.plot(c[0], c[1], "ko", ms=4)
            ax_poly.text(c[0]+1, c[1]-1, f"({c[0]},{c[1]})",
                         fontsize=5, color="navy")
        cx = sum(c[0] for c in pts)/len(pts)
        cy = sum(c[1] for c in pts)/len(pts)
        ax_poly.text(cx, cy, room["type"], ha="center", va="center",
                     fontsize=6, fontweight="bold")
    ax_poly.set_title("原始多边形（顶点坐标）", fontsize=10)
    ax_poly.set_xlim(*xlim); ax_poly.set_ylim(*ylim)
    ax_poly.set_aspect("equal")

    # 右图：节点图，节点标编号+坐标，边标序号
    xs_n = [v[0] for v in verts]
    ys_n = [v[1] for v in verts]
    edge_no = 0
    for i in range(n):
        for j in adj[i]:
            if j > i:
                ax_node.plot([xs_n[i], xs_n[j]], [ys_n[i], ys_n[j]],
                             "b-", lw=1, alpha=0.5, zorder=1)
                # 在边中点标序号
                mx = (xs_n[i] + xs_n[j]) / 2
                my = (ys_n[i] + ys_n[j]) / 2
                ax_node.text(mx, my, str(edge_no),
                             fontsize=5, color="darkblue",
                             ha="center", va="center",
                             bbox=dict(fc="white", ec="none", pad=0.5),
                             zorder=2)
                edge_no += 1
    ax_node.scatter(xs_n, ys_n, c="tomato", s=60, zorder=3)
    for i in range(n):
        ax_node.text(xs_n[i]+1, ys_n[i]-2,
                     f"{i}\n({verts[i][0]},{verts[i][1]})",
                     fontsize=5, color="black", zorder=4)
    print(f"  总边数: {edge_no}")
    ax_node.set_title(f"节点图（编号+坐标，n={n}）", fontsize=10)
    ax_node.set_xlim(*xlim); ax_node.set_ylim(*ylim)
    ax_node.set_aspect("equal")

    fig.suptitle(f"[{idx+1}] {r['prompt']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(VIZ / f"verify_{idx+1:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\n可视化图像已保存到 {VIZ}/")
