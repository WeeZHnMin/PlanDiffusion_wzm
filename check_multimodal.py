"""
检查多模态假设：同一张邻接图 G 是否对应多种不同的坐标布局

输出：
  1. 数据集里有多少张唯一的图 G
  2. 有多少张图出现了 2+ 次（不同坐标）
  3. 对出现最多次的几张图，打印坐标方差
  4. 可视化：同一张图的不同布局并排显示

用法：
    python check_multimodal.py \
        --data data/jsonl/test_graph_dataset_10k.jsonl \
        --out  outputs/multimodal_check.png \
        --top  5
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def adj_fingerprint(adj, n):
    """只取上三角（不含对角线），转成 tuple 作为哈希 key。"""
    rows, cols = np.triu_indices(n, k=1)
    return tuple(adj[rows, cols].astype(np.int8).tolist())


def center(coords, n):
    valid = coords[:n]
    return valid - valid.mean(axis=0)


def draw_layout(ax, coords, adj, n, title=""):
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        color='#999', lw=0.7, zorder=0)
    ax.scatter(coords[:n, 0], coords[:n, 1],
               s=20, color='steelblue', zorder=2,
               edgecolors='k', linewidths=0.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/jsonl/test_graph_dataset_10k.jsonl')
    parser.add_argument('--out',  default='outputs/multimodal_check.png')
    parser.add_argument('--top',  type=int, default=5, help='可视化出现次数最多的前N张图')
    parser.add_argument('--max_samples', type=int, default=50000, help='最多读取多少条样本')
    args = parser.parse_args()

    print(f"加载: {args.data}")
    records = []
    with open(args.data, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            records.append(json.loads(line))

    M = len(records)
    print(f"读取样本数: {M}")

    coords_all, adj_all, n_nodes = [], [], []
    for rec in records:
        n = int(rec['n_nodes'])
        coords = np.array(rec['node_coords'][:n], dtype=np.float32)
        adj    = np.array(rec['adj_matrix'],      dtype=np.float32)[:n, :n]
        np.fill_diagonal(adj, 0)
        coords_all.append(coords)
        adj_all.append(adj)
        n_nodes.append(n)

    # ── 按邻接图指纹分组 ─────────────────────────────────────────────────────
    groups = defaultdict(list)   # fingerprint → [idx, ...]
    for i in range(M):
        n = n_nodes[i]
        fp = adj_fingerprint(adj_all[i], n)
        groups[fp].append(i)

    total_unique = len(groups)
    multi = {fp: idxs for fp, idxs in groups.items() if len(idxs) >= 2}
    print(f"\n唯一邻接图数量: {total_unique}")
    print(f"出现 ≥2 次的图: {len(multi)} ({100*len(multi)/total_unique:.1f}%)")

    if not multi:
        print("没有重复图，多模态假设不成立（数据集里每张图只出现一次）")
        return

    # ── 对重复出现的图，计算坐标方差 ─────────────────────────────────────────
    variances = []
    for fp, idxs in multi.items():
        n = n_nodes[idxs[0]]
        # 各布局中心归一化后的坐标
        centered = [center(coords_all[i, :n], n) for i in idxs]
        # 所有布局堆叠后计算方差（节点平均）
        stack = np.stack(centered, axis=0)   # [K, n, 2]
        var   = stack.var(axis=0).mean()     # 标量
        variances.append((var, fp, idxs, n))

    variances.sort(key=lambda x: -x[0])

    print(f"\n坐标方差最大的前10张重复图（中心归一化后）：")
    print(f"{'出现次数':>8}  {'节点数':>6}  {'坐标方差':>10}")
    for var, fp, idxs, n in variances[:10]:
        print(f"{len(idxs):>8}  {n:>6}  {var:>10.2f}")

    mean_var = np.mean([v[0] for v in variances])
    print(f"\n所有重复图的平均坐标方差: {mean_var:.2f}")
    print("（方差接近0 → 布局几乎相同；方差大 → 同一图对应多种布局）")

    # ── 可视化 ───────────────────────────────────────────────────────────────
    top_n = min(args.top, len(variances))
    top   = variances[:top_n]
    max_cols = max(min(len(v[2]), 6) for v in top)
    fig, axes = plt.subplots(
        top_n, max_cols,
        figsize=(max_cols * 2.0, top_n * 2.0),
        squeeze=False,
    )
    fig.suptitle("同一邻接图 → 不同坐标布局（中心归一化）", fontsize=9)

    for row_i, (var, fp, idxs, n) in enumerate(top):
        show_idxs = idxs[:max_cols]
        centered  = [center(coords_all[i], n) for i in show_idxs]
        for col_i in range(max_cols):
            ax = axes[row_i][col_i]
            if col_i < len(show_idxs):
                draw_layout(ax, centered[col_i],
                            adj_all[show_idxs[col_i]], n,
                            title=f"样本#{show_idxs[col_i]}")
            else:
                ax.axis('off')
        axes[row_i][0].set_ylabel(
            f"n={n} 出现{len(idxs)}次\nvar={var:.1f}", fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存: {args.out}")


if __name__ == '__main__':
    main()
