"""
检查同一邻接图 G 对应的多个样本，其文本描述是否相同或相似。

输出：
  1. 重复图中，文本完全相同 vs 不同的比例
  2. 文本不同时，平均 token 级 Jaccard 相似度
  3. 可视化：抽样若干重复图，并排显示其文本描述与坐标布局

用法：
    python check_text_similarity.py \
        --data data/jsonl/final_graph_dataset_v3.jsonl \
        --out  outputs/text_similarity_check.png \
        --max_samples 50000 \
        --top 8
"""

import argparse
import json
import os
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 中文字体支持 ─────────────────────────────────────────────────────────────
def _find_cjk_font():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'PingFang SC', 'Hiragino Sans GB',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

_cjk = _find_cjk_font()
if _cjk:
    plt.rcParams['font.family'] = _cjk
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def adj_fingerprint(adj, n):
    rows, cols = np.triu_indices(n, k=1)
    return tuple(adj[rows, cols].astype(np.int8).tolist())


def jaccard(a: str, b: str) -> float:
    """词级 Jaccard 相似度（忽略大小写）"""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def center(coords, n):
    valid = coords[:n]
    return valid - valid.mean(axis=0)


def draw_layout(ax, coords, adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        color='#aaa', lw=0.8, zorder=0)
    ax.scatter(coords[:n, 0], coords[:n, 1],
               s=18, color='steelblue', zorder=2,
               edgecolors='k', linewidths=0.4)
    ax.set_aspect('equal')
    ax.axis('off')


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/jsonl/final_graph_dataset_v3.jsonl')
    parser.add_argument('--out',  default='outputs/text_similarity_check.png')
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--top',  type=int, default=8,
                        help='可视化多少张重复图（文本最不同的优先）')
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

    coords_all, adj_all, n_nodes, prompts = [], [], [], []
    for rec in records:
        n = int(rec['n_nodes'])
        coords = np.array(rec['node_coords'][:n], dtype=np.float32)
        adj    = np.array(rec['adj_matrix'], dtype=np.float32)[:n, :n]
        np.fill_diagonal(adj, 0)
        coords_all.append(coords)
        adj_all.append(adj)
        n_nodes.append(n)
        prompts.append(rec.get('prompt', ''))

    # ── 按邻接图指纹分组 ─────────────────────────────────────────────────────
    groups = defaultdict(list)
    for i in range(M):
        fp = adj_fingerprint(adj_all[i], n_nodes[i])
        groups[fp].append(i)

    multi = {fp: idxs for fp, idxs in groups.items() if len(idxs) >= 2}
    print(f"\n唯一邻接图数量: {len(groups)}")
    print(f"出现 >=2 次的图: {len(multi)} ({100*len(multi)/len(groups):.1f}%)")

    if not multi:
        print("没有重复图，退出。")
        return

    # ── 文本相似度分析 ────────────────────────────────────────────────────────
    n_identical   = 0   # 所有文本完全相同
    n_all_same    = 0   # 所有文本完全相同（组级别）
    n_mix         = 0   # 组内有不同文本
    jacc_scores   = []  # 每对文本的 Jaccard

    # 同时记录"文本差异"最大的组用于可视化
    group_stats = []   # (min_jacc, fp, idxs, n)

    for fp, idxs in multi.items():
        texts = [prompts[i] for i in idxs]
        # 两两计算
        pairs = []
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                j = jaccard(texts[a], texts[b])
                pairs.append(j)
                jacc_scores.append(j)
                if texts[a] == texts[b]:
                    n_identical += 1

        min_j = min(pairs) if pairs else 1.0
        all_same = all(t == texts[0] for t in texts)
        if all_same:
            n_all_same += 1
        else:
            n_mix += 1

        group_stats.append((min_j, fp, idxs, n_nodes[idxs[0]]))

    total_pairs = len(jacc_scores)
    identical_rate = n_identical / total_pairs if total_pairs else 0
    mean_jacc = np.mean(jacc_scores) if jacc_scores else 0

    print(f"\n── 文本相似度分析 ──────────────────────────────────────")
    print(f"总分析对数:          {total_pairs:>8,}")
    print(f"文本完全相同的对:    {n_identical:>8,}  ({100*identical_rate:.1f}%)")
    print(f"  组内文本全相同:    {n_all_same:>8,}  ({100*n_all_same/len(multi):.1f}% 的重复图)")
    print(f"  组内文本有不同:    {n_mix:>8,}  ({100*n_mix/len(multi):.1f}% 的重复图)")
    print(f"平均词级 Jaccard:    {mean_jacc:>8.4f}  (1.0=完全相同, 0.0=完全不同)")

    # Jaccard 分布
    arr = np.array(jacc_scores)
    for thresh in [0.99, 0.95, 0.80, 0.50]:
        pct = 100 * (arr >= thresh).mean()
        print(f"  Jaccard >= {thresh:.2f}: {pct:.1f}%")

    # ── 可视化：文本最不同的重复图 ────────────────────────────────────────────
    # 按 min_jacc 升序（差异最大的在前）
    group_stats.sort(key=lambda x: x[0])
    top_groups = group_stats[:args.top]

    MAX_SHOW = 3   # 每组最多显示 3 个样本
    fig_rows  = len(top_groups)
    # 每行: 1 列坐标布局 x MAX_SHOW + 1 列文本对比
    TEXT_COL  = MAX_SHOW
    fig_cols  = MAX_SHOW + 1

    fig, axes = plt.subplots(
        fig_rows, fig_cols,
        figsize=(fig_cols * 3.5, fig_rows * 3.0),
        squeeze=False,
    )
    fig.suptitle("同一邻接图 → 文本差异最大的重复图（左=布局, 右=文本描述）",
                 fontsize=10, y=1.01)

    for row_i, (min_j, fp, idxs, n) in enumerate(top_groups):
        show_idxs = idxs[:MAX_SHOW]

        # 左侧: 坐标布局
        for col_i, idx in enumerate(show_idxs):
            ax = axes[row_i][col_i]
            c  = center(coords_all[idx], n)
            draw_layout(ax, c, adj_all[idx], n)
            ax.set_title(f"样本#{idx}", fontsize=7)

        # 填充空列
        for col_i in range(len(show_idxs), MAX_SHOW):
            axes[row_i][col_i].axis('off')

        # 右侧: 文本对比（垂直排列）
        ax_text = axes[row_i][TEXT_COL]
        ax_text.axis('off')
        lines = [f"n={n}  重复={len(idxs)}次  min_Jaccard={min_j:.3f}\n"]
        for k, idx in enumerate(show_idxs):
            txt = prompts[idx]
            wrapped = textwrap.fill(txt, width=55)
            lines.append(f"[{k}] {wrapped}\n")
        font_kw = {'family': _cjk} if _cjk else {}
        ax_text.text(0.0, 1.0, "\n".join(lines),
                     transform=ax_text.transAxes,
                     va='top', ha='left',
                     fontsize=6, wrap=True,
                     **font_kw)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存: {args.out}")

    # ── 随机抽样：打印几组原始文本供肉眼确认 ────────────────────────────────
    print("\n── 抽样原始文本对比（文本差异最大的前5组）────────────────")
    for min_j, fp, idxs, n in group_stats[:5]:
        print(f"\n  [n={n}, 重复{len(idxs)}次, min_Jaccard={min_j:.3f}]")
        for idx in idxs[:3]:
            print(f"    #{idx}: {prompts[idx][:120]}")


if __name__ == '__main__':
    main()
