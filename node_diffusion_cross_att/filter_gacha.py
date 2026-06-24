"""
从 visualize_gacha.py 输出的 result_input.jsonl 中筛选合格渲染结果。

质检标准：
  1. 图连通（所有有效节点在同一连通分量）
  2. 欧拉公式：find_faces 返回的面数 == E - V + 1（无缺失面/空环）

输出：
  - 重新渲染的图（合格列正常渲染，不合格列打叉）
  - 只含合格行的新 JSONL，供 IoU 评估使用

Usage (from project root):
    python -m node_diffusion_cross_att.filter_gacha \\
        --jsonl  outputs/visualize_gacha/result_input.jsonl \\
        --vocab  node_diffusion_cross_att/type_combo_vocab_old.json \\
        --out    outputs/visualize_gacha_filtered
"""

import argparse
import json
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

from .render import load_vocab, find_faces, vote_room_type, ROOM_COLORS, ROOM_LABELS

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize':   7,
    'font.size':        7,
})


# ── 质检函数 ──────────────────────────────────────────────────────────────────

def is_connected(adj: List[List[float]], n: int) -> bool:
    visited = set()
    q = deque([0])
    while q:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in range(n):
            if v != u and adj[u][v] and v not in visited:
                q.append(v)
    return len(visited) == n


def count_edges(adj: List[List[float]], n: int) -> int:
    return sum(1 for i in range(n) for j in range(i + 1, n) if adj[i][j])


def is_good_render(pred_coords: List, adj_matrix: List, n_nodes: int) -> Tuple[bool, str]:
    """
    返回 (合格, 原因说明)。
    pred_coords: [[x,y], ...] 长度 n_nodes
    adj_matrix: 完整 40×40，只用前 n_nodes × n_nodes
    """
    adj = [row[:n_nodes] for row in adj_matrix[:n_nodes]]

    # 1. 连通性
    if not is_connected(adj, n_nodes):
        return False, 'disconnected'

    # 2. 欧拉公式：有界面数应等于 E - V + 1
    E = count_edges(adj, n_nodes)
    expected = E - n_nodes + 1
    if expected <= 0:
        return False, f'degenerate graph (E={E} V={n_nodes})'

    coords_list = [(float(c[0]), float(c[1])) for c in pred_coords]
    adj_list    = [[float(v) for v in row] for row in adj]
    faces = find_faces(coords_list, adj_list)

    if len(faces) < expected:
        return False, f'missing faces ({len(faces)}<{expected})'

    return True, 'ok'


# ── 渲染函数 ──────────────────────────────────────────────────────────────────

def render_to_ax(ax, pred_coords, adj_matrix, n_nodes, node_types):
    coords_list = [(float(pred_coords[i][0]), float(pred_coords[i][1]))
                   for i in range(n_nodes)]
    adj = [row[:n_nodes] for row in adj_matrix[:n_nodes]]
    adj_list = [[float(v) for v in row] for row in adj]

    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj_list[i][j] == 1:
                all_nbrs[i].append(j)

    faces      = find_faces(coords_list, adj_list)
    face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]

    xs = [c[0] for c in coords_list]
    ys = [c[1] for c in coords_list]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span   = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def norm(x, y):
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#F8F8F8')

    for face, rt in zip(faces, face_types):
        pts  = [norm(*coords_list[i]) for i in face]
        poly = MplPolygon(pts, closed=True,
                          facecolor=ROOM_COLORS.get(rt, '#EAEDED'),
                          edgecolor='#555555', linewidth=0.8, alpha=0.88, zorder=1)
        ax.add_patch(poly)
        try:
            rp = ShapelyPolygon(pts).representative_point()
            cx, cy = rp.x, rp.y
        except Exception:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, ROOM_LABELS.get(rt, rt),
                ha='center', va='center', fontsize=5.5, color='#222222', zorder=3)


def mark_invalid_ax(ax, reason: str):
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#F0F0F0')
    ax.plot([0.1, 0.9], [0.1, 0.9], color='#CC3333', linewidth=2, zorder=2)
    ax.plot([0.1, 0.9], [0.9, 0.1], color='#CC3333', linewidth=2, zorder=2)
    ax.text(0.5, 0.15, reason, ha='center', va='center',
            fontsize=5, color='#CC3333', transform=ax.transAxes)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', default='outputs/visualize_gacha/result_input.jsonl')
    p.add_argument('--vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--out',   default='outputs/visualize_gacha_filtered')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_combo = load_vocab(Path(args.vocab))

    # 读取并按 idx 分组
    rows_by_idx = defaultdict(list)
    with open(args.jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_idx[row['idx']].append(row)

    # 按 idx 排序，roll_k 排序
    all_idxs = sorted(rows_by_idx.keys())
    for idx in all_idxs:
        rows_by_idx[idx].sort(key=lambda r: r['roll_k'])

    total_rolls = sum(len(v) for v in rows_by_idx.values())
    good_rolls  = 0
    good_samples = 0

    filtered_jsonl = out_dir / 'result_filtered.jsonl'
    with open(filtered_jsonl, 'w', encoding='utf-8') as jf:
        for idx in all_idxs:
            rolls = rows_by_idx[idx]
            K     = len(rolls)
            n     = rolls[0]['n_nodes']
            text  = rolls[0]['text']

            # ── 质检每个 roll ─────────────────────────────────────────────
            results = []  # (good: bool, reason: str, row: dict)
            for row in rolls:
                good, reason = is_good_render(
                    row['pred_coords'], row['adj_matrix'], row['n_nodes'])
                results.append((good, reason, row))

            n_good = sum(1 for g, _, _ in results if g)
            good_rolls += n_good
            if n_good > 0:
                good_samples += 1

            # ── 渲染：合格列正常渲染，不合格列打叉 ──────────────────────
            COL_W = [2.5] * K
            fig, axes = plt.subplots(
                1, K,
                figsize=(sum(COL_W) + 0.3, 3.2),
                gridspec_kw={'width_ratios': COL_W},
                constrained_layout=True,
            )
            if K == 1:
                axes = [axes]

            for k, (good, reason, row) in enumerate(results):
                ax = axes[k]
                if good:
                    node_types = [id_to_combo.get(int(row['pred_combo_ids'][i]), ['other'])
                                  for i in range(n)]
                    try:
                        render_to_ax(ax, row['pred_coords'], row['adj_matrix'], n, node_types)
                    except Exception as e:
                        mark_invalid_ax(ax, str(e)[:30])
                else:
                    mark_invalid_ax(ax, reason)

                status = '✓' if good else '✗'
                ax.set_title(f'Roll {k + 1} {status}', fontsize=7, pad=3,
                             color='#228B22' if good else '#CC3333')

            wrapped = textwrap.fill(text, width=100)
            fig.suptitle(f"#{idx}  {wrapped}", fontsize=6,
                         ha='left', x=0.01, y=1.01, va='bottom')

            png_out = out_dir / f'{idx:06d}.png'
            fig.savefig(png_out, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # ── 保存合格行到新 JSONL ──────────────────────────────────────
            for good, reason, row in results:
                if good:
                    jf.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'样本数: {len(all_idxs)}')
    print(f'总 roll 数: {total_rolls}')
    print(f'合格 roll 数: {good_rolls}  ({good_rolls/total_rolls*100:.1f}%)')
    print(f'至少有1个合格 roll 的样本数: {good_samples}')
    print(f'\n合格数据 → {filtered_jsonl}')
    print(f'重渲染图 → {out_dir}')


if __name__ == '__main__':
    main()
