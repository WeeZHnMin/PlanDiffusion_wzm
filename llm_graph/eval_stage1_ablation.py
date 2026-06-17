"""
Stage1 推理约束消融评估脚本。

对应论文表 tab:ablation_constraints：逐步启用 C1→C5，统计：
  - GVR (Graph Validity Rate)：生成图连通且无孤立节点的样本占比
  - 无三角率 (Triangle-free)：生成图中无3-cycle 的比例
  - 最小度≥2率 (Min-deg≥2)：所有节点度 >= 2 的样本占比

用法：
  python -m llm_graph.eval_stage1_ablation \
      --ckpt checkpoints/llm_graph/stage2/20260603_045254/best.pt \
      --n-samples 1000
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from .infer_stage1 import (
    generate, load_model, load_dataset,
    get_prefix_and_gt, parse_sequence, node_degrees,
    has_triangle,
)

# ── 消融变体定义（逐步加入 C1→C5）────────────────────────────────────────────
VARIANTS = [
    ('无约束',          dict(use_c1=False, use_c2=False, use_c3=False, use_c4=False, use_c5=False)),
    ('+C1',             dict(use_c1=True,  use_c2=False, use_c3=False, use_c4=False, use_c5=False)),
    ('+C1+C2',          dict(use_c1=True,  use_c2=True,  use_c3=False, use_c4=False, use_c5=False)),
    ('+C1+C2+C3',       dict(use_c1=True,  use_c2=True,  use_c3=True,  use_c4=False, use_c5=False)),
    ('+C1+C2+C3+C4',    dict(use_c1=True,  use_c2=True,  use_c3=True,  use_c4=True,  use_c5=False)),
    ('+C1+C2+C3+C4+C5', dict(use_c1=True,  use_c2=True,  use_c3=True,  use_c4=True,  use_c5=True)),
]


# ── 图属性检测 ────────────────────────────────────────────────────────────────

def check_connected(adj: list) -> bool:
    """BFS 判断图是否连通。"""
    n = len(adj)
    if n == 0:
        return False
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        u = queue.pop()
        for v in range(n):
            if adj[u][v] and not visited[v]:
                visited[v] = True
                queue.append(v)
    return all(visited)


def check_gvr(adj: list) -> bool:
    """GVR：连通 且 无孤立节点（度 >= 1）。"""
    if not check_connected(adj):
        return False
    return all(d >= 1 for d in node_degrees(adj))


def check_triangle_free(adj: list) -> bool:
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v] and has_triangle(adj, u, v):
                return False
    return True


def check_min_deg2(adj: list) -> bool:
    return all(d >= 2 for d in node_degrees(adj))


# ── 主评估循环 ────────────────────────────────────────────────────────────────

def evaluate_variant(
    model, all_tokens, all_lengths, all_textlens,
    indices, device, temperature, constraints: dict
) -> dict:
    stats = defaultdict(list)

    for idx in indices:
        prefix, gt_seq = get_prefix_and_gt(idx, all_tokens, all_lengths, all_textlens)
        gen_seq = generate(model, prefix, device,
                           max_new_tokens=200, temperature=temperature,
                           **constraints)
        gen = parse_sequence(gen_seq)

        if gen['valid']:
            adj = gen['adj']
            stats['gvr'].append(int(check_gvr(adj)))
            stats['triangle_free'].append(int(check_triangle_free(adj)))
            stats['min_deg2'].append(int(check_min_deg2(adj)))
        else:
            stats['gvr'].append(0)
            stats['triangle_free'].append(0)
            stats['min_deg2'].append(0)

    def mean(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    return {
        'gvr':           mean(stats['gvr']),
        'triangle_free': mean(stats['triangle_free']),
        'min_deg2_rate': mean(stats['min_deg2']),
        'n':             len(indices),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260603_045254/best.pt')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree_test_10k.npz')
    p.add_argument('--n-samples',   type=int, default=1000,
                   help='每个变体评估的样本数（默认1000）')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--out',         default='llm_graph/ablation_results.json',
                   help='结果保存路径')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)
    all_tokens, all_lengths, all_textlens = load_dataset(args.data)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_tokens), size=args.n_samples, replace=False)

    all_results = {}
    header = f"{'约束配置':<20} {'GVR↑':>8} {'无三角率':>10} {'度≥2率':>8}"
    print(f'\n{header}')
    print('─' * 52)

    for name, constraints in VARIANTS:
        print(f'  评估: {name} ...', flush=True)
        result = evaluate_variant(
            model, all_tokens, all_lengths, all_textlens,
            indices, device, args.temperature, constraints,
        )
        all_results[name] = result
        print(f"  {name:<20} {result['gvr']:>7.1%} {result['triangle_free']:>9.1%} {result['min_deg2_rate']:>7.1%}")

    print('─' * 52)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\n结果保存至 {args.out}')


if __name__ == '__main__':
    main()
