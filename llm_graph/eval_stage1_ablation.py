"""
Stage1 约束消融评估脚本。
对每个消融变体生成 N 条样本，统计：
  - 图合法率    (valid_rate)       序列能被正确解析
  - 树合法率    (tree_valid_rate)  parents 满足 p_k < k
  - 无三角率    (triangle_free)    生成图中无3-cycle
  - 最小度≥2率  (min_deg2_rate)    所有节点度 >= 2
  - 平均GED     (avg_ged)          与真实图的图编辑距离

用法：
  python -m llm_graph.eval_stage1_ablation \
      --ckpt checkpoints/llm_graph/stage2/20260603_045254/best.pt \
      --n-samples 200
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from .infer_stage1 import (
    generate, load_model, load_dataset,
    get_prefix_and_gt, parse_sequence, node_degrees,
    has_triangle, VOCAB_SIZE,
)

# ── 消融变体定义 ──────────────────────────────────────────────────────────────
VARIANTS = [
    ('Full',       dict(use_c1=True,  use_c2=True,  use_c3=True,  use_c4=True)),
    ('w/o C1',     dict(use_c1=False, use_c2=True,  use_c3=True,  use_c4=True)),
    ('w/o C2',     dict(use_c1=True,  use_c2=False, use_c3=True,  use_c4=True)),
    ('w/o C3',     dict(use_c1=True,  use_c2=True,  use_c3=False, use_c4=True)),
    ('w/o C4',     dict(use_c1=True,  use_c2=True,  use_c3=True,  use_c4=False)),
    ('w/o All',    dict(use_c1=False, use_c2=False, use_c3=False, use_c4=False)),
]


# ── 图属性检测 ────────────────────────────────────────────────────────────────

def check_tree_valid(parents: list) -> bool:
    """parents[k] 是节点 k+1 的父节点，合法要求 parents[k] <= k。"""
    for k, p in enumerate(parents):
        if p > k:
            return False
    return True


def check_triangle_free(adj: list) -> bool:
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                if has_triangle(adj, u, v):
                    return False
    return True


def check_min_deg2(adj: list) -> bool:
    return all(d >= 2 for d in node_degrees(adj))


def graph_edit_distance(adj_gen: list, adj_gt: list) -> int:
    """
    简化版 GED：只算边集的对称差（edge insertion/deletion 代价各为1）。
    若节点数不同，不同节点数部分的边全算缺失。
    """
    n_gen = len(adj_gen)
    n_gt  = len(adj_gt)
    n     = max(n_gen, n_gt)

    def edges(adj, size):
        return {(i, j) for i in range(size) for j in range(i + 1, size) if adj[i][j]}

    e_gen = edges(adj_gen, n_gen)
    e_gt  = edges(adj_gt,  n_gt)
    # 节点数不同时，多出的节点对应的所有可能边视作差异
    return len(e_gen.symmetric_difference(e_gt))


# ── 主评估循环 ────────────────────────────────────────────────────────────────

def evaluate_variant(
    model, all_tokens, all_lengths, all_textlens,
    indices, device, temperature, constraints: dict
) -> dict:
    stats = defaultdict(list)

    for idx in indices:
        prefix, gt_seq = get_prefix_and_gt(idx, all_tokens, all_lengths, all_textlens)
        gt = parse_sequence(gt_seq)

        gen_seq = generate(model, prefix, device,
                           max_new_tokens=200, temperature=temperature,
                           **constraints)
        gen = parse_sequence(gen_seq)

        stats['valid'].append(int(gen['valid']))

        if gen['valid']:
            stats['tree_valid'].append(int(check_tree_valid(gen['parents'])))
            stats['triangle_free'].append(int(check_triangle_free(gen['adj'])))
            stats['min_deg2'].append(int(check_min_deg2(gen['adj'])))
            if gt['valid']:
                ged = graph_edit_distance(gen['adj'], gt['adj'])
                stats['ged'].append(ged)
        else:
            stats['tree_valid'].append(0)
            stats['triangle_free'].append(0)
            stats['min_deg2'].append(0)

    def mean(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    return {
        'valid_rate':     mean(stats['valid']),
        'tree_valid':     mean(stats['tree_valid']),
        'triangle_free':  mean(stats['triangle_free']),
        'min_deg2_rate':  mean(stats['min_deg2']),
        'avg_ged':        mean(stats['ged']),
        'n':              len(indices),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260603_045254/best.pt')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree.npz')
    p.add_argument('--n-samples',   type=int, default=200,
                   help='每个变体评估的样本数（默认200）')
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
    header = f"{'变体':<12} {'合法率':>8} {'树合法':>8} {'无三角':>8} {'度≥2':>8} {'GED':>8}"
    print(f'\n{header}')
    print('─' * 60)

    for name, constraints in VARIANTS:
        print(f'  评估: {name} ...', flush=True)
        result = evaluate_variant(
            model, all_tokens, all_lengths, all_textlens,
            indices, device, args.temperature, constraints,
        )
        all_results[name] = result

        row = (f"{name:<12}"
               f"  {result['valid_rate']:>6.1%}"
               f"  {result['tree_valid']:>6.1%}"
               f"  {result['triangle_free']:>6.1%}"
               f"  {result['min_deg2_rate']:>6.1%}"
               f"  {result['avg_ged']:>6.2f}")
        print(row)

    print('─' * 60)
    print(f'\n结果保存至 {args.out}')
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
