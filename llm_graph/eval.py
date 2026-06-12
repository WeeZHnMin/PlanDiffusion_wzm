"""
llm_graph 主评估脚本（测试集）：
  - avg_ged        : 生成图与真实图的平均边集对称差（条件对齐质量）
  - Node-count KL  : 节点数分布 KL 散度  KL(P_real || Q_gen)
  - Degree KL      : 节点度分布 KL 散度  KL(P_real || Q_gen)

用法：
  python -m llm_graph.eval \
      --ckpt checkpoints/llm_graph/stage2/best.pt \
      --n-samples 1000
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch

from .infer_stage1 import (
    generate, load_model, load_dataset,
    get_prefix_and_gt, parse_sequence, node_degrees,
)


# ── 图编辑距离（边集对称差）────────────────────────────────────────────────────

def graph_edit_distance(adj_gen: list, adj_gt: list) -> int:
    def edges(adj, n):
        return {(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]}
    e_gen = edges(adj_gen, len(adj_gen))
    e_gt  = edges(adj_gt,  len(adj_gt))
    return len(e_gen.symmetric_difference(e_gt))


# ── KL 散度 ───────────────────────────────────────────────────────────────────

def kl_divergence(p_counts: Counter, q_counts: Counter, eps: float = 1e-8) -> float:
    """
    KL(P || Q)，P 为真实分布，Q 为生成分布。
    Q 中缺失的值用 eps 平滑，防止 log(0)。
    """
    support = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if p_total == 0 or q_total == 0:
        return float('nan')

    kl = 0.0
    for x in support:
        p = p_counts.get(x, 0) / p_total
        q = max(q_counts.get(x, 0) / q_total, eps)
        if p > 0:
            kl += p * np.log(p / q)
    return float(kl)


# ── 主评估循环 ────────────────────────────────────────────────────────────────

def evaluate(model, all_tokens, all_lengths, all_textlens,
             indices, device, temperature=1.0):
    ged_list = []
    gt_node_counts  = Counter()
    gen_node_counts = Counter()
    gt_degrees      = Counter()
    gen_degrees     = Counter()
    n_invalid_gen   = 0

    for i, idx in enumerate(indices):
        prefix, gt_seq = get_prefix_and_gt(idx, all_tokens, all_lengths, all_textlens)
        gt = parse_sequence(gt_seq)
        if not gt['valid']:
            continue

        gen_seq = generate(model, prefix, device,
                           max_new_tokens=200, temperature=temperature)
        gen = parse_sequence(gen_seq)

        # 真实图分布（始终统计，保证 P 基于完整样本集）
        gt_node_counts[gt['n_nodes']] += 1
        for d in node_degrees(gt['adj']):
            gt_degrees[d] += 1

        if gen['valid']:
            gen_node_counts[gen['n_nodes']] += 1
            for d in node_degrees(gen['adj']):
                gen_degrees[d] += 1
            ged_list.append(graph_edit_distance(gen['adj'], gt['adj']))
        else:
            n_invalid_gen += 1

        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(indices)} done ...')

    avg_ged     = float(np.mean(ged_list)) if ged_list else float('nan')
    node_kl     = kl_divergence(gt_node_counts, gen_node_counts)
    degree_kl   = kl_divergence(gt_degrees, gen_degrees)

    return {
        'n_samples':     len(indices),
        'n_valid_gen':   len(ged_list),
        'n_invalid_gen': n_invalid_gen,
        'avg_ged':       round(avg_ged, 4),
        'node_count_kl': round(node_kl,   4),
        'degree_kl':     round(degree_kl, 4),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/best.pt')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree.npz')
    p.add_argument('--n-samples',   type=int,   default=1000,
                   help='评估样本数（从数据集随机采样）')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--out',         default='llm_graph/eval_results.json')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)
    all_tokens, all_lengths, all_textlens = load_dataset(args.data)

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_tokens), size=args.n_samples, replace=False)

    results = evaluate(model, all_tokens, all_lengths, all_textlens,
                       indices, device, temperature=args.temperature)

    print(f'\n{"─" * 40}')
    print(f'  avg_ged        : {results["avg_ged"]}')
    print(f'  Node-count KL  : {results["node_count_kl"]}')
    print(f'  Degree KL      : {results["degree_kl"]}')
    print(f'  valid gen      : {results["n_valid_gen"]}/{results["n_samples"]}')
    print(f'{"─" * 40}')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'结果保存至 {args.out}')


if __name__ == '__main__':
    main()
