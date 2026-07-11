"""
llm_graph 主评估脚本（测试集）：
  - avg_ged        : 生成图与真实图的平均边集对称差（条件对齐质量）
  - Node-count KL  : 节点数分布 KL 散度  KL(P_real || Q_gen)
  - Degree KL      : 节点度分布 KL 散度  KL(P_real || Q_gen)

用法：
  python -m llm_graph.eval \
      --ckpt checkpoints/llm_graph/stage2/best.pt \
      --data data/jsonl/test_graph_dataset_18k5.jsonl \
      --n_samples 1000
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .infer_stage1 import (
    load_model, parse_sequence, node_degrees,
    encode_text, BOS_ID, generate,
)
from .infer_batch import generate_batch


# ── 图编辑距离（边集对称差）────────────────────────────────────────────────────

def graph_edit_distance(adj_gen: list, adj_gt: list) -> int:
    n_gen = len(adj_gen)
    n_gt  = len(adj_gt)
    def edges(adj, n):
        return {(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]}
    e_gen = edges(adj_gen, n_gen)
    e_gt  = edges(adj_gt,  n_gt)
    return len(e_gen.symmetric_difference(e_gt))


def face_count(adj: list) -> int:
    """内部面数（房间数）= E - N + 1，连通平面图欧拉公式。"""
    n = len(adj)
    e = sum(adj[i][j] for i in range(n) for j in range(i + 1, n))
    return max(0, e - n + 1)


# ── KL 散度 ───────────────────────────────────────────────────────────────────

def kl_divergence(p_counts: Counter, q_counts: Counter, eps: float = 1e-8) -> float:
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

def evaluate(model, rows, vocab, device, temperature=1.0, batch_size=16, one_by_one=False, use_c2=True):
    ged_list        = []
    face_diff_list  = []
    gt_node_counts  = Counter()
    gen_node_counts = Counter()
    gt_degrees      = Counter()
    gen_degrees     = Counter()
    n_invalid_gen   = 0
    n_match_list    = []   # exact
    n_match1_list   = []   # ±1
    n_match2_list   = []   # ±2
    n_match3_list   = []   # ±3
    gen_n_list      = []
    samples_out     = []

    _bs = 1 if one_by_one else batch_size
    for b_start in range(0, len(rows), _bs):
        batch = rows[b_start: b_start + _bs]

        prefixes, gt_list = [], []
        for rec in batch:
            prompt = rec.get('prompt', '').replace('\n', ' ').strip()
            prefix = encode_text(prompt, vocab) + [BOS_ID]
            prefixes.append(prefix)

            n = int(rec['n_nodes'])
            adj_raw = rec['adj_matrix']
            gt_adj  = [list(row[:n]) for row in adj_raw[:n]]
            gt_list.append({
                'source_index': rec.get('_source_index'),
                'n_nodes': n,
                'adj': gt_adj,
                'prompt': prompt,
            })

        if one_by_one:
            gen_seqs = [generate(model, prefixes[0], device,
                                 max_new_tokens=200, temperature=temperature)]
            print(f'  {b_start + 1}/{len(rows)} done ...')
        else:
            gen_seqs = generate_batch(model, prefixes, device,
                                      max_new_tokens=200, temperature=temperature,
                                      use_c2=use_c2)

        for gt, gen_seq in zip(gt_list, gen_seqs):
            gen = parse_sequence(gen_seq, count_actual_n=not use_c2)

            gt_node_counts[gt['n_nodes']] += 1
            for d in node_degrees(gt['adj']):
                gt_degrees[d] += 1

            sample = {
                'source_index': gt['source_index'],
                'prompt':      gt['prompt'],
                'gt_n_nodes':  gt['n_nodes'],
                'gt_adj':      gt['adj'],
                'gen_tokens':  gen_seq,
                'gen_valid':   gen['valid'],
                'gen_n_nodes': gen['n_nodes'] if gen['valid'] else None,
                'gen_adj':     gen['adj']     if gen['valid'] else None,
                'ged':         None,
            }

            if gen['valid']:
                gen_node_counts[gen['n_nodes']] += 1
                for d in node_degrees(gen['adj']):
                    gen_degrees[d] += 1
                ged = graph_edit_distance(gen['adj'], gt['adj'])
                ged_list.append(ged)
                diff_n = abs(gen['n_nodes'] - gt['n_nodes'])
                n_match_list.append(int(diff_n == 0))
                n_match1_list.append(int(diff_n <= 1))
                n_match2_list.append(int(diff_n <= 2))
                n_match3_list.append(int(diff_n <= 3))
                gen_n_list.append(gen['n_nodes'])
                sample['ged'] = ged
                fd = abs(face_count(gen['adj']) - face_count(gt['adj']))
                face_diff_list.append(fd)
                sample['face_diff'] = fd
                sample['gen_faces'] = face_count(gen['adj'])
                sample['gt_faces']  = face_count(gt['adj'])
            else:
                n_invalid_gen += 1

            samples_out.append(sample)

        if not one_by_one:
            done = min(b_start + _bs, len(rows))
            if done % 100 < _bs or done == len(rows):
                print(f'  {done}/{len(rows)} done ...')

    avg_ged       = float(np.mean(ged_list))       if ged_list       else float('nan')
    avg_face_diff = float(np.mean(face_diff_list)) if face_diff_list else float('nan')
    node_kl       = kl_divergence(gt_node_counts, gen_node_counts)
    degree_kl     = kl_divergence(gt_degrees, gen_degrees)
    n_match_rate  = float(np.mean(n_match_list))  if n_match_list  else float('nan')
    n_match1_rate = float(np.mean(n_match1_list)) if n_match1_list else float('nan')
    n_match2_rate = float(np.mean(n_match2_list)) if n_match2_list else float('nan')
    n_match3_rate = float(np.mean(n_match3_list)) if n_match3_list else float('nan')
    gen_n_std     = float(np.std(gen_n_list))     if gen_n_list    else float('nan')
    gen_n_mean    = float(np.mean(gen_n_list))    if gen_n_list    else float('nan')

    return {
        'n_samples':      len(rows),
        'n_valid_gen':    len(ged_list),
        'n_invalid_gen':  n_invalid_gen,
        'avg_ged':        round(avg_ged, 4),
        'avg_face_diff':  round(avg_face_diff, 4),
        'node_count_kl':  round(node_kl,   4),
        'degree_kl':      round(degree_kl, 4),
        'n_match_rate':   round(n_match_rate,  4),
        'n_match1_rate':  round(n_match1_rate, 4),
        'n_match2_rate':  round(n_match2_rate, 4),
        'n_match3_rate':  round(n_match3_rate, 4),
        'gen_n_mean':     round(gen_n_mean, 2),
        'gen_n_std':      round(gen_n_std,  2),
        'samples':        samples_out,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/best.pt')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_18k5.jsonl')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--n_samples',   type=int,   default=0, help='评估样本数（0=全部）')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--out',         default='outputs/llm_graph_eval.jsonl')
    p.add_argument('--out-theta2',  default='', help='θ₂输入用JSONL路径（空=不输出）')
    p.add_argument('--one-by-one',  action='store_true', help='逐条推理（慢但稳定，默认批量）')
    p.add_argument('--no-c2',       action='store_true', help='关闭 C2：让模型自己决定 SEP 时机，用实际 parent 数推算 N')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)

    print(f'读取数据集: {args.data}')
    rows = []
    with open(args.data, encoding='utf-8') as f:
        for source_index, line in enumerate(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec['_source_index'] = source_index
                rows.append(rec)

    if args.n_samples > 0 and args.n_samples < len(rows):
        rng  = np.random.default_rng(args.seed)
        idxs = rng.choice(len(rows), size=args.n_samples, replace=False)
        rows = [rows[i] for i in idxs]

    print(f'评估样本数: {len(rows)}  batch_size: {args.batch_size}')

    results = evaluate(model, rows, args.vocab, device,
                       temperature=args.temperature,
                       batch_size=args.batch_size,
                       one_by_one=args.one_by_one,
                       use_c2=not args.no_c2)

    print(f'\n{"─" * 40}')
    print(f'  avg_ged        : {results["avg_ged"]}')
    print(f'  avg_face_diff  : {results["avg_face_diff"]}')
    print(f'  Node-count KL  : {results["node_count_kl"]}')
    print(f'  Degree KL      : {results["degree_kl"]}')
    print(f'  valid gen      : {results["n_valid_gen"]}/{results["n_samples"]}')
    print(f'  N match (exact): {results["n_match_rate"]:.1%}')
    print(f'  N match (±1)   : {results["n_match1_rate"]:.1%}')
    print(f'  N match (±2)   : {results["n_match2_rate"]:.1%}')
    print(f'  N match (±3)   : {results["n_match3_rate"]:.1%}')
    print(f'  gen N mean±std : {results["gen_n_mean"]} ± {results["gen_n_std"]}')
    print(f'{"─" * 40}')

    samples = results.pop('samples')

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f'样本结果保存至 {args.out}')

    # θ₂ 输入 JSONL：每条包含 prompt / gen_adj / gen_n_nodes / gt_adj / gt_n_nodes
    if args.out_theta2:
        MAX_N = 40
        Path(args.out_theta2).parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(args.out_theta2, 'w', encoding='utf-8') as f:
            for s in samples:
                if not s['gen_valid']:
                    continue
                n = s['gen_n_nodes']
                adj = s['gen_adj']                          # list[list[int]], n×n
                # 补齐到 MAX_N×MAX_N
                adj_full = [[0] * MAX_N for _ in range(MAX_N)]
                for i in range(n):
                    for j in range(n):
                        adj_full[i][j] = adj[i][j]
                node_mask = [1] * n + [0] * (MAX_N - n)
                rec = {
                    'source_index': s.get('source_index'),
                    'prompt':     s['prompt'],
                    'n_nodes':    n,
                    'adj_matrix': adj_full,
                    'node_mask':  node_mask,
                    'gt_n_nodes': s['gt_n_nodes'],
                    'gt_adj':     s['gt_adj'],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                written += 1
        print(f'θ₂ 输入 JSONL 保存至 {args.out_theta2}  ({written} 条有效)')
    print(f'样本结果保存至 {args.out}')

    # 汇总指标单独保存
    summary_path = args.out.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'汇总指标保存至 {summary_path}')


if __name__ == '__main__':
    main()
