"""
构建 TextCondGNN 训练用的 NPZ 文件。

与 build_graph_npz.py 的区别：
  - 保留 node_combo_ids（类型分类标签，扩散 NPZ 不需要但 TextCondGNN 必须有）
  - 默认输出路径独立，避免与扩散模型 NPZ 混用

字段：
  adj_matrix    : (N, 40, 40)  float32
  node_mask     : (N, 40)      float32
  node_combo_ids: (N, 40)      int32    节点类型标签（1-32，0=padding）
  node_coords   : (N, 40, 2)   int32
  prompt_tokens : (N, 192)     int32    BERT input_ids
  prompt_mask   : (N, 192)     int32    BERT attention_mask
  prompt_lens   : (N,)         int32
  n_nodes       : (N,)         int32

用法：
  python -m data.scripts.build_v3.build_type_npz
  python -m data.scripts.build_v3.build_type_npz --augment 8 --jsonl data/jsonl/final_graph_dataset_v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

MAX_NODES    = 40
MAX_TEXT_LEN = 192


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl',   default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--bert',    default='models/bert-base-uncased')
    p.add_argument('--output',  default='data/processed/node_diffusion_cross_att/type_dataset.npz')
    p.add_argument('--augment', type=int, default=12)
    p.add_argument('--seed',    type=int, default=42)
    return p.parse_args()


def permute_graph(adj, combo_ids, coords, n, perm):
    full_perm  = perm + list(range(n, MAX_NODES))
    return (adj[np.ix_(full_perm, full_perm)],
            combo_ids[full_perm],
            coords[full_perm])


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert)

    adj_list    = []
    mask_list   = []
    ids_list    = []
    coords_list = []
    ptok_list   = []
    pmask_list  = []
    plen_list   = []
    nnodes_list = []

    t0 = time.perf_counter()
    n_graphs  = 0
    n_skipped = 0

    with open(args.jsonl, encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rec    = json.loads(line)
            prompt = rec.get('prompt', '').replace('\n', ' ').strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            n = int(rec['n_nodes'])
            n_graphs += 1

            adj_full = np.array(rec['adj_matrix'], dtype=np.int32)
            np.fill_diagonal(adj_full, 0)

            combo_ids = np.array(rec['node_combo_ids'][:MAX_NODES], dtype=np.int32)

            raw_coords = rec['node_coords'][:MAX_NODES]
            coords = np.zeros((MAX_NODES, 2), dtype=np.int32)
            coords[:len(raw_coords)] = raw_coords

            mask = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n] = 1

            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen = len(enc['input_ids'])
            padded[:tlen]   = enc['input_ids']
            attn_msk[:tlen] = enc['attention_mask']

            base_perm = list(range(n))
            perms = [base_perm]
            for _ in range(args.augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_ids, new_coords = permute_graph(
                    adj_full, combo_ids, coords, n, perm)
                adj_list.append(new_adj)
                mask_list.append(mask)
                ids_list.append(new_ids)
                coords_list.append(new_coords)
                ptok_list.append(padded)
                pmask_list.append(attn_msk)
                plen_list.append(tlen)
                nnodes_list.append(n)

            if (line_no + 1) % 10000 == 0:
                print(f'  {line_no+1} 张图 → {len(adj_list)} 条记录  '
                      f'({time.perf_counter()-t0:.1f}s)')

    print(f'\n共 {n_graphs} 张图（跳过 {n_skipped} 条 >{MAX_TEXT_LEN} tokens），'
          f'增强后 {len(adj_list)} 条记录')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        adj_matrix     = np.stack(adj_list),
        node_mask      = np.stack(mask_list),
        node_combo_ids = np.stack(ids_list),
        node_coords    = np.stack(coords_list),
        prompt_tokens  = np.stack(ptok_list),
        prompt_mask    = np.stack(pmask_list),
        prompt_lens    = np.array(plen_list,   dtype=np.int32),
        n_nodes        = np.array(nnodes_list, dtype=np.int32),
    )
    print(f'保存 → {out_path}  ({time.perf_counter()-t0:.1f}s)')

    # 节点类型分布统计
    arr   = np.stack(ids_list)
    valid = arr[np.stack(mask_list).astype(bool)]
    unique, counts = np.unique(valid, return_counts=True)
    print('\n节点类型分布（前10）:')
    for uid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f'  combo_id={uid:2d}  count={cnt}')


if __name__ == '__main__':
    main()
