"""
构建 text_graph_align 训练用 NPZ。

使用 BERT tokenizer，输出字段与 node_diffusion_room_tri/build_graph_npz.py 兼容：
  node_coords   (N, MAX_NODES, 2)          float32  节点坐标
  adj_matrix    (N, MAX_NODES, MAX_NODES)  uint8    二值邻接矩阵
  node_mask     (N, MAX_NODES)             uint8    1=有效节点
  prompt_tokens (N, MAX_TEXT_LEN)          int32    BERT token ids
  prompt_mask   (N, MAX_TEXT_LEN)          int32    1=有效 token

支持多个 jsonl 合并（--extra_jsonls），适合把 train + test 数据合并用于对比预训练。

用法：
  # 仅训练集
  python -m text_graph_align.build_npz \
      --jsonl   data/jsonl/final_graph_dataset_v3.jsonl \
      --output  data/processed/text_graph_align/train.npz

  # 训练集 + 额外数据合并
  python -m text_graph_align.build_npz \
      --jsonl        data/jsonl/final_graph_dataset_v3.jsonl \
      --extra_jsonls data/jsonl/test_graph_dataset_30k.jsonl \
      --output       data/processed/text_graph_align/train.npz \
      --workers 8

  # 同时构建验证集
  python -m text_graph_align.build_npz \
      --jsonl        data/jsonl/final_graph_dataset_v3.jsonl \
      --extra_jsonls data/jsonl/test_graph_dataset_30k.jsonl \
      --val_jsonl    data/jsonl/val_graph_dataset_18k5.jsonl \
      --output       data/processed/text_graph_align/train.npz \
      --workers 8
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
MAX_TEXT_LEN = 224  # p99=205，224 覆盖绝大多数，超出直接丢弃


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl',         default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--extra_jsonls',  nargs='*', default=[],
                   help='额外合并进训练集的 jsonl 文件列表')
    p.add_argument('--val_jsonl',     default='', help='验证集 jsonl，留空则跳过')
    p.add_argument('--output',        default='data/processed/text_graph_align/train.npz')
    p.add_argument('--bert',          default='models/bert-base-uncased')
    p.add_argument('--augment',       type=int, default=3,
                   help='节点排列增强倍数（验证集固定=1）')
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--workers',       type=int, default=0)
    p.add_argument('--max_samples',   type=int, default=0,
                   help='每个 jsonl 最多读取条数，0=全量')
    return p.parse_args()


def _permute(adj, coords, n, perm):
    full_perm   = perm + list(range(n, MAX_NODES))
    new_adj     = adj[np.ix_(full_perm, full_perm)]
    new_coords  = coords[full_perm]
    return new_adj, new_coords


def process_jsonl(jsonl_path, tokenizer, max_samples=0, augment=1, rng=None):
    if rng is None:
        rng = random.Random(42)

    coords_list = []
    adj_list    = []
    mask_list   = []
    ptok_list   = []
    pmsk_list   = []

    n_graphs = n_skipped = 0
    t0 = time.perf_counter()

    with open(jsonl_path, encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_samples > 0 and n_graphs >= max_samples:
                break

            rec    = json.loads(line)
            prompt = rec.get('prompt', '').replace('\n', ' ').strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            n = min(int(rec['n_nodes']), MAX_NODES)
            n_graphs += 1

            # 坐标
            raw_coords = rec['node_coords'][:n]
            coords_pad = np.zeros((MAX_NODES, 2), dtype=np.float32)
            coords_pad[:n] = raw_coords

            # 邻接矩阵
            adj_raw = np.array(rec['adj_matrix'], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.uint8)
            adj_pad[:n, :n] = adj_raw.clip(0, 1)

            # node_mask
            mask = np.zeros(MAX_NODES, dtype=np.uint8)
            mask[:n] = 1

            # BERT tokens
            tlen     = len(enc['input_ids'])
            ptok     = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            pmsk     = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            ptok[:tlen] = enc['input_ids']
            pmsk[:tlen] = enc['attention_mask']

            # 节点排列增强
            base_perm = list(range(n))
            perms = [base_perm]
            for _ in range(augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_coords = _permute(adj_pad, coords_pad, n, perm)
                coords_list.append(new_coords)
                adj_list.append(new_adj)
                mask_list.append(mask)
                ptok_list.append(ptok)
                pmsk_list.append(pmsk)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f'  {line_no+1} 行 → {len(adj_list)} 条  ({elapsed:.1f}s)',
                      flush=True)

    total = len(adj_list)
    print(f'  {jsonl_path}: {n_graphs} 张图（跳过 {n_skipped} 条），增强后 {total} 条')
    return coords_list, adj_list, mask_list, ptok_list, pmsk_list


def merge_and_save(all_lists, out_path):
    coords_list, adj_list, mask_list, ptok_list, pmsk_list = all_lists
    arrays = dict(
        node_coords   = np.stack(coords_list, axis=0),
        adj_matrix    = np.stack(adj_list,    axis=0),
        node_mask     = np.stack(mask_list,   axis=0),
        prompt_tokens = np.stack(ptok_list,   axis=0),
        prompt_mask   = np.stack(pmsk_list,   axis=0),
    )
    np.savez_compressed(out_path, **arrays)
    n = len(mask_list)
    avg_n = arrays['node_mask'].sum(axis=1).mean()
    print(f'  保存 {n} 条 -> {out_path}  avg_nodes={avg_n:.1f}')
    return arrays


def main():
    args      = parse_args()
    rng       = random.Random(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    print(f'tokenizer: {args.bert}  vocab_size={tokenizer.vocab_size}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 训练集（主 + 额外合并）──────────────────────────────────────────────────
    all_coords, all_adj, all_mask, all_ptok, all_pmsk = [], [], [], [], []

    jsonls = [args.jsonl] + (args.extra_jsonls or [])
    for jpath in jsonls:
        t0 = time.perf_counter()
        print(f'\n处理: {jpath}  (augment={args.augment})')
        c, a, m, pt, pm = process_jsonl(
            jpath, tokenizer, args.max_samples, args.augment, rng)
        all_coords.extend(c)
        all_adj.extend(a)
        all_mask.extend(m)
        all_ptok.extend(pt)
        all_pmsk.extend(pm)
        print(f'  ({time.perf_counter()-t0:.1f}s)')

    t0 = time.perf_counter()
    print(f'\n合并后共 {len(all_mask)} 条，保存中...')
    merge_and_save((all_coords, all_adj, all_mask, all_ptok, all_pmsk), out_path)
    print(f'训练集完成  ({time.perf_counter()-t0:.1f}s)')

    # ── 验证集 ─────────────────────────────────────────────────────────────────
    if args.val_jsonl:
        val_path = out_path.parent / 'val.npz'
        t0 = time.perf_counter()
        print(f'\n处理验证集: {args.val_jsonl}  (augment=1)')
        c, a, m, pt, pm = process_jsonl(
            args.val_jsonl, tokenizer, 0, augment=1)
        merge_and_save((c, a, m, pt, pm), val_path)
        print(f'验证集完成  ({time.perf_counter()-t0:.1f}s)')


if __name__ == '__main__':
    main()
