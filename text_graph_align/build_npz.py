"""
构建 text_graph_align 训练用 NPZ。

只用 text_graph_align/vocab 词表做 tokenize，不需要 BERT。
增强样本共用同一条文本，去重后存储：
  text_input_ids  (n_unique, T)        int32   自定义词表 token IDs
  text_attn_mask  (n_unique, T)        bool    1=有效 token
  text_idx        (N,)                 int32   每条样本对应的 unique 索引

图结构字段：
  node_mask       (N, 40)              uint8
  adj_matrix      (N, 40, 40)          uint8
  room_membership (N, 40, MAX_ROOMS)   float32

用法：
  python -m text_graph_align.build_npz \\
      --jsonl     data/jsonl/final_graph_dataset_v3.jsonl \\
      --val_jsonl data/jsonl/val_graph_dataset_18k5.jsonl \\
      --augment 3 --output data/processed/align/train.npz
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

from .model import _assign_room_membership_single, MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192
VOCAB_PATH   = str(Path(__file__).parent / 'vocab')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl',       default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--val_jsonl',   default='', help='同时构建验证集，留空则跳过')
    p.add_argument('--output',      default='data/processed/align/train.npz')
    p.add_argument('--augment',     type=int, default=3, help='节点排列增强倍数，验证集固定=1')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--workers',     type=int, default=0)
    p.add_argument('--max_samples', type=int, default=0)
    return p.parse_args()


# ── 节点排列增强 ───────────────────────────────────────────────────────────────

def _permute_adj(adj, n, perm):
    full_perm = perm + list(range(n, MAX_NODES))
    return adj[np.ix_(full_perm, full_perm)]


# ── room_membership 并行 worker ───────────────────────────────────────────────

def _compute_room_membership(args_tuple):
    idx, adj_row, n = args_tuple
    full = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    if n >= 3:
        m = _assign_room_membership_single(adj_row[:n, :n].astype(bool), n)
        full[:n, :] = m
    return idx, full


# ── 处理单个 jsonl ─────────────────────────────────────────────────────────────

def process_jsonl(jsonl_path, tokenizer, max_samples=0, n_workers=1,
                  augment=1, rng=None):
    if rng is None:
        rng = random.Random(42)

    mask_list     = []
    adj_list      = []
    n_nodes_list  = []
    text_idx_list = []

    prompt_to_uid    = {}
    unique_ids_list  = []
    unique_mask_list = []

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

            # 唯一文本去重
            if prompt not in prompt_to_uid:
                uid      = len(unique_ids_list)
                prompt_to_uid[prompt] = uid
                ids      = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
                attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
                tlen     = len(enc['input_ids'])
                ids[:tlen]      = enc['input_ids']
                attn_msk[:tlen] = enc['attention_mask']
                unique_ids_list.append(ids)
                unique_mask_list.append(attn_msk)
            uid = prompt_to_uid[prompt]

            # 邻接矩阵
            adj_raw = np.array(rec['adj_matrix'], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.uint8)
            adj_pad[:n, :n] = adj_raw.clip(0, 1)

            # node_mask
            mask = np.zeros(MAX_NODES, dtype=np.uint8)
            mask[:n] = 1

            # 节点排列增强
            base_perm = list(range(n))
            perms = [base_perm]
            for _ in range(augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj = _permute_adj(adj_pad, n, perm)
                mask_list.append(mask)
                adj_list.append(new_adj)
                n_nodes_list.append(n)
                text_idx_list.append(uid)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f'  {line_no+1} 行 → {len(adj_list)} 条  '
                      f'unique_prompts={len(unique_ids_list)}  ({elapsed:.1f}s)')

    total = len(adj_list)
    print(f'共 {n_graphs} 张图（跳过 {n_skipped} 条），增强后 {total} 条  '
          f'unique_prompts={len(unique_ids_list)}')

    # 并行计算 room_membership
    print('计算 room_membership...')
    adj_arr  = np.stack(adj_list,  axis=0)
    mask_arr = np.stack(mask_list, axis=0)
    tasks    = [(i, adj_arr[i], int(n_nodes_list[i])) for i in range(total)]
    membership_out = np.zeros((total, MAX_NODES, MAX_ROOMS), dtype=np.float32)
    with Pool(processes=n_workers) as pool:
        for done, (idx, m) in enumerate(
            pool.imap_unordered(_compute_room_membership, tasks, chunksize=256)
        ):
            membership_out[idx] = m
            if (done + 1) % 50000 == 0:
                print(f'  room_membership: {done+1}/{total}', flush=True)

    arrays = dict(
        node_mask       = mask_arr,
        adj_matrix      = adj_arr,
        room_membership = membership_out,
        text_idx        = np.array(text_idx_list, dtype=np.int32),
        text_input_ids  = np.stack(unique_ids_list,  axis=0).astype(np.int32),
        text_attn_mask  = np.stack(unique_mask_list, axis=0).astype(bool),
    )
    return arrays


# ── main ──────────────────────────────────────────────────────────────────────

def _print_stats(arrays):
    mask  = arrays['node_mask']
    valid = mask.astype(bool)
    mb    = arrays['room_membership']
    density = mb.sum(axis=2)[valid]
    n_unique = len(arrays['text_input_ids'])
    print(f'  样本数: {len(mask)}  unique_prompts: {n_unique}')
    print(f'  平均节点数: {valid.sum(axis=1).mean():.1f}')
    print(f'  平均每节点属于 {density.mean():.2f} 个环')


def main():
    args      = parse_args()
    rng       = random.Random(args.seed)
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)
    print(f'tokenizer: {VOCAB_PATH}  vocab_size={tokenizer.vocab_size}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 训练集 ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f'\n处理训练集: {args.jsonl}  (augment={args.augment})')
    arrays = process_jsonl(args.jsonl, tokenizer, args.max_samples,
                           n_workers, args.augment, rng)
    np.savez_compressed(out_path, **arrays)
    print(f'训练集 -> {out_path}  ({time.perf_counter()-t0:.1f}s)')
    _print_stats(arrays)

    # ── 验证集 ────────────────────────────────────────────────────────────────
    if args.val_jsonl:
        val_path = out_path.parent / (out_path.stem.replace('train', 'val') + '.npz')
        if val_path == out_path:
            val_path = out_path.parent / 'val.npz'
        t0 = time.perf_counter()
        print(f'\n处理验证集: {args.val_jsonl}  (augment=1)')
        val_arrays = process_jsonl(args.val_jsonl, tokenizer, 0, n_workers,
                                   augment=1)
        np.savez_compressed(val_path, **val_arrays)
        print(f'验证集 -> {val_path}  ({time.perf_counter()-t0:.1f}s)')
        _print_stats(val_arrays)


if __name__ == '__main__':
    main()
