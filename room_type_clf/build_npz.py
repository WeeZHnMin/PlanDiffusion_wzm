"""
构建 room_type_clf 训练用 NPZ。

保存字段：
  node_mask       (N, 40)            uint8
  adj_matrix      (N, 40, 40)        uint8
  room_membership (N, 40, MAX_ROOMS) float32
  prompt_tokens   (N, 192)           int32
  prompt_mask     (N, 192)           int32
  type_labels     (N, 40)            int32   padding 节点=-1

同目录同名 .json 保存 type_vocab。

用法：
  python -m room_type_clf.build_npz
  python -m room_type_clf.build_npz \\
      --jsonl data/jsonl/val_graph_dataset_18k5.jsonl \\
      --output data/processed/room_type_clf/val.npz
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

from .dataset import build_type_vocab, MAX_NODES, MAX_TEXT_LEN
from .model import _assign_room_membership_single, MAX_ROOMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",       default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--val_jsonl",   default="",
                   help="同时构建验证集（共用词表），留空则跳过")
    p.add_argument("--bert",        default="models/bert-base-uncased")
    p.add_argument("--output",      default="data/processed/room_type_clf/train.npz")
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0,
                   help="最多读取原始图数量，0=全量")
    return p.parse_args()


# ── 并行 worker ───────────────────────────────────────────────────────────────

def _compute_room_membership(args_tuple):
    idx, adj_row, n = args_tuple
    full = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    if n >= 3:
        m = _assign_room_membership_single(adj_row[:n, :n].astype(bool), n)
        full[:n, :] = m
    return idx, full


# ── 处理单个 jsonl ─────────────────────────────────────────────────────────────

def process_jsonl(jsonl_path, tokenizer, type_vocab, max_samples=0, n_workers=1):
    other_id = type_vocab.get('__other__', len(type_vocab) - 1)

    mask_list   = []
    adj_list    = []
    ptok_list   = []
    pmask_list  = []
    labels_list = []
    n_nodes_list = []

    n_graphs = n_skipped = 0
    t0 = time.perf_counter()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_samples > 0 and n_graphs >= max_samples:
                break

            rec    = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            n = min(int(rec["n_nodes"]), MAX_NODES)
            n_graphs += 1

            # 邻接矩阵
            adj_raw = np.array(rec["adj_matrix"], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.uint8)
            adj_pad[:n, :n] = adj_raw.clip(0, 1)

            # node_mask
            mask = np.zeros(MAX_NODES, dtype=np.uint8)
            mask[:n] = 1

            # 文本 token
            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen = len(enc['input_ids'])
            padded[:tlen]   = enc['input_ids']
            attn_msk[:tlen] = enc['attention_mask']

            # 节点类型标签
            raw_types = rec.get("node_types", [])
            labels    = np.full(MAX_NODES, -1, dtype=np.int32)
            for i in range(n):
                t = raw_types[i] if i < len(raw_types) else '__other__'
                if isinstance(t, list):
                    t = t[0] if t else '__other__'
                t = str(t)
                labels[i] = type_vocab.get(t, other_id)

            mask_list.append(mask)
            adj_list.append(adj_pad)
            ptok_list.append(padded)
            pmask_list.append(attn_msk)
            labels_list.append(labels)
            n_nodes_list.append(n)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 行 → {n_graphs} 条记录  ({elapsed:.1f}s)")

    total = len(adj_list)
    print(f"共 {n_graphs} 条（跳过 {n_skipped} 条文本过长）")

    # 并行计算 room_membership
    print("计算 room_membership（并行）...")
    adj_arr  = np.stack(adj_list, axis=0)
    mask_arr = np.stack(mask_list, axis=0)

    tasks     = [(i, adj_arr[i], int(n_nodes_list[i])) for i in range(total)]
    membership_out = np.zeros((total, MAX_NODES, MAX_ROOMS), dtype=np.float32)

    with Pool(processes=n_workers) as pool:
        for done, (idx, m) in enumerate(
            pool.imap_unordered(_compute_room_membership, tasks, chunksize=256)
        ):
            membership_out[idx] = m
            if (done + 1) % 50000 == 0:
                print(f"  room_membership: {done+1}/{total}", flush=True)

    return dict(
        node_mask       = mask_arr,
        adj_matrix      = adj_arr,
        room_membership = membership_out,
        prompt_tokens   = np.stack(ptok_list,   axis=0),
        prompt_mask     = np.stack(pmask_list,  axis=0),
        type_labels     = np.stack(labels_list, axis=0),
    )


def main():
    args = parse_args()
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    tokenizer = BertTokenizer.from_pretrained(args.bert)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 词表：先扫描训练集，再合并验证集里出现的新类型
    print("构建类型词表...")
    from .dataset import build_type_vocab as _build
    type_vocab = _build(args.jsonl)
    if args.val_jsonl:
        for k in _build(args.val_jsonl):
            if k not in type_vocab:
                type_vocab[k] = len(type_vocab)
    print(f"词表大小: {len(type_vocab)}")

    vocab_path = out_path.with_suffix('.vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(type_vocab, f, ensure_ascii=False, indent=2)
    print(f"词表 -> {vocab_path}")

    # 构建训练集
    t0 = time.perf_counter()
    print(f"\n处理训练集: {args.jsonl}")
    arrays = process_jsonl(args.jsonl, tokenizer, type_vocab, args.max_samples, n_workers)
    np.savez_compressed(out_path, **arrays)
    print(f"训练集 -> {out_path}  ({time.perf_counter()-t0:.1f}s)")
    _print_stats(arrays)

    # 构建验证集
    if args.val_jsonl:
        val_path = out_path.parent / (out_path.stem.replace('train', 'val') + '.npz')
        if val_path == out_path:
            val_path = out_path.parent / 'val.npz'
        t0 = time.perf_counter()
        print(f"\n处理验证集: {args.val_jsonl}")
        val_arrays = process_jsonl(args.val_jsonl, tokenizer, type_vocab, 0, n_workers)
        np.savez_compressed(val_path, **val_arrays)
        print(f"验证集 -> {val_path}  ({time.perf_counter()-t0:.1f}s)")
        _print_stats(val_arrays)


def _print_stats(arrays):
    mask  = arrays['node_mask']          # [N, 40]
    valid = mask.astype(bool)
    mb    = arrays['room_membership']    # [N, 40, MAX_ROOMS]
    density = mb.sum(axis=2)[valid]
    labels  = arrays['type_labels']
    valid_labels = labels[valid]
    unique, counts = np.unique(valid_labels, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    print(f"  样本数: {len(mask)}")
    print(f"  平均节点数: {valid.sum(axis=1).mean():.1f}")
    print(f"  平均每节点属于 {density.mean():.2f} 个环")
    print(f"  top5 type_id: {[(int(uid), int(cnt)) for cnt, uid in top5]}")


if __name__ == "__main__":
    main()
