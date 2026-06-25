"""
构建 node_diffusion_room 训练用 NPZ。

与 build_v3/build_graph_npz.py 结构一致，额外保存：
  room_membership : (N, 40, MAX_ROOMS)  float32，二值，节点-环隶属矩阵

用法：
  python -m node_diffusion_room.build_graph_npz
  python -m node_diffusion_room.build_graph_npz --augment 8 --workers 8
  # 小训练集（5k张图×8增强=40k条）
  python -m node_diffusion_room.build_graph_npz \\
      --max_samples 5000 \\
      --output data/processed/node_diffusion_room/graph_dataset_5k.npz
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",       default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--bert",        default="models/bert-base-uncased")
    p.add_argument("--output",      default="data/processed/node_diffusion_room/graph_dataset.npz")
    p.add_argument("--augment",     type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0,
                   help="最多读取的原始图数量，0=全量")
    return p.parse_args()


def permute_graph(adj, combo_ids, coords, n, perm):
    full_perm  = perm + list(range(n, MAX_NODES))
    new_adj    = adj[np.ix_(full_perm, full_perm)]
    new_ids    = combo_ids[full_perm]
    new_coords = coords[full_perm]
    return new_adj, new_ids, new_coords


def _compute_room_membership(args_tuple):
    """multiprocessing worker：计算单条记录的 room_membership [MAX_NODES, MAX_ROOMS]。"""
    idx, adj_row, mask_row = args_tuple
    n    = int(mask_row.sum())
    full = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    if n >= 3:
        m = _assign_room_membership_single(adj_row[:n, :n].astype(bool), n)
        full[:n, :] = m
    return idx, full


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert)

    mask_list   = []
    ids_list    = []
    coords_list = []
    ptok_list   = []
    pmask_list  = []
    plen_list   = []
    nnodes_list = []
    adj_list    = []

    t0        = time.perf_counter()
    n_graphs  = 0
    n_skipped = 0

    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rec    = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            if args.max_samples > 0 and n_graphs >= args.max_samples:
                break

            n        = int(rec["n_nodes"])
            n_graphs += 1

            adj_full = np.array(rec["adj_matrix"], dtype=np.int32)
            np.fill_diagonal(adj_full, 0)

            combo_ids = np.array(rec["node_combo_ids"][:MAX_NODES], dtype=np.int32)

            raw_coords = rec["node_coords"][:MAX_NODES]
            coords     = np.zeros((MAX_NODES, 2), dtype=np.int32)
            coords[:len(raw_coords)] = raw_coords

            mask     = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n] = 1

            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen     = len(enc['input_ids'])
            padded[:tlen]   = enc['input_ids']
            attn_msk[:tlen] = enc['attention_mask']

            base_perm = list(range(n))
            perms     = [base_perm]
            for _ in range(args.augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_ids, new_coords = permute_graph(
                    adj_full, combo_ids, coords, n, perm)
                mask_list.append(mask)
                ids_list.append(new_ids)
                coords_list.append(new_coords)
                ptok_list.append(padded)
                pmask_list.append(attn_msk)
                plen_list.append(tlen)
                nnodes_list.append(n)
                adj_list.append(new_adj)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 张图 → {len(adj_list)} 条记录  ({elapsed:.1f}s)")

    total = len(adj_list)
    print(f"\n共 {n_graphs} 张图（跳过 {n_skipped} 条），增强后 {total} 条记录")

    # ── 并行计算 room_membership ──────────────────────────────────────────────
    print("计算 room_membership（并行）...")
    adj_arr  = np.stack(adj_list,  axis=0)   # [total, MAX_NODES, MAX_NODES]
    mask_arr = np.stack(mask_list, axis=0)   # [total, MAX_NODES]
    del adj_list, mask_list                  # 释放内存

    n_workers    = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    tasks        = [(i, adj_arr[i], mask_arr[i]) for i in range(total)]
    membership_out = np.zeros((total, MAX_NODES, MAX_ROOMS), dtype=np.float32)

    with Pool(processes=n_workers) as pool:
        for done, (idx, m) in enumerate(
            pool.imap_unordered(_compute_room_membership, tasks, chunksize=256)
        ):
            membership_out[idx] = m
            if (done + 1) % 50000 == 0:
                print(f"  room_membership: {done+1}/{total}", flush=True)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        node_mask        = mask_arr,
        node_combo_ids   = np.stack(ids_list,    axis=0),
        node_coords      = np.stack(coords_list, axis=0),
        prompt_tokens    = np.stack(ptok_list,   axis=0),
        prompt_mask      = np.stack(pmask_list,  axis=0),
        prompt_lens      = np.array(plen_list,   dtype=np.int32),
        n_nodes          = np.array(nnodes_list, dtype=np.int32),
        room_membership  = membership_out,
    )

    elapsed = time.perf_counter() - t0
    print(f"保存 → {out_path}  (耗时 {elapsed:.1f}s)")
    density = membership_out.sum(axis=2)  # [N, 40] 每个节点属于几个环
    print(f"room_membership: 平均每节点属于 {density[mask_arr==1].mean():.2f} 个环")


if __name__ == "__main__":
    main()
