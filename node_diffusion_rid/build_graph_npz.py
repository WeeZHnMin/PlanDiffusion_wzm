"""
构建 node_diffusion_rid 训练用 NPZ，与 build_v3/build_graph_npz.py 完全一致，
额外在保存时附带每个样本的 room_ids（从邻接矩阵拓扑计算的房间编号）。

输出：
  data/processed/node_diffusion_rid/graph_dataset.npz

字段（与 node_diffusion_cross_att 相同，加一个 room_ids）：
  adj_matrix    : (N, 40, 40)  int32
  node_mask     : (N, 40)      int32
  node_combo_ids: (N, 40)      int32
  node_coords   : (N, 40, 2)   int32
  prompt_tokens : (N, 192)     int32   BERT input_ids
  prompt_mask   : (N, 192)     int32   BERT attention_mask
  prompt_lens   : (N,)         int32
  n_nodes       : (N,)         int32
  room_ids      : (N, 40)      int32   0=无房间，1..K=房间编号

用法：
  python -m node_diffusion_rid.build_graph_npz
  python -m node_diffusion_rid.build_graph_npz --augment 8 --workers 8
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

from .model import _assign_room_ids_single, MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",    default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--bert",     default="models/bert-base-uncased")
    p.add_argument("--output",   default="data/processed/node_diffusion_rid/graph_dataset.npz")
    p.add_argument("--augment",  type=int, default=8,
                   help="每张图随机节点重排次数（1=不增强）")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--workers",  type=int, default=0,
                   help="room_ids 并行计算的 worker 数，0=cpu_count()-1")
    return p.parse_args()


def permute_graph(adj, combo_ids, coords, n, perm):
    full_perm  = perm + list(range(n, MAX_NODES))
    new_adj    = adj[np.ix_(full_perm, full_perm)]
    new_ids    = combo_ids[full_perm]
    new_coords = coords[full_perm]
    return new_adj, new_ids, new_coords


def _compute_room_ids(args_tuple):
    """multiprocessing worker：计算单个样本的 room_ids。"""
    idx, adj_row, mask_row = args_tuple
    n = int(mask_row.sum())
    full = [0] * MAX_NODES
    if n >= 3:
        ids = _assign_room_ids_single(adj_row[:n, :n].astype(bool), n)
        full[:n] = ids
    return idx, full


def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
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

    t0         = time.perf_counter()
    n_graphs   = 0
    n_skipped  = 0

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

            n        = int(rec["n_nodes"])
            n_graphs += 1

            adj_full = np.array(rec["adj_matrix"], dtype=np.int32)
            np.fill_diagonal(adj_full, 0)

            raw_ids   = rec["node_combo_ids"]
            combo_ids = np.array(raw_ids[:MAX_NODES], dtype=np.int32)

            raw_coords = rec["node_coords"][:MAX_NODES]
            coords     = np.zeros((MAX_NODES, 2), dtype=np.int32)
            coords[:len(raw_coords)] = raw_coords

            mask        = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n]    = 1

            padded      = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk    = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen        = len(enc['input_ids'])
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
                adj_list.append(new_adj)
                mask_list.append(mask)
                ids_list.append(new_ids)
                coords_list.append(new_coords)
                ptok_list.append(padded)
                pmask_list.append(attn_msk)
                plen_list.append(tlen)
                nnodes_list.append(n)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 张图 → {len(adj_list)} 条记录  ({elapsed:.1f}s)")

    total = len(adj_list)
    print(f"\n共 {n_graphs} 张图（跳过 {n_skipped} 条），增强后 {total} 条记录")

    # ── 并行计算 room_ids ─────────────────────────────────────────────────────
    print("计算 room_ids（并行）...")
    adj_arr  = np.stack(adj_list,  axis=0)   # [N, 40, 40]
    mask_arr = np.stack(mask_list, axis=0)   # [N, 40]

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    tasks     = [(i, adj_arr[i], mask_arr[i]) for i in range(total)]
    room_ids_out = np.zeros((total, MAX_NODES), dtype=np.int32)

    with Pool(processes=n_workers) as pool:
        for done, (idx, ids) in enumerate(
            pool.imap_unordered(_compute_room_ids, tasks, chunksize=256)
        ):
            room_ids_out[idx] = ids
            if (done + 1) % 50000 == 0:
                print(f"  room_ids: {done+1}/{total}", flush=True)

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        adj_matrix     = adj_arr,
        node_mask      = mask_arr,
        node_combo_ids = np.stack(ids_list,    axis=0),
        node_coords    = np.stack(coords_list, axis=0),
        prompt_tokens  = np.stack(ptok_list,   axis=0),
        prompt_mask    = np.stack(pmask_list,  axis=0),
        prompt_lens    = np.array(plen_list,   dtype=np.int32),
        n_nodes        = np.array(nnodes_list, dtype=np.int32),
        room_ids       = room_ids_out,
    )

    elapsed = time.perf_counter() - t0
    print(f"保存 → {out_path}  (耗时 {elapsed:.1f}s)")
    print(f"room_ids: max_id={room_ids_out.max()}  零值比例={( room_ids_out==0).mean():.3f}")


if __name__ == "__main__":
    main()
