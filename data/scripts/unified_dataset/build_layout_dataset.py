"""
生成布局数据集 npz，每条记录包含：
  - prompt_tokens : (max_text_len,)  BPE提示词token ID，不足补PAD
  - prompt_lens   : ()               提示词实际长度
  - adj_matrix    : (MAX_NODES, MAX_NODES)  0/1邻接矩阵，按visit顺序重排
  - node_coords   : (MAX_NODES, 2)   整数坐标，重心在原点，按visit顺序重排
  - node_mask     : (MAX_NODES,)     有效节点掩码
  - n_nodes       : ()               实际节点数

节点顺序与SENT游走一致，节点i对应graph_only.npz里token序列的第i+1个节点token。
每张图做 augment 次随机游走 → augment 条记录。

输入：
  data/jsonl/final_graph_dataset.jsonl
  data/processed/unified_vocab/bpe_tokenizer.json

输出：
  data/processed/unified_dataset/layout_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",        type=Path, default=Path("data/jsonl/final_graph_dataset.jsonl"))
    p.add_argument("--bpe-tokenizer",type=Path, default=Path("data/processed/unified_vocab/bpe_tokenizer.json"))
    p.add_argument("--output",       type=Path, default=Path("data/processed/unified_dataset/layout_dataset.npz"))
    p.add_argument("--augment",      type=int,  default=15)
    p.add_argument("--max-text-len", type=int,  default=128)
    p.add_argument("--max-nodes",    type=int,  default=40)
    p.add_argument("--seed",         type=int,  default=42)
    return p.parse_args()


def sample_sent(adj: np.ndarray, n: int, rng: random.Random) -> list:
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                neighbors[i].add(j)

    unvisited = set(range(n))
    all_nodes = set(range(n))
    v = rng.choice(list(unvisited))
    unvisited.remove(v)
    current_trail = [(v, set())]
    sent = []

    while unvisited:
        unvisited_nbrs = neighbors[v] & unvisited
        if not unvisited_nbrs:
            sent.append(current_trail)
            v = rng.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            current_trail = [(v, neighbors[v] & visited)]
        else:
            u = rng.choice(list(unvisited_nbrs))
            unvisited.remove(u)
            visited = all_nodes - unvisited
            current_trail.append((u, (neighbors[u] - {v}) & visited))
            v = u

    sent.append(current_trail)
    return sent


def get_visit_order(sent: list) -> list:
    """从SENT中按首次出现顺序提取节点访问序列"""
    seen = []
    seen_set = set()
    for trail in sent:
        for node_idx, _ in trail:
            if node_idx not in seen_set:
                seen.append(node_idx)
                seen_set.add(node_idx)
    return seen


def reorder_by_visit(visit_order: list, coords: np.ndarray,
                     adj: np.ndarray, n_nodes: int, max_nodes: int):
    """
    按visit_order重排坐标和邻接矩阵，坐标中心化并取整
    """
    coords_r = np.zeros((max_nodes, 2), dtype=np.float32)
    adj_r    = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    for new_i, orig_i in enumerate(visit_order):
        coords_r[new_i] = coords[orig_i]

    for new_i, orig_i in enumerate(visit_order):
        for new_j, orig_j in enumerate(visit_order):
            adj_r[new_i, new_j] = adj[orig_i, orig_j]

    # 中心化：有效节点重心移到原点，取整
    valid_coords = coords_r[:n_nodes]
    centroid = valid_coords.mean(axis=0)
    coords_r[:n_nodes] = np.round(valid_coords - centroid).astype(np.float32)

    return coords_r, adj_r


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    bpe     = Tokenizer.from_file(str(args.bpe_tokenizer))
    bpe_pad = 0  # BPE词表中PAD对应ID（未登录用0填充）
    MAX     = args.max_nodes

    prompt_tokens_list = []
    prompt_lens_list   = []
    adj_list           = []
    coords_list        = []
    mask_list          = []
    n_nodes_list       = []

    with args.jsonl.open("r", encoding="utf-8") as f:
        for rec_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record  = json.loads(line)
            n_nodes = int(record["n_nodes"])

            # 坐标
            raw_coords = np.array(record["node_coords"][:n_nodes], dtype=np.float32)

            # 邻接矩阵（去掉自环）
            adj = np.array(record["adj_matrix"], dtype=np.float32)[:n_nodes, :n_nodes].copy()
            np.fill_diagonal(adj, 0)

            # 提示词BPE编码
            prompt   = record.get("prompt", "").replace("\n", " ").strip()
            text_ids = bpe.encode(prompt).ids[:args.max_text_len]
            text_len = len(text_ids)
            padded_text = np.zeros(args.max_text_len, dtype=np.int32)
            padded_text[:text_len] = text_ids

            for aug_idx in range(args.augment):
                rng   = random.Random(args.seed + rec_idx * 1000 + aug_idx)
                sent  = sample_sent(adj, n_nodes, rng)
                visit = get_visit_order(sent)

                coords_r, adj_r = reorder_by_visit(visit, raw_coords, adj, n_nodes, MAX)

                # 节点掩码
                mask = np.zeros(MAX, dtype=np.int32)
                mask[:n_nodes] = 1

                # 邻接矩阵转int
                adj_r_int = adj_r.astype(np.int32)

                # 坐标转int（有效节点已取整，padding位保持0）
                coords_int = coords_r.astype(np.int32)

                prompt_tokens_list.append(padded_text)
                prompt_lens_list.append(text_len)
                adj_list.append(adj_r_int)
                coords_list.append(coords_int)
                mask_list.append(mask)
                n_nodes_list.append(n_nodes)

            if (rec_idx + 1) % 5000 == 0:
                print(f"处理 {rec_idx + 1} 张图 -> {len(adj_list)} 条记录")

    print(f"\n总记录数: {len(adj_list)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prompt_tokens = np.array(prompt_tokens_list, dtype=np.int32),  # (N, max_text_len)
        prompt_lens   = np.array(prompt_lens_list,   dtype=np.int32),  # (N,)
        adj_matrix    = np.array(adj_list,           dtype=np.int32),  # (N, MAX_NODES, MAX_NODES)
        node_coords   = np.array(coords_list,        dtype=np.int32),  # (N, MAX_NODES, 2)
        node_mask     = np.array(mask_list,          dtype=np.int32),  # (N, MAX_NODES)
        n_nodes       = np.array(n_nodes_list,       dtype=np.int32),  # (N,)
    )

    # 统计
    coords_arr = np.array(coords_list, dtype=np.int32)
    valid_mask = np.array(mask_list, dtype=bool)
    all_valid_coords = []
    for i in range(len(coords_arr)):
        n = int(n_nodes_list[i])
        all_valid_coords.append(coords_arr[i, :n])
    all_coords_flat = np.concatenate(all_valid_coords, axis=0)
    print(f"坐标范围: x=[{all_coords_flat[:,0].min()}, {all_coords_flat[:,0].max()}] "
          f"y=[{all_coords_flat[:,1].min()}, {all_coords_flat[:,1].max()}]")
    print(f"已保存 -> {args.output}")


if __name__ == "__main__":
    main()
