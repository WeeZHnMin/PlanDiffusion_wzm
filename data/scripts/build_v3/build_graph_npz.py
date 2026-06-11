"""
将 final_graph_dataset_v2.jsonl 转换为 DiGress 训练用的 NPZ 文件。

输入：
  data/jsonl/final_graph_dataset_v2.jsonl
  data/processed/unified_vocab/bpe_tokenizer.json

输出：
  data/processed/graph_diffusion/graph_dataset.npz

字段：
  adj_matrix    : (N, 40, 40)  int32   二值邻接矩阵（无自环）
  node_mask     : (N, 40)      int32   1=有效节点，0=padding
  node_combo_ids: (N, 40)      int32   节点类型 ID（1~32），padding=0
  node_coords   : (N, 40, 2)   int32   节点坐标（中心化后取整），padding=0
  prompt_tokens : (N, 192)     int32   BERT input_ids
  prompt_lens   : (N,)         int32   文本实际长度
  n_nodes       : (N,)         int32   有效节点数

增强：
  每张图做 augment 次随机节点重排，增加数据多样性。
  DiGress 模型本身有置换等变性，augment=3 足够。

用法：
  python -m data.scripts.build_v2.build_graph_npz
  python -m data.scripts.build_v2.build_graph_npz --augment 5
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
    p.add_argument("--jsonl",    default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--bert",     default="models/bert-base-uncased")
    p.add_argument("--output",   default="data/processed/node_diffusion_cross_att/graph_dataset.npz")
    p.add_argument("--augment",  type=int, default=8,
                   help="每张图随机节点重排次数（1=不增强，只用原始顺序）")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def permute_graph(adj: np.ndarray, combo_ids: np.ndarray, coords: np.ndarray,
                  n: int, perm: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 perm 重排前 n 个节点，padding 部分不动。"""
    full_perm = perm + list(range(n, MAX_NODES))
    new_adj    = adj[np.ix_(full_perm, full_perm)]
    new_ids    = combo_ids[full_perm]
    new_coords = coords[full_perm]
    return new_adj, new_ids, new_coords


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
    n_graphs   = 0
    n_skipped  = 0

    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rec    = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()

            # BERT 编码文本（超出 MAX_TEXT_LEN 的直接跳过）
            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            n       = int(rec["n_nodes"])
            n_graphs += 1

            # 邻接矩阵（去除自环，取前 n×n）
            adj_full = np.array(rec["adj_matrix"], dtype=np.int32)   # (40, 40)
            np.fill_diagonal(adj_full, 0)

            # 节点类型 ID（长度 40，padding=0）
            raw_ids   = rec["node_combo_ids"]
            combo_ids = np.array(raw_ids[:MAX_NODES], dtype=np.int32)

            # 节点坐标（中心化整数坐标，shape (40,2)，padding=0）
            raw_coords = rec["node_coords"][:MAX_NODES]   # list of [x, y]
            coords = np.zeros((MAX_NODES, 2), dtype=np.int32)
            coords[:len(raw_coords)] = raw_coords

            # 节点掩码
            mask = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n] = 1

            # padding 到定长
            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen = len(enc['input_ids'])
            padded[:tlen]   = enc['input_ids']
            attn_msk[:tlen] = enc['attention_mask']
            text_len = tlen

            # 生成 augment 个随机排列
            base_perm = list(range(n))
            perms = [base_perm]   # 第一个是原始顺序
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
                plen_list.append(text_len)
                nnodes_list.append(n)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                total_records = len(adj_list)
                print(f"  {line_no+1} 张图 → {total_records} 条记录  ({elapsed:.1f}s)")

    total_records = len(adj_list)
    print(f"\n共 {n_graphs} 张图（跳过 {n_skipped} 条 prompt >{MAX_TEXT_LEN} tokens），增强后 {total_records} 条记录")
    print("打包为 numpy 数组...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        adj_matrix     = np.stack(adj_list,    axis=0),  # (N, 40, 40)
        node_mask      = np.stack(mask_list,   axis=0),  # (N, 40)
        node_combo_ids = np.stack(ids_list,    axis=0),  # (N, 40)
        node_coords    = np.stack(coords_list, axis=0),  # (N, 40, 2)
        prompt_tokens  = np.stack(ptok_list,   axis=0),  # (N, 128) BERT input_ids
        prompt_mask    = np.stack(pmask_list,  axis=0),  # (N, 128) BERT attention_mask
        prompt_lens    = np.array(plen_list,   dtype=np.int32),
        n_nodes        = np.array(nnodes_list, dtype=np.int32),
    )

    elapsed = time.perf_counter() - t0
    print(f"保存 → {out_path}")
    print(f"耗时：{elapsed:.1f}s")

    # 简单统计
    arr = np.stack(ids_list, axis=0)
    valid = arr[np.stack(mask_list).astype(bool)]
    unique, counts = np.unique(valid, return_counts=True)
    print(f"\n节点类型分布（前10）:")
    for uid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  combo_id={uid:2d}  count={cnt}")


if __name__ == "__main__":
    main()
