"""
生成树+补边格式的训练数据。

序列格式：
  [BPE文本] BOS_G [N_tok] [p1 p2 ... p_{N-1}] SEP [i1 j1 i2 j2 ...] EOS_G

  N_tok    : N=k → N_START + (k-1)
  父节点 pi : 节点 i 的父节点 j → NODE_START + j
  补边 (i,j): 两个 token → NODE_START+i, NODE_START+j

词表布局（12084个token）：
  0 ~ 11999  : BPE 文本
  12000      : PAD
  12001      : BOS_G
  12002      : EOS_G
  12003      : SEP
  12004~12043: N_START（N=1~40）
  12044~12083: NODE_START（节点0~39）

输入：
  data/jsonl/final_graph_dataset_v2.jsonl
  node_diffusion/unified_vocab/bpe_tokenizer.json

输出：
  data/processed/graph_tree/text_graph_tree.npz
  data/processed/graph_tree/vocab_config.json

用法：
  python -m data.scripts.build_v2.build_text_graph_tree
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

MAX_NODES    = 40
MAX_TEXT_LEN = 128
MAX_SEQ_LEN  = 384    # 文本128 + BOS + N + 父节点39 + SEP + 补边最多~60×2 + EOS

PAD_ID    = 12000
BOS_ID    = 12001
EOS_ID    = 12002
SEP_ID    = 12003
N_START   = 12004   # N=k → N_START + (k-1)
NODE_START = 12044  # 节点j → NODE_START + j
VOCAB_SIZE = 12084


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",   default="data/jsonl/final_graph_dataset_v2.jsonl")
    p.add_argument("--bpe",     default="node_diffusion/unified_vocab/bpe_tokenizer.json")
    p.add_argument("--output",  default="data/processed/graph_tree/text_graph_tree.npz")
    p.add_argument("--vocab-out", default="data/processed/graph_tree/vocab_config.json")
    return p.parse_args()


def bfs_spanning_tree(adj: list, n: int) -> tuple[list[int], set[tuple[int,int]]]:
    """
    从节点0做BFS，返回：
      parents   : list[int], 长度 N-1，parents[i-1] = 节点i的父节点
      tree_edges: set of (min,max) tuples
    """
    visited = [False] * n
    parents = [-1] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True

    while queue:
        v = queue.popleft()
        for u in range(n):
            if adj[v][u] == 1 and not visited[u]:
                visited[u] = True
                parents[u] = v
                tree_edges.add((min(v, u), max(v, u)))
                queue.append(u)

    return parents[1:], tree_edges   # parents[0] 是节点1的父节点


def get_extra_edges(adj: list, n: int,
                    tree_edges: set[tuple[int,int]]) -> list[tuple[int,int]]:
    """返回不在生成树里的边（上三角，按(i,j)排序）"""
    extra = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1 and (i, j) not in tree_edges:
                extra.append((i, j))
    return extra


def graph_to_tokens(n: int, parents: list[int],
                    extra_edges: list[tuple[int,int]],
                    text_ids: list[int]) -> list[int]:
    """
    构建完整序列：
    text_ids + BOS_G + N_tok + 父节点序列 + SEP + 补边 + EOS_G
    """
    tokens = list(text_ids)
    tokens.append(BOS_ID)
    tokens.append(N_START + (n - 1))           # N token

    for p in parents:                          # N-1 个父节点 token
        tokens.append(NODE_START + p)

    tokens.append(SEP_ID)

    for i, j in extra_edges:                   # 补边，每条边两个 token
        tokens.append(NODE_START + i)
        tokens.append(NODE_START + j)

    tokens.append(EOS_ID)
    return tokens


def save_vocab_config(output_path: Path):
    cfg = {
        "bpe_vocab_size":  12000,
        "PAD_ID":          PAD_ID,
        "BOS_ID":          BOS_ID,
        "EOS_ID":          EOS_ID,
        "SEP_ID":          SEP_ID,
        "N_START":         N_START,
        "NODE_START":      NODE_START,
        "MAX_NODES":       MAX_NODES,
        "MAX_TEXT_LEN":    MAX_TEXT_LEN,
        "MAX_SEQ_LEN":     MAX_SEQ_LEN,
        "VOCAB_SIZE":      VOCAB_SIZE,
        "note": "N=k → N_START+(k-1), node_j → NODE_START+j",
    }
    output_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"vocab config → {output_path}")


def main():
    args   = parse_args()
    bpe    = Tokenizer.from_file(args.bpe)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokens_list  = []
    lengths_list = []
    text_lens_list = []
    n_graphs = 0
    truncated = 0
    t0 = time.perf_counter()

    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            n   = int(rec["n_nodes"])
            n_graphs += 1

            # 邻接矩阵（去除自环）
            adj = rec["adj_matrix"][:n]
            adj = [row[:n] for row in adj]
            for i in range(n):
                adj[i][i] = 0

            # BFS 生成树
            parents, tree_edges = bfs_spanning_tree(adj, n)

            # 补充边
            extra_edges = get_extra_edges(adj, n, tree_edges)

            # BPE 文本编码
            prompt   = rec.get("prompt", "").replace("\n", " ").strip()
            text_ids = bpe.encode(prompt).ids[:MAX_TEXT_LEN]

            # 组装序列
            seq = graph_to_tokens(n, parents, extra_edges, text_ids)

            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
                truncated += 1

            seq_len  = len(seq)
            padded   = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int32)
            padded[:seq_len] = seq

            tokens_list.append(padded)
            lengths_list.append(seq_len)
            text_lens_list.append(len(text_ids) + 1)  # +1 for BOS_G

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 张图 → {n_graphs} 条序列  ({elapsed:.1f}s)")

    print(f"\n共 {n_graphs} 张图，截断 {truncated} 条")
    print("打包保存...")

    np.savez_compressed(
        out_path,
        tokens    = np.stack(tokens_list,  axis=0),   # (N, MAX_SEQ_LEN)
        lengths   = np.array(lengths_list, dtype=np.int32),
        text_lens = np.array(text_lens_list, dtype=np.int32),
    )

    elapsed = time.perf_counter() - t0
    print(f"保存 → {out_path}  ({elapsed:.1f}s)")

    # 统计序列长度
    lens = np.array(lengths_list)
    print(f"序列长度: min={lens.min()} max={lens.max()} mean={lens.mean():.1f} p95={int(np.percentile(lens, 95))}")

    save_vocab_config(Path(args.vocab_out))


if __name__ == "__main__":
    main()
