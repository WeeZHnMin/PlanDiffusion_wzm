"""
Build tree-plus-extra-edge training data for llm_graph.

Sequence format:
  [BPE text] BOS_G [N_tok] [p1 p2 ... p_{N-1}] SEP [i1 j1 i2 j2 ...] EOS_G

  N_tok    : N=k -> N_START + (k-1)
  parent pi: parent index of node i -> NODE_START + parent
  edge(i,j): two tokens -> NODE_START+i, NODE_START+j
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

MAX_NODES = 40
MAX_SEQ_LEN = 384

PAD_ID = 10000
BOS_ID = 10001
EOS_ID = 10002
SEP_ID = 10003
N_START = 10004
NODE_START = 10044

DEFAULT_JSONL_BY_SPLIT = {
    "train": "data/jsonl/graph_160k_spatial_train.jsonl",
    "val": "data/jsonl/graph_160k_spatial_val.jsonl",
}

DEFAULT_OUTPUT_BY_SPLIT = {
    "train": "data/processed/graph_tree/text_graph_tree_train.npz",
    "val": "data/processed/graph_tree/text_graph_tree_val.npz",
}

DEFAULT_AUGMENT_BY_SPLIT = {
    "train": 12,
    "val": 1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--jsonl", default=None, help="Override the default jsonl path for the chosen split.")
    p.add_argument("--bpe", default="llm_graph/vocab/wp_tokenizer.json")
    p.add_argument("--output", default=None, help="Override the default output path for the chosen split.")
    p.add_argument(
        "--augment",
        type=int,
        default=None,
        help="How many BFS root choices to try per graph. Default: train=12, val=1.",
    )
    p.add_argument("--progress-every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bfs_spanning_tree(
    adj: list[list[int]], n: int, start: int = 0
) -> tuple[list[int], set[tuple[int, int]] | None, dict[int, int] | None]:
    """
    Run BFS from `start`.

    Returns:
      parents: parent ids in the reordered BFS index space, length N-1
      tree_edges: undirected tree edges in reordered ids
      new_id: original node id -> reordered BFS node id

    If the graph is disconnected, return failure markers so caller can skip it.
    """
    visited = [False] * n
    orig_parent = [-1] * n
    visit_order: list[int] = []
    queue = deque([start])
    visited[start] = True

    while queue:
        v = queue.popleft()
        visit_order.append(v)
        for u in range(n):
            if adj[v][u] == 1 and not visited[u]:
                visited[u] = True
                orig_parent[u] = v
                queue.append(u)

    if not all(visited):
        return [], None, None

    new_id = {orig: new for new, orig in enumerate(visit_order)}

    parents: list[int] = []
    tree_edges: set[tuple[int, int]] = set()
    for new_node in range(1, n):
        orig_node = visit_order[new_node]
        orig_p = orig_parent[orig_node]
        new_p = new_id[orig_p]
        parents.append(new_p)
        tree_edges.add((min(new_node, new_p), max(new_node, new_p)))

    return parents, tree_edges, new_id


def get_extra_edges(
    adj: list[list[int]], n: int, tree_edges: set[tuple[int, int]], new_id: dict[int, int]
) -> list[tuple[int, int]]:
    extra = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                ni, nj = new_id[i], new_id[j]
                edge = (min(ni, nj), max(ni, nj))
                if edge not in tree_edges:
                    extra.append(edge)
    return sorted(extra)


def graph_to_tokens(
    n: int, parents: list[int], extra_edges: list[tuple[int, int]], text_ids: list[int]
) -> list[int]:
    tokens = list(text_ids)
    tokens.append(BOS_ID)
    tokens.append(N_START + (n - 1))

    for parent in parents:
        tokens.append(NODE_START + parent)

    tokens.append(SEP_ID)

    for i, j in extra_edges:
        tokens.append(NODE_START + i)
        tokens.append(NODE_START + j)

    tokens.append(EOS_ID)
    return tokens


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    bpe = Tokenizer.from_file(args.bpe)
    jsonl_path = Path(args.jsonl or DEFAULT_JSONL_BY_SPLIT[args.split])
    out_path = Path(args.output or DEFAULT_OUTPUT_BY_SPLIT[args.split])
    augment = args.augment if args.augment is not None else DEFAULT_AUGMENT_BY_SPLIT[args.split]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokens_list = []
    lengths_list = []
    text_lens_list = []
    coords_list = []
    mask_list = []
    n_graphs = 0
    n_empty_text = 0
    n_disconnected = 0
    n_invalid = 0
    truncated = 0
    t0 = time.perf_counter()

    print(f"split={args.split} jsonl={jsonl_path} output={out_path} augment={augment}")

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()
            text_ids = bpe.encode(prompt).ids
            if not text_ids:
                n_empty_text += 1

            n = int(rec["n_nodes"])
            if n < 1 or n > MAX_NODES:
                n_invalid += 1
                continue
            n_graphs += 1

            raw_adj = rec.get("adj_matrix", [])
            raw_coords = rec.get("node_coords", [])
            raw_mask = rec.get("node_mask", [])
            if len(raw_adj) < n or len(raw_coords) < n or len(raw_mask) < n:
                n_invalid += 1
                continue

            adj = [list(row[:n]) for row in raw_adj[:n]]
            for i in range(n):
                if len(adj[i]) < n:
                    n_invalid += 1
                    adj = []
                    break
                for j in range(n):
                    if adj[i][j]:
                        adj[j][i] = 1
                adj[i][i] = 0
            if not adj:
                continue

            text_len = len(text_ids) + 1
            orig_coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
            orig_mask = np.zeros(MAX_NODES, dtype=np.int32)
            orig_coords[:n] = np.array(raw_coords[:n], dtype=np.float32)
            orig_mask[:n] = np.array(raw_mask[:n], dtype=np.int32)

            _, tree_edges0, _ = bfs_spanning_tree(adj, n, start=0)
            if tree_edges0 is None:
                n_disconnected += 1
                continue

            starts = [0] + rng.sample(range(1, n), min(augment - 1, n - 1))

            for start in starts:
                parents, tree_edges, new_id = bfs_spanning_tree(adj, n, start)
                if tree_edges is None or new_id is None:
                    continue

                extra_edges = get_extra_edges(adj, n, tree_edges, new_id)
                seq = graph_to_tokens(n, parents, extra_edges, text_ids)
                if len(seq) > MAX_SEQ_LEN:
                    truncated += 1
                    continue

                seq_len = len(seq)
                padded = np.full(MAX_SEQ_LEN, PAD_ID, dtype=np.int32)
                padded[:seq_len] = seq

                visit_order = sorted(new_id.keys(), key=lambda node: new_id[node])
                new_coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
                new_mask = np.zeros(MAX_NODES, dtype=np.int32)
                for new_i, orig_i in enumerate(visit_order):
                    new_coords[new_i] = orig_coords[orig_i]
                    new_mask[new_i] = orig_mask[orig_i]

                tokens_list.append(padded)
                lengths_list.append(seq_len)
                text_lens_list.append(text_len)
                coords_list.append(new_coords)
                mask_list.append(new_mask)

            if args.progress_every > 0 and line_no % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"processed={line_no} kept={len(tokens_list)} "
                    f"empty_text={n_empty_text} invalid={n_invalid} disconnected={n_disconnected} "
                    f"truncated={truncated} elapsed={elapsed:.1f}s"
                )

    if not tokens_list:
        raise RuntimeError("No samples were kept. Check prompt length, connectivity, and MAX_SEQ_LEN.")

    print(
        f"\nTotal graphs={n_graphs}, empty_text={n_empty_text}, invalid={n_invalid}, "
        f"filtered_disconnected={n_disconnected}, kept={len(tokens_list)}, truncated={truncated}"
    )
    print("Saving...")

    np.savez_compressed(
        out_path,
        tokens=np.stack(tokens_list, axis=0),
        lengths=np.array(lengths_list, dtype=np.int32),
        text_lens=np.array(text_lens_list, dtype=np.int32),
        node_coords=np.stack(coords_list, axis=0),
        node_mask=np.stack(mask_list, axis=0),
    )

    elapsed = time.perf_counter() - t0
    lens = np.array(lengths_list)
    print(f"Saved -> {out_path} ({elapsed:.1f}s)")
    print(
        f"Sequence length: min={lens.min()} max={lens.max()} "
        f"mean={lens.mean():.1f} p95={int(np.percentile(lens, 95))}"
    )


if __name__ == "__main__":
    main()
