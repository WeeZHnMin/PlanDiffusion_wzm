"""Convert LLM graph eval JSONL to node_diffusion_room_tri.infer input.

This is intentionally a thin field mapping:
  gen_n_nodes -> n_nodes
  gen_adj     -> adj_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .graph_prune import prune_non_cycle_nodes

MAX_NODES = 40


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_jsonl", default="data/jsonl/graph_160k_spatial_val.jsonl")
    p.add_argument("--llm_jsonl", default="outputs/llm_graph_eval_val.jsonl")
    p.add_argument("--out", default="outputs/llm_graph_for_tri_infer.jsonl")
    return p.parse_args()


def read_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    val_rows = read_jsonl(args.val_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped_missing_adj = skipped_missing_source = skipped_over_max_nodes = 0
    with open(args.llm_jsonl, encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            llm = json.loads(line)
            gen_n = llm.get("gen_n_nodes")
            gen_adj = llm.get("gen_adj")
            if gen_adj is None:
                skipped_missing_adj += 1
                continue
            source_index = llm.get("source_index")
            if source_index is None:
                skipped_missing_source += 1
                continue
            source_index = int(source_index)
            if source_index < 0 or source_index >= len(val_rows):
                skipped_missing_source += 1
                continue
            gt = val_rows[source_index]

            gen_n = int(gen_n) if gen_n is not None else len(gen_adj)
            adj_np = np.array(gen_adj, dtype=np.int32)[:gen_n, :gen_n]
            adj_np, _ = prune_non_cycle_nodes(adj_np)
            gen_n = int(adj_np.shape[0])
            if gen_n > MAX_NODES:
                skipped_over_max_nodes += 1
                continue
            rec = {
                "source_index": source_index,
                "prompt": llm.get("prompt", ""),
                "n_nodes": gen_n,
                "adj_matrix": adj_np.astype(int).tolist(),
                "gt_n_nodes": int(gt["n_nodes"]),
                "gt_adj_matrix": gt["adj_matrix"],
                "gt_node_coords": gt.get("node_coords"),
                "gt_node_types": gt.get("node_types"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"read llm rows: {total}")
    print(f"kept: {kept}")
    print(f"skipped_missing_gen_adj: {skipped_missing_adj}")
    print(f"skipped_missing_source: {skipped_missing_source}")
    print(f"skipped_over_max_nodes: {skipped_over_max_nodes}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
