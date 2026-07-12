"""Build tri diffusion infer JSONL from LLM graph evaluation output.

The LLM graph evaluator writes prompt + generated adjacency only. This script
joins those rows back to the original JSONL by source_index, then emits the
format expected by node_diffusion_room_tri.infer:

  prompt, n_nodes, adj_matrix, optional node_coords/node_types
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_jsonl", default="data/jsonl/graph_160k_spatial_val.jsonl")
    p.add_argument("--llm_jsonl", default="outputs/llm_graph_eval_val.jsonl")
    p.add_argument("--out", default="outputs/llm_graph_for_tri_infer.jsonl")
    p.add_argument("--keep_invalid", action="store_true",
                   help="Keep invalid LLM generations as zero-edge placeholders.")
    p.add_argument("--allow_truncate", action="store_true",
                   help="Allow gen_n_nodes <= original n_nodes and truncate coords/types.")
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

    total = kept = skipped_invalid = skipped_missing = skipped_n = 0
    with open(args.llm_jsonl, encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            total += 1
            llm = json.loads(line)
            source_index = llm.get("source_index", line_no)
            if source_index is None or source_index < 0 or source_index >= len(val_rows):
                skipped_missing += 1
                continue

            if not llm.get("gen_valid", False):
                if not args.keep_invalid:
                    skipped_invalid += 1
                    continue
                gen_n = int(llm.get("gt_n_nodes", 0))
                gen_adj = [[0] * gen_n for _ in range(gen_n)]
            else:
                gen_n = int(llm["gen_n_nodes"])
                gen_adj = llm["gen_adj"]

            val = val_rows[source_index]
            orig_n = int(val["n_nodes"])
            if gen_n > orig_n:
                skipped_n += 1
                continue
            if gen_n != orig_n and not args.allow_truncate:
                skipped_n += 1
                continue

            rec = {
                "source_index": source_index,
                "prompt": llm.get("prompt", val.get("prompt", "")),
                "n_nodes": gen_n,
                "adj_matrix": [row[:gen_n] for row in gen_adj[:gen_n]],
                "llm_ged": llm.get("ged"),
                "llm_gen_faces": llm.get("gen_faces"),
                "llm_gt_faces": llm.get("gt_faces"),
            }
            if "node_coords" in val:
                rec["node_coords"] = val["node_coords"][:gen_n]
            if "node_types" in val:
                rec["node_types"] = val["node_types"][:gen_n]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"read llm rows: {total}")
    print(f"kept: {kept}")
    print(f"skipped_invalid: {skipped_invalid}")
    print(f"skipped_missing_source: {skipped_missing}")
    print(f"skipped_node_count: {skipped_n}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
