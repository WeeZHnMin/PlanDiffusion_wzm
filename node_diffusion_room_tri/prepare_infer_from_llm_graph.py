"""Convert LLM graph eval JSONL to node_diffusion_room_tri.infer input.

This is intentionally a thin field mapping:
  gen_n_nodes -> n_nodes
  gen_adj     -> adj_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_jsonl", default="outputs/llm_graph_eval_val.jsonl")
    p.add_argument("--out", default="outputs/llm_graph_for_tri_infer.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped = 0
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
            if gen_n is None or gen_adj is None:
                skipped += 1
                continue

            gen_n = int(gen_n)
            rec = {
                "source_index": llm.get("source_index"),
                "prompt": llm.get("prompt", ""),
                "n_nodes": gen_n,
                "adj_matrix": [row[:gen_n] for row in gen_adj[:gen_n]],
                "llm_gen_valid": llm.get("gen_valid"),
                "llm_ged": llm.get("ged"),
                "llm_face_diff": llm.get("face_diff"),
                "llm_gen_faces": llm.get("gen_faces"),
                "llm_gt_faces": llm.get("gt_faces"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"read llm rows: {total}")
    print(f"kept: {kept}")
    print(f"skipped_missing_gen_graph: {skipped}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
