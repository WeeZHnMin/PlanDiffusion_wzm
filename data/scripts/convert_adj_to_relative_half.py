"""
Convert node JSONL adjacency matrix into a half relative-offset matrix.

Rule:
- For each pair (i, j) with i <= j (upper-triangular half):
  - if both nodes are valid and adj_matrix[i][j] == 1:
      store (x_i - x_j, y_i - y_j)
  - else:
      store (0, 0)

Default output keeps key fields and drops dense adj_matrix to save space.

Usage:
  python data/scripts/convert_adj_to_relative_half.py
  python data/scripts/convert_adj_to_relative_half.py --keep-all
  python data/scripts/convert_adj_to_relative_half.py --keep-adj
  python data/scripts/convert_adj_to_relative_half.py --input data/jsonl/train_nodes.jsonl --output data/jsonl/train_nodes_rel_half.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_rel_adj_half(
    node_coords: List[List[int]],
    node_mask: List[int],
    adj_matrix: List[List[int]],
    drop_diagonal: bool = False,
) -> List[List[List[int]]]:
    n_total = len(node_coords)
    out: List[List[List[int]]] = []
    for i in range(n_total):
        xi, yi = node_coords[i]
        row: List[List[int]] = []
        valid_i = i < len(node_mask) and node_mask[i] == 1
        j_start = i + 1 if drop_diagonal else i
        for j in range(j_start, n_total):
            valid_j = j < len(node_mask) and node_mask[j] == 1
            connected = (
                valid_i
                and valid_j
                and i < len(adj_matrix)
                and j < len(adj_matrix[i])
                and adj_matrix[i][j] == 1
            )
            if connected:
                xj, yj = node_coords[j]
                row.append([xi - xj, yi - yj])
            else:
                row.append([0, 0])
        out.append(row)
    return out


def build_adj_half(
    node_mask: List[int],
    adj_matrix: List[List[int]],
    drop_diagonal: bool = False,
) -> List[List[int]]:
    n_total = len(node_mask)
    out: List[List[int]] = []
    for i in range(n_total):
        row: List[int] = []
        valid_i = node_mask[i] == 1
        j_start = i + 1 if drop_diagonal else i
        for j in range(j_start, n_total):
            valid_j = node_mask[j] == 1
            connected = (
                valid_i
                and valid_j
                and i < len(adj_matrix)
                and j < len(adj_matrix[i])
                and adj_matrix[i][j] == 1
            )
            row.append(1 if connected else 0)
        out.append(row)
    return out


def convert_record(
    rec: Dict[str, Any], keep_all: bool, keep_adj: bool, drop_diagonal: bool
) -> Dict[str, Any]:
    node_coords = rec["node_coords"]
    node_mask = rec["node_mask"]
    adj_matrix = rec["adj_matrix"]
    rel_adj_half = build_rel_adj_half(
        node_coords, node_mask, adj_matrix, drop_diagonal=drop_diagonal
    )

    if keep_all:
        out = dict(rec)
        out["rel_adj_half"] = rel_adj_half
        return out

    out = {
        "prompt": rec.get("prompt", ""),
        "n_nodes": rec.get("n_nodes", 0),
        "node_coords": node_coords,
        "node_mask": node_mask,
        "rel_adj_half": rel_adj_half,
        "image": rec.get("image", ""),
        "source_file": rec.get("source_file", ""),
        "source_line": rec.get("source_line", 0),
    }
    if keep_adj:
        out["adj_matrix"] = build_adj_half(
            node_mask=node_mask,
            adj_matrix=adj_matrix,
            drop_diagonal=drop_diagonal,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/jsonl/train_nodes.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="data/jsonl/train_nodes_rel_half.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all original fields and add rel_adj_half",
    )
    parser.add_argument(
        "--keep-adj",
        action="store_true",
        help="Keep original adj_matrix in compact output mode",
    )
    parser.add_argument(
        "--drop-diagonal",
        action="store_true",
        help="Exclude diagonal entries; shape becomes (n-1),(n-2),...,1",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Missing input file: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse failed at line {line_no}: {e}") from e

            needed = {"node_coords", "node_mask", "adj_matrix"}
            if not needed.issubset(rec.keys()):
                missing = sorted(needed - set(rec.keys()))
                raise SystemExit(f"Missing keys at line {line_no}: {missing}")

            out_obj = convert_record(
                rec,
                keep_all=args.keep_all,
                keep_adj=args.keep_adj,
                drop_diagonal=args.drop_diagonal,
            )
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} rows -> {out_path}")


if __name__ == "__main__":
    main()
