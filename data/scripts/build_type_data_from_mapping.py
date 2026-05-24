"""
Build a type_data-like JSONL from mapping.jsonl and the original train_jsonl rows.

Output rows contain the fields needed by prepare_node_data.py:
    prompt, rooms, vertices, adj_matrix
plus provenance:
    image, source_file, source_line
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MAPPING_FILE = DATA_DIR / "viz_50000" / "mapping.jsonl"
SRC_DIR = DATA_DIR / "Architext_v1" / "train_jsonl"
OUT_FILE = DATA_DIR / "jsonl" / "mapped_type_data.jsonl"


def load_mapping():
    rows = []
    wanted = {}
    with MAPPING_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            wanted.setdefault(row["source_file"], set()).add(int(row["source_line"]))
    return rows, wanted


def load_needed_rows(wanted):
    found = {}
    for src_file, line_set in wanted.items():
        src_path = SRC_DIR / src_file
        if not src_path.exists():
            raise SystemExit(f"Missing source file: {src_path}")
        with src_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no not in line_set:
                    continue
                obj = json.loads(line)
                found[(src_file, line_no)] = obj
    return found


def build_adj_matrix(n_max, vertex_adj):
    n = len(vertex_adj)
    adj = [[0] * n_max for _ in range(n_max)]
    for i in range(n):
        adj[i][i] = 1
        for j in vertex_adj[i]:
            adj[i][j] = 1
    return adj


def main() -> None:
    if not MAPPING_FILE.exists():
        raise SystemExit(f"Missing mapping file: {MAPPING_FILE}")

    mapping_rows, wanted = load_mapping()
    source_rows = load_needed_rows(wanted)

    n_max = 0
    for row in source_rows.values():
        n_max = max(n_max, len(row["vertices"]))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for row in mapping_rows:
            key = (row["source_file"], int(row["source_line"]))
            src = source_rows.get(key)
            if src is None:
                raise SystemExit(f"Missing mapped source row: {key}")

            out_obj = {
                "prompt": src["prompt"],
                "rooms": src["rooms"],
                "vertices": src["vertices"],
                "adj_matrix": build_adj_matrix(n_max, src["vertex_adj"]),
                "image": row["image"],
                "source_file": row["source_file"],
                "source_line": int(row["source_line"]),
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mapping_rows)} rows -> {OUT_FILE}")
    print(f"n_max={n_max}")


if __name__ == "__main__":
    main()
