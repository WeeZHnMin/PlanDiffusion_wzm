"""
Convert data/jsonl/mapped_type_data.jsonl into a node_data-like JSONL.

Output fields:
    prompt, n_nodes, node_coords, node_mask, node_types, adj_matrix
plus provenance:
    image, source_file, source_line

node_types: list of room type strings, length=n_max, padding with "" for non-nodes.
  e.g. ["bathroom","living_room","kitchen","bedroom","","",...]
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
SRC_FILE = DATA_DIR / "jsonl" / "mapped_type_data.jsonl"
OUT_FILE = DATA_DIR / "jsonl" / "mapped_node_data.jsonl"


def main() -> None:
    if not SRC_FILE.exists():
        raise SystemExit(f"Missing source file: {SRC_FILE}")

    records = []
    with SRC_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise SystemExit(f"No records found in {SRC_FILE}")

    n_max = max(len(r["vertices"]) for r in records)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for r in records:
            n = len(r["vertices"])

            # 每个顶点可能被多个房间共享，取第一个包含它的房间类型
            coord_to_type = {}
            for room in r["rooms"]:
                for coord in room["coords"]:
                    key = tuple(coord)
                    if key not in coord_to_type:   # 先到先得，保留第一个房间的类型
                        coord_to_type[key] = room["type"]

            node_types = [coord_to_type.get(tuple(v), "unknown") for v in r["vertices"]]
            node_types = node_types + [""] * (n_max - n)

            node_coords = r["vertices"] + [[0, 0]] * (n_max - n)
            node_mask   = [1] * n + [0] * (n_max - n)
            out_obj = {
                "prompt":      r["prompt"],
                "n_nodes":     n,
                "node_coords": node_coords,
                "node_mask":   node_mask,
                "node_types":  node_types,
                "adj_matrix":  r["adj_matrix"],
                "image":       r.get("image", ""),
                "source_file": r.get("source_file", ""),
                "source_line": r.get("source_line", 0),
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} rows -> {OUT_FILE}")
    print(f"n_max={n_max}")


if __name__ == "__main__":
    main()
