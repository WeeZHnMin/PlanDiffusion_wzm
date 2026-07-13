"""Inspect recovered room rings and their aligned LLM-pipeline inference images.

This is a small sanity-check helper before sending images to a multimodal LLM.
It prints the image path and the node-index cycle for each recovered room so
the numbers can be compared with the rendered node labels in the PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .eval_iou_llm_room_types import recover_pred_rooms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="outputs/tri_from_llm_graph_ddim500.jsonl")
    p.add_argument("--image_dir", default="outputs/tri_from_llm_graph_imgs/ddim500")
    p.add_argument("--image_ext", default=".png")
    p.add_argument("--rows", nargs="*", type=int, default=None, help="0-based JSONL row indices to inspect")
    p.add_argument("--n", type=int, default=2, help="Inspect first N rows when --rows is omitted")
    return p.parse_args()


def read_selected_rows(path: str, rows: List[int] | None, n: int) -> Iterable[tuple[int, Dict[str, Any]]]:
    wanted = set(rows) if rows is not None else None
    with open(path, encoding="utf-8") as f:
        yielded = 0
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if wanted is not None and idx not in wanted:
                continue
            yield idx, json.loads(line)
            yielded += 1
            if wanted is None and yielded >= n:
                break


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"

    for row_index, row in read_selected_rows(args.jsonl, args.rows, args.n):
        image_index = int(row.get("image_index", row_index))
        original_image_index = row.get("original_image_index")
        image_path = image_dir / f"{image_index:05d}{image_ext}"
        rooms, _ = recover_pred_rooms(row)

        print("=" * 80)
        print(f"row_index: {row_index}")
        print(f"image_index: {image_index}")
        if original_image_index is not None:
            print(f"original_image_index: {original_image_index}")
        print(f"image_path: {image_path}")
        print(f"image_exists: {image_path.exists()}")
        print(f"n_nodes: {row.get('n_nodes')}")
        print(f"n_recovered_rooms: {len(rooms)}")
        for room in rooms:
            nodes = room["nodes"]
            ring = " -> ".join(map(str, nodes + [nodes[0]]))
            print(f"{room['room_id']}: nodes={nodes}  ring={ring}")


if __name__ == "__main__":
    main()
