"""Backfill gt_adj_matrix into tri inference JSONL outputs.

The input inference JSONL may already contain prompt, generated adjacency,
GT coordinates/types, and predicted coordinates, but miss gt_adj_matrix. This
script rematches each row against the original validation JSONL and copies the
validation adj_matrix into gt_adj_matrix.

Matching is intentionally conservative:
  1. source_index is used when present because it is an explicit val-row id.
  2. Otherwise, match by stable GT content keys, not by row number.
  3. Ambiguous or missing matches are skipped unless --strict is set.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True, help="Inference JSONL to patch")
    p.add_argument("--val_jsonl", default="data/jsonl/graph_160k_spatial_val.jsonl")
    p.add_argument(
        "--out",
        default=None,
        help="Output JSONL with gt_adj_matrix. Default: overwrite --in_jsonl in place.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row cannot be matched uniquely",
    )
    p.add_argument(
        "--write_source_index",
        action="store_true",
        help="Also write source_index for rows matched by GT content",
    )
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_of(row: Dict[str, Any]) -> str:
    # infer.py normalizes prompt the same way before writing, so compare in that
    # normalized space instead of relying on byte-identical whitespace.
    return str(row.get("prompt", row.get("text", ""))).replace("\n", " ").strip()


def canonical(obj: Any) -> str:
    return json.dumps(_canonical_obj(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonical_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical_obj(v) for v in obj]
    if isinstance(obj, float):
        rounded = round(obj)
        if abs(obj - rounded) < 1e-6:
            return int(rounded)
        return round(obj, 6)
    return obj


def coords_of(row: Dict[str, Any]) -> Any:
    return row.get("gt_node_coords", row.get("node_coords"))


def types_of(row: Dict[str, Any]) -> Any:
    return row.get("gt_node_types", row.get("node_types"))


Key = Tuple[str, str, str, str]


def make_keys(row: Dict[str, Any]) -> Iterable[Tuple[str, Key]]:
    prompt = prompt_of(row)
    coords = coords_of(row)
    types = types_of(row)

    if prompt and coords is not None and types is not None:
        yield ("prompt_coords_types", ("pct", prompt, canonical(coords), canonical(types)))
    if prompt and coords is not None:
        yield ("prompt_coords", ("pc", prompt, canonical(coords), ""))
    if coords is not None and types is not None:
        yield ("coords_types", ("ct", canonical(coords), canonical(types), ""))


def build_indexes(val_rows: List[Dict[str, Any]]) -> Dict[str, Dict[Key, List[int]]]:
    indexes: Dict[str, Dict[Key, List[int]]] = {
        "prompt_coords_types": defaultdict(list),
        "prompt_coords": defaultdict(list),
        "coords_types": defaultdict(list),
    }
    for idx, row in enumerate(val_rows):
        for name, key in make_keys(row):
            indexes[name][key].append(idx)
    return indexes


def find_match(
    row: Dict[str, Any],
    val_rows: List[Dict[str, Any]],
    indexes: Dict[str, Dict[Key, List[int]]],
) -> Tuple[Optional[int], str]:
    source_index = row.get("source_index")
    if source_index is not None:
        try:
            idx = int(source_index)
        except (TypeError, ValueError):
            return None, "bad_source_index"
        if 0 <= idx < len(val_rows):
            return idx, "source_index"
        return None, "bad_source_index"

    saw_ambiguous = False
    for name, key in make_keys(row):
        matches = indexes[name].get(key, [])
        if len(matches) == 1:
            return matches[0], name
        if len(matches) > 1:
            saw_ambiguous = True
    return None, "ambiguous" if saw_ambiguous else "no_match"


def main() -> None:
    args = parse_args()
    val_rows = read_jsonl(args.val_jsonl)
    indexes = build_indexes(val_rows)

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out) if args.out else in_path
    write_path = (
        out_path.with_name(f"{out_path.name}.tmp")
        if out_path.resolve() == in_path.resolve()
        else out_path
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = matched = skipped = 0
    reasons: Dict[str, int] = defaultdict(int)
    with in_path.open(encoding="utf-8") as fin, write_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            idx, reason = find_match(row, val_rows, indexes)
            reasons[reason] += 1
            if idx is None:
                skipped += 1
                if args.strict:
                    raise RuntimeError(f"row {total - 1}: failed to match validation row ({reason})")
                continue

            patched = dict(row)
            patched["gt_adj_matrix"] = val_rows[idx]["adj_matrix"]
            if args.write_source_index and "source_index" not in patched:
                patched["source_index"] = idx
            fout.write(json.dumps(patched, ensure_ascii=False) + "\n")
            matched += 1

    print(f"read inference rows: {total}")
    print(f"matched: {matched}")
    print(f"skipped: {skipped}")
    for reason, count in sorted(reasons.items()):
        print(f"{reason}: {count}")
    if write_path != out_path:
        write_path.replace(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
