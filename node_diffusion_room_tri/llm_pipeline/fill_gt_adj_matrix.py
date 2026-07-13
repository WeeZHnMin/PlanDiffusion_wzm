"""Backfill GT fields into tri LLM-pipeline inference JSONL outputs.

The input inference JSONL may already contain prompt, generated adjacency,
and predicted coordinates, but may have missing/placeholder GT fields. This
script rematches each row against the original validation JSONL and copies the
validation GT fields back into the inference JSONL.

Matching uses normalized prompt text only.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
        help="Output JSONL with GT fields. Default: overwrite --in_jsonl in place.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row cannot be matched uniquely",
    )
    p.add_argument(
        "--keep_unmatched",
        action="store_true",
        help="Keep unmatched/ambiguous rows unchanged. Default: drop them.",
    )
    p.add_argument(
        "--image_dir",
        default=None,
        help="Optional image directory to rebuild so 00000.png-style files match filtered JSONL rows.",
    )
    p.add_argument(
        "--image_ext",
        default=".png",
        help="Image extension used with --image_dir. Default: .png",
    )
    p.add_argument("--write_source_index", action="store_true", default=True)
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


Key = Tuple[str, str, str, str]


def make_keys(row: Dict[str, Any]) -> Iterable[Tuple[str, Key]]:
    prompt = prompt_of(row)

    if prompt:
        yield ("prompt", ("prompt", prompt, "", ""))


def build_indexes(val_rows: List[Dict[str, Any]]) -> Dict[str, Dict[Key, List[int]]]:
    indexes: Dict[str, Dict[Key, List[int]]] = {
        "prompt": defaultdict(list),
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
    for name, key in make_keys(row):
        matches = indexes[name].get(key, [])
        if len(matches) == 1:
            return matches[0], name
        if len(matches) > 1:
            return None, "ambiguous"
    return None, "no_match"


def main() -> None:
    args = parse_args()
    val_rows = read_jsonl(args.val_jsonl)
    indexes = build_indexes(val_rows)
    image_dir = Path(args.image_dir) if args.image_dir else None
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out) if args.out else in_path
    write_path = (
        out_path.with_name(f"{out_path.name}.tmp")
        if out_path.resolve() == in_path.resolve()
        else out_path
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = matched = kept_unmatched = dropped_unmatched = 0
    image_pairs: List[Tuple[int, int]] = []
    reasons: Dict[str, int] = defaultdict(int)
    with in_path.open(encoding="utf-8") as fin, write_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row_index = total
            total += 1
            row = json.loads(line)
            idx, reason = find_match(row, val_rows, indexes)
            reasons[reason] += 1
            if idx is None:
                if args.strict:
                    raise RuntimeError(f"row {total - 1}: failed to match validation row ({reason})")
                if args.keep_unmatched:
                    new_image_index = len(image_pairs)
                    patched = dict(row)
                    patched["image_index"] = new_image_index
                    patched["original_image_index"] = row_index
                    fout.write(json.dumps(patched, ensure_ascii=False) + "\n")
                    image_pairs.append((row_index, new_image_index))
                    kept_unmatched += 1
                else:
                    dropped_unmatched += 1
                continue

            patched = dict(row)
            new_image_index = len(image_pairs)
            patched["image_index"] = new_image_index
            patched["original_image_index"] = row_index
            patched["gt_n_nodes"] = int(val_rows[idx]["n_nodes"])
            patched["gt_adj_matrix"] = val_rows[idx]["adj_matrix"]
            patched["gt_node_coords"] = val_rows[idx].get("node_coords")
            patched["gt_node_types"] = val_rows[idx].get("node_types")
            if args.write_source_index:
                patched["source_index"] = idx
            fout.write(json.dumps(patched, ensure_ascii=False) + "\n")
            image_pairs.append((row_index, new_image_index))
            matched += 1

    print(f"read inference rows: {total}")
    failed = kept_unmatched + dropped_unmatched
    print(f"backfill_success: {matched}")
    print(f"backfill_failed: {failed}")
    print(f"matched: {matched}")
    print(f"kept_unmatched: {kept_unmatched}")
    print(f"dropped_unmatched: {dropped_unmatched}")
    for reason, count in sorted(reasons.items()):
        print(f"{reason}: {count}")
    if write_path != out_path:
        write_path.replace(out_path)
    if image_dir is not None:
        synced_images, missing_images = sync_image_dir(image_dir, image_ext, image_pairs)
        print(f"synced_images: {synced_images}")
        print(f"missing_images: {missing_images}")
    print(f"saved -> {out_path}")


def sync_image_dir(image_dir: Path, image_ext: str, image_pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Rebuild image_dir so image filenames align with filtered JSONL row order."""
    image_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = image_dir.with_name(f"{image_dir.name}.tmp_sync")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    synced = missing = 0
    for original_index, new_index in image_pairs:
        src = image_dir / f"{original_index:05d}{image_ext}"
        dst = tmp_dir / f"{new_index:05d}{image_ext}"
        if src.exists():
            shutil.copy2(src, dst)
            synced += 1
        else:
            missing += 1

    for old_image in image_dir.glob(f"*{image_ext}"):
        old_image.unlink()
    for new_image in tmp_dir.glob(f"*{image_ext}"):
        shutil.move(str(new_image), str(image_dir / new_image.name))
    tmp_dir.rmdir()
    return synced, missing


if __name__ == "__main__":
    main()
