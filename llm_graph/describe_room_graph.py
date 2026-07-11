"""
Generate per-sample room-node graph descriptions from graph JSONL files.

Example:
  python -m llm_graph.describe_room_graph \
    --jsonl data/jsonl/val_graph_dataset_18k5.jsonl \
    --out data/jsonl/val_graph_dataset_18k5_room_graph_desc_5k.jsonl \
    --mode node_edges
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/jsonl/val_graph_dataset_18k5.jsonl",
                   help="Input graph JSONL path.")
    p.add_argument("--out", default="data/jsonl/val_graph_dataset_18k5_room_graph_desc_5k.jsonl",
                   help="Output JSONL path.")
    p.add_argument("--field", default="room_graph_description",
                   help="Output field name for the generated description.")
    p.add_argument("--mode", choices=["stats", "node_edges"], default="node_edges",
                   help="Description style: aggregate stats or explicit node/edge text.")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="Maximum rows to process, 0 means all rows.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used when sampling rows.")
    p.add_argument("--sequential", action="store_true",
                   help="Use the first max_samples rows instead of random sampling.")
    p.add_argument("--keep_original", action="store_true",
                   help="Write the original record plus the generated field. Default writes a compact record.")
    p.add_argument("--include_edge_counts", action="store_true",
                   help="Also describe graph-edge counts between room node groups.")
    return p.parse_args()


def normalize_room_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def active_node_types(record: dict) -> List[Tuple[str, ...]]:
    n_nodes = int(record.get("n_nodes", 0))
    mask = record.get("node_mask") or []
    raw_types = record.get("node_types") or []
    out = []
    for i, types in enumerate(raw_types[:n_nodes]):
        if mask and not mask[i]:
            continue
        if isinstance(types, str):
            types = [types]
        clean = tuple(sorted({normalize_room_name(t) for t in types if str(t).strip()}))
        if clean:
            out.append(clean)
    return out


def room_order(node_types: Sequence[Tuple[str, ...]]) -> List[str]:
    counts = Counter()
    first_seen = {}
    for idx, types in enumerate(node_types):
        for room in types:
            counts[room] += 1
            first_seen.setdefault(room, idx)
    return sorted(counts, key=lambda r: (first_seen[r], r))


def room_node_labels(node_types: Sequence[Tuple[str, ...]]) -> Dict[int, Dict[str, str]]:
    counters = Counter()
    labels = {}
    for idx, type_tuple in enumerate(node_types):
        labels[idx] = {}
        for room in type_tuple:
            counters[room] += 1
            labels[idx][room] = f"{room} A{counters[room]}"
    return labels


def edge_counts_by_room(record: dict, node_types: Sequence[Tuple[str, ...]]) -> Dict[Tuple[str, str], int]:
    adj = record.get("adj_matrix") or []
    n = len(node_types)
    counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for a in node_types[i]:
                for b in node_types[j]:
                    if a == b:
                        continue
                    key = tuple(sorted((a, b)))
                    counts[key] += 1
    return counts


def describe_record(record: dict, include_edge_counts: bool = False) -> str:
    types = active_node_types(record)
    rooms = room_order(types)
    room_sets = {room: [] for room in rooms}
    for idx, type_tuple in enumerate(types):
        type_set = set(type_tuple)
        for room in rooms:
            if room in type_set:
                room_sets[room].append(idx)

    shared_counts = {room: Counter() for room in rooms}
    independent_counts = Counter()
    for type_tuple in types:
        type_set = set(type_tuple)
        if len(type_set) == 1:
            independent_counts[next(iter(type_set))] += 1
        for room in type_set:
            for other in type_set:
                if other != room:
                    shared_counts[room][other] += 1

    edge_counts = edge_counts_by_room(record, types) if include_edge_counts else {}
    sentences = []
    for room in rooms:
        total = len(room_sets[room])
        parts = [f"{room} has {total} nodes"]
        independent = independent_counts[room]
        if independent:
            parts.append(f"{independent} independent nodes")
        else:
            parts.append("0 independent nodes")

        neighbors = []
        for other, count in sorted(shared_counts[room].items(), key=lambda x: (-x[1], x[0])):
            edge_part = ""
            if include_edge_counts:
                edge_n = edge_counts.get(tuple(sorted((room, other))), 0)
                edge_part = f" and {edge_n} graph edges"
            neighbors.append(f"{count} shared nodes with {other}{edge_part}")
        if neighbors:
            parts.append(", ".join(neighbors))
        else:
            parts.append("no shared nodes with other rooms")
        sentences.append("; ".join(parts) + ".")
    return " ".join(sentences)


def describe_record_node_edges(record: dict) -> str:
    types = active_node_types(record)
    rooms = room_order(types)
    labels = room_node_labels(types)
    adj = record.get("adj_matrix") or []

    node_sentences = []
    for room in rooms:
        room_nodes = []
        for idx, type_tuple in enumerate(types):
            if room not in type_tuple:
                continue
            others = [r for r in type_tuple if r != room]
            label = labels[idx][room]
            if others:
                other_labels = [labels[idx][other] for other in others]
                room_nodes.append(f"{label} shared with {', '.join(other_labels)}")
            else:
                room_nodes.append(f"{label} independent")
        node_sentences.append(f"{room}: " + "; ".join(room_nodes) + ".")

    edge_parts = []
    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            left_labels = [labels[i][room] for room in types[i]]
            right_labels = [labels[j][room] for room in types[j]]
            edge_parts.append(f"{'/'.join(left_labels)} - {'/'.join(right_labels)}")
    edge_text = "; ".join(edge_parts) if edge_parts else "none"
    return "Nodes: " + " ".join(node_sentences) + " Edges: " + edge_text + "."


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_jsonl(path: Path, max_samples: int, seed: int, sequential: bool) -> List[dict]:
    if max_samples <= 0:
        return list(iter_jsonl(path))
    if sequential:
        rows = []
        for rec in iter_jsonl(path):
            rows.append(rec)
            if len(rows) >= max_samples:
                break
        return rows

    rng = random.Random(seed)
    reservoir = []
    for seen, rec in enumerate(iter_jsonl(path), start=1):
        if len(reservoir) < max_samples:
            reservoir.append(rec)
            continue
        j = rng.randrange(seen)
        if j < max_samples:
            reservoir[j] = rec
    return reservoir


def main():
    args = parse_args()
    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = sample_jsonl(in_path, args.max_samples, args.seed, args.sequential)
    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in rows:
            if args.mode == "stats":
                desc = describe_record(rec, include_edge_counts=args.include_edge_counts)
            else:
                desc = describe_record_node_edges(rec)
            if args.keep_original:
                new_rec = dict(rec)
                new_rec[args.field] = desc
            else:
                new_rec = {
                    "n_nodes": rec.get("n_nodes"),
                    "prompt": rec.get("prompt"),
                    "node_coords": rec.get("node_coords"),
                    "node_types": rec.get("node_types"),
                    args.field: desc,
                }
            out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            total += 1
            if total % 10000 == 0:
                print(f"processed {total} rows")

    print(f"saved {total} rows -> {out_path}")


if __name__ == "__main__":
    main()
