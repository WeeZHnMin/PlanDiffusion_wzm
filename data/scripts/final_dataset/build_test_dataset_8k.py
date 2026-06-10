"""
Extract 8,000 test samples from viz_50000 (zero overlap with 150k training set).

Steps:
  1. Load viz_50000/mapping.jsonl  →  image → {source_file, source_line}
  2. Load viz_50000_captions_multi_en.jsonl  →  image → best caption (ok=True)
  3. Load Architext_v1 source rows for all referenced (source_file, source_line) pairs
  4. Load existing type_combo_vocab.json so test IDs are consistent with training
  5. Build records identical in format to final_graph_dataset_v3.jsonl
  6. Filter: drop if WP-tokenized prompt > 224 tokens
  7. Randomly sample 8,000 records (seed=0) and write to test_graph_dataset_8k.jsonl

Usage:
    python data/scripts/final_dataset/build_test_dataset_8k.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

MAX_NODES = 40
MAX_TEXT_LEN = 224

ROOM_TYPE_ORDER = [
    "bathroom",
    "bedroom",
    "living_room",
    "kitchen",
    "corridor",
    "dining_room",
    "other",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping",   default="data/viz_50000/mapping.jsonl")
    parser.add_argument("--captions",  default="data/jsonl/translated_en/viz_50000_captions_multi_en.jsonl")
    parser.add_argument("--src-dir",   default="data/Architext_v1/train_jsonl")
    parser.add_argument("--vocab",     default="data/processed/type_combo_vocab.json")
    parser.add_argument("--tokenizer", default="node_diffusion/unified_vocab_wp/wp_tokenizer.json")
    parser.add_argument("--output",    default="data/jsonl/test_graph_dataset_8k.jsonl")
    parser.add_argument("--n",         type=int, default=8000)
    parser.add_argument("--seed",      type=int, default=0)
    return parser.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def load_mapping(path: Path):
    rows = []
    wanted = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_line"] = int(row["source_line"])
            rows.append(row)
            wanted[row["source_file"]].add(row["source_line"])
    return rows, wanted


def load_captions(path: Path):
    captions = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ok") and row.get("caption"):
                captions[row["file"]] = row["caption"]
    return captions


def load_source_rows(src_dir: Path, wanted: dict):
    found = {}
    for src_file, line_numbers in wanted.items():
        src_path = src_dir / src_file
        if not src_path.exists():
            raise FileNotFoundError(f"Missing: {src_path}")
        with src_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no in line_numbers:
                    found[(src_file, line_no)] = json.loads(line)
    return found


def load_vocab(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    combo_to_id = {
        tuple(json.loads(k)): v
        for k, v in payload["combo_to_id"].items()
    }
    return combo_to_id


# ── graph conversion (mirrors build_final_graph_dataset.py) ──────────────────

def build_adj_matrix(vertex_adj, n_max):
    adj = [[0] * n_max for _ in range(n_max)]
    n = len(vertex_adj)
    for i in range(n):
        adj[i][i] = 1
        for j in vertex_adj[i]:
            adj[i][j] = 1
    return adj


def extract_node_type_combos(rooms, vertices):
    coord_to_types = defaultdict(set)
    for room in rooms:
        for coord in room["coords"]:
            coord_to_types[tuple(coord)].add(room["type"])
    combos = []
    for vertex in vertices:
        types = sorted(coord_to_types.get(tuple(vertex), set()))
        combos.append(types if types else ["other"])
    return combos


def center_node_coords(vertices):
    if not vertices:
        return []
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    return [[int(round(x - cx)), int(round(y - cy))] for x, y in vertices]


# graph-token serialisation (identical to build_final_graph_dataset.py)
PAD_ID   = 0
TOK_OPEN  = MAX_NODES + 1
TOK_CLOSE = MAX_NODES + 2
TOK_BREAK = MAX_NODES + 3
BOS_ID    = MAX_NODES + 4
EOS_ID    = MAX_NODES + 5


def sample_sent(adj, n, rng):
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                neighbors[i].add(j)
    unvisited = set(range(n))
    all_nodes  = set(range(n))
    v = rng.choice(list(unvisited))
    unvisited.remove(v)
    current_trail = [(v, set())]
    sent = []
    while unvisited:
        unvisited_nbrs = neighbors[v] & unvisited
        if not unvisited_nbrs:
            sent.append(current_trail)
            v = rng.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            current_trail = [(v, neighbors[v] & visited)]
        else:
            u = rng.choice(list(unvisited_nbrs))
            unvisited.remove(u)
            visited = all_nodes - unvisited
            current_trail.append((u, (neighbors[u] - {v}) & visited))
            v = u
    sent.append(current_trail)
    return sent


def sent_to_tokens(sent):
    tokens = []
    node_to_id = {}
    next_id = [1]

    def get_id(node):
        if node not in node_to_id:
            node_to_id[node] = next_id[0]
            next_id[0] += 1
        return node_to_id[node]

    for seg_idx, trail in enumerate(sent):
        if seg_idx > 0:
            tokens.append(TOK_BREAK)
        for node, nbrs in trail:
            v_id = get_id(node)
            nbr_ids = sorted(get_id(u) for u in nbrs)
            tokens.append(v_id)
            tokens.append(TOK_OPEN)
            tokens.extend(nbr_ids)
            tokens.append(TOK_CLOSE)
    return [BOS_ID] + tokens + [EOS_ID]


def build_record(mapping_row, source_row, caption_text, combo_to_id, seed_offset):
    vertices = source_row["vertices"]
    rooms    = source_row["rooms"]
    n_nodes  = len(vertices)

    centered_coords = center_node_coords(vertices)
    node_coords = centered_coords + [[0, 0]] * (MAX_NODES - n_nodes)
    node_mask   = [1] * n_nodes + [0] * (MAX_NODES - n_nodes)
    node_types  = extract_node_type_combos(rooms, vertices) + [[] for _ in range(MAX_NODES - n_nodes)]
    adj_matrix  = build_adj_matrix(source_row["vertex_adj"], MAX_NODES)

    adj_n = [row[:n_nodes] for row in adj_matrix[:n_nodes]]
    for i in range(n_nodes):
        adj_n[i][i] = 0

    rng    = random.Random(seed_offset)
    tokens = sent_to_tokens(sample_sent(adj_n, n_nodes, rng))

    node_combo_ids = []
    for combo in node_types:
        key = tuple(combo) if combo else ()
        node_combo_ids.append(combo_to_id.get(key, 0))

    return {
        "prompt":         caption_text,
        "image":          mapping_row["image"],
        "source_file":    mapping_row["source_file"],
        "source_line":    mapping_row["source_line"],
        "n_nodes":        n_nodes,
        "node_coords":    node_coords,
        "node_types":     node_types,
        "node_combo_ids": node_combo_ids,
        "node_mask":      node_mask,
        "adj_matrix":     adj_matrix,
        "tokens":         tokens,
        "length":         len(tokens),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading tokenizer …")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print("Loading mapping …")
    mapping_rows, wanted = load_mapping(Path(args.mapping))
    print(f"  {len(mapping_rows):,} mapping rows, {sum(len(v) for v in wanted.values()):,} unique source lines")

    print("Loading captions …")
    captions = load_captions(Path(args.captions))
    print(f"  {len(captions):,} valid captions")

    print("Loading source rows …")
    source_rows = load_source_rows(Path(args.src_dir), wanted)
    print(f"  {len(source_rows):,} source rows loaded")

    print("Loading type combo vocab …")
    combo_to_id = load_vocab(Path(args.vocab))
    print(f"  {len(combo_to_id):,} combo types")

    print("Building records …")
    all_records = []
    skip_caption = skip_source = skip_tokens = 0

    for idx, mapping_row in enumerate(mapping_rows):
        key        = (mapping_row["source_file"], mapping_row["source_line"])
        source_row = source_rows.get(key)
        if source_row is None:
            skip_source += 1
            continue

        caption_text = captions.get(mapping_row["image"])
        if not caption_text:
            skip_caption += 1
            continue

        enc = tokenizer.encode(caption_text)
        if len(enc.ids) > MAX_TEXT_LEN:
            skip_tokens += 1
            continue

        record = build_record(mapping_row, source_row, caption_text, combo_to_id, args.seed + idx)
        all_records.append(record)

    print(f"  built {len(all_records):,} valid records")
    print(f"  skipped: {skip_source} missing source, {skip_caption} missing caption, {skip_tokens} long caption")

    if len(all_records) < args.n:
        raise RuntimeError(
            f"Only {len(all_records)} valid records, need {args.n}. "
            "Lower --n or fix data paths."
        )

    print(f"Sampling {args.n:,} records (seed={args.seed}) …")
    rng = random.Random(args.seed)
    sampled = rng.sample(all_records, args.n)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(sampled):,} records → {output_path}")


if __name__ == "__main__":
    main()
