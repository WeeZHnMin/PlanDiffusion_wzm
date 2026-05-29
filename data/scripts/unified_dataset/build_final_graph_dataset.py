import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MAX_NODES = 40
PAD_ID = 0
TOK_OPEN = MAX_NODES + 1
TOK_CLOSE = MAX_NODES + 2
TOK_BREAK = MAX_NODES + 3
BOS_ID = MAX_NODES + 4
EOS_ID = MAX_NODES + 5
ROOM_TYPE_ORDER = [
    "bathroom",
    "bedroom",
    "living_room",
    "kitchen",
    "corridor",
    "dining_room",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final graph dataset from source files only."
    )
    parser.add_argument(
        "--mapping",
        default="data/viz_50000/mapping.jsonl",
        help="Path to mapping.jsonl.",
    )
    parser.add_argument(
        "--captions",
        default="data/jsonl/captions_unique.jsonl",
        help="Path to captions.jsonl.",
    )
    parser.add_argument(
        "--src-dir",
        default="data/Architext_v1/train_jsonl",
        help="Directory containing original train_jsonl files.",
    )
    parser.add_argument(
        "--output",
        default="data/jsonl/final_graph_dataset.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--vocab-output",
        default="data/processed/type_combo_vocab.json",
        help="Output vocab json path aligned to final_graph_dataset.jsonl.",
    )
    parser.add_argument(
        "--core-vocab-output",
        default="core/vocab/type_combo_vocab.json",
        help="Optional mirrored vocab json path for code that still reads core/vocab.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for adjacency token sequence generation.",
    )
    return parser.parse_args()


def load_mapping(path: Path):
    rows = []
    wanted = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_line"] = int(row["source_line"])
            rows.append(row)
            wanted[row["source_file"]].add(row["source_line"])
    return rows, wanted


def load_source_rows(src_dir: Path, wanted):
    found = {}
    for src_file, line_numbers in wanted.items():
        src_path = src_dir / src_file
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source file: {src_path}")
        with src_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no not in line_numbers:
                    continue
                found[(src_file, line_no)] = json.loads(line)
    return found


def load_captions(path: Path):
    captions = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ok") and row.get("caption"):
                captions[row["file"]] = row["caption"]
    return captions


def build_adj_matrix(vertex_adj, n_max):
    n = len(vertex_adj)
    adj = [[0] * n_max for _ in range(n_max)]
    for i in range(n):
        adj[i][i] = 1
        for j in vertex_adj[i]:
            adj[i][j] = 1
    return adj


def extract_node_type_combos(rooms, vertices):
    coord_to_types = defaultdict(set)
    for room in rooms:
        for coord in room["coords"]:
            key = tuple(coord)
            coord_to_types[key].add(room["type"])

    combos = []
    for vertex in vertices:
        types = sorted(coord_to_types.get(tuple(vertex), set()))
        combos.append(types if types else ["other"])
    return combos


def center_node_coords(vertices):
    coords = [list(v) for v in vertices]
    if not coords:
        return []
    cx = sum(v[0] for v in coords) / len(coords)
    cy = sum(v[1] for v in coords) / len(coords)
    centered = []
    for x, y in coords:
        centered.append([int(round(x - cx)), int(round(y - cy))])
    return centered


def sample_sent(adj, n, rng):
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                neighbors[i].add(j)

    unvisited = set(range(n))
    all_nodes = set(range(n))
    v = rng.choice(list(unvisited))
    unvisited.remove(v)
    current_trail = [(v, set())]
    sent = []

    while unvisited:
        unvisited_neighbors = neighbors[v] & unvisited
        if not unvisited_neighbors:
            sent.append(current_trail)
            v = rng.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            current_trail = [(v, neighbors[v] & visited)]
        else:
            u = rng.choice(list(unvisited_neighbors))
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


def build_record(mapping_row, source_row, caption_text, seed_offset):
    vertices = source_row["vertices"]
    rooms = source_row["rooms"]
    n_nodes = len(vertices)

    centered_coords = center_node_coords(vertices)
    node_coords = centered_coords + [[0, 0]] * (MAX_NODES - n_nodes)
    node_mask = [1] * n_nodes + [0] * (MAX_NODES - n_nodes)
    node_types = extract_node_type_combos(rooms, vertices) + [[] for _ in range(MAX_NODES - n_nodes)]
    adj_matrix = build_adj_matrix(source_row["vertex_adj"], MAX_NODES)

    adj_n = [row[:n_nodes] for row in adj_matrix[:n_nodes]]
    for i in range(n_nodes):
        adj_n[i][i] = 0
    rng = random.Random(seed_offset)
    tokens = sent_to_tokens(sample_sent(adj_n, n_nodes, rng))

    return {
        "prompt": caption_text,
        "image": mapping_row["image"],
        "source_file": mapping_row["source_file"],
        "source_line": mapping_row["source_line"],
        "n_nodes": n_nodes,
        "node_coords": node_coords,
        "node_types": node_types,
        "node_mask": node_mask,
        "adj_matrix": adj_matrix,
        "tokens": tokens,
        "length": len(tokens),
    }


def combo_sort_key(combo):
    order_index = {name: idx for idx, name in enumerate(ROOM_TYPE_ORDER)}
    return tuple(order_index.get(name, len(ROOM_TYPE_ORDER)) for name in combo)


def build_combo_vocab(records):
    combo_to_id = {}

    for idx, room_type in enumerate(ROOM_TYPE_ORDER, start=1):
        combo_to_id[(room_type,)] = idx

    next_id = len(ROOM_TYPE_ORDER) + 1
    combos_found = set()
    for record in records:
        for combo in record["node_types"]:
            if combo:
                combos_found.add(tuple(combo))

    for combo in sorted(combos_found, key=combo_sort_key):
        if combo not in combo_to_id:
            combo_to_id[combo] = next_id
            next_id += 1

    return combo_to_id


def serialize_vocab(combo_to_id):
    return {
        "combo_to_id": {
            json.dumps(list(combo), ensure_ascii=False): combo_id
            for combo, combo_id in sorted(combo_to_id.items(), key=lambda item: item[1])
        },
        "id_to_combo": {
            str(combo_id): list(combo)
            for combo, combo_id in sorted(combo_to_id.items(), key=lambda item: item[1])
        },
        "N_TYPES": max(combo_to_id.values()) if combo_to_id else 0,
        "ROOM_TYPE_ORDER": ROOM_TYPE_ORDER,
        "PAD_ID": PAD_ID,
        "NODE_TOKEN_START": 1,
        "NODE_TOKEN_END": MAX_NODES,
        "TOK_OPEN": TOK_OPEN,
        "TOK_CLOSE": TOK_CLOSE,
        "TOK_BREAK": TOK_BREAK,
        "BOS_ID": BOS_ID,
        "EOS_ID": EOS_ID,
        "VOCAB_SIZE": EOS_ID + 1,
        "MAX_NODES": MAX_NODES,
    }


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping)
    captions_path = Path(args.captions)
    src_dir = Path(args.src_dir)
    output_path = Path(args.output)
    vocab_output_path = Path(args.vocab_output)
    core_vocab_output_path = Path(args.core_vocab_output)

    mapping_rows, wanted = load_mapping(mapping_path)
    source_rows = load_source_rows(src_dir, wanted)
    captions = load_captions(captions_path)

    final_records = []
    written = 0
    missing_caption = 0
    missing_source = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, mapping_row in enumerate(mapping_rows):
        key = (mapping_row["source_file"], mapping_row["source_line"])
        source_row = source_rows.get(key)
        if source_row is None:
            missing_source += 1
            continue

        caption_text = captions.get(mapping_row["image"])
        if not caption_text:
            missing_caption += 1
            continue

        record = build_record(mapping_row, source_row, caption_text, args.seed + idx)
        final_records.append(record)
        written += 1

    combo_to_id = build_combo_vocab(final_records)
    for record in final_records:
        record["node_combo_ids"] = [
            combo_to_id[tuple(combo)] if combo else 0
            for combo in record["node_types"]
        ]

    with output_path.open("w", encoding="utf-8") as out:
        for record in final_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    vocab_payload = json.dumps(
        serialize_vocab(combo_to_id), ensure_ascii=False, indent=2
    )

    vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_output_path.write_text(vocab_payload, encoding="utf-8")

    if core_vocab_output_path:
        core_vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
        core_vocab_output_path.write_text(vocab_payload, encoding="utf-8")

    print(f"mapping rows: {len(mapping_rows)}")
    print(f"written: {written}")
    print(f"missing source rows: {missing_source}")
    print(f"missing captions: {missing_caption}")
    print(f"saved -> {output_path}")
    print(f"saved vocab -> {vocab_output_path}")
    if core_vocab_output_path:
        print(f"mirrored vocab -> {core_vocab_output_path}")


if __name__ == "__main__":
    main()
