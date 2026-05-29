import argparse
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOM_TYPE_IDS = {
    "bathroom": 1,
    "bedroom": 2,
    "living_room": 3,
    "kitchen": 4,
    "corridor": 5,
    "dining_room": 6,
    "other": 7,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-encode final_graph_dataset.jsonl into the old combo-token format."
    )
    parser.add_argument(
        "--input",
        default="data/jsonl/final_graph_dataset.jsonl",
        help="Path to final_graph_dataset.jsonl.",
    )
    parser.add_argument(
        "--old-vocab",
        default="data/processed/type_combo_vocab_old.json",
        help="Path to the old combo vocab json.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/graph_tokens_combo_from_final_old.npz",
        help="Output npz path.",
    )
    parser.add_argument(
        "--prompt-output",
        default="data/processed/graph_prompts_combo_from_final_old.txt",
        help="Output prompt txt path.",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=5,
        help="How many random-walk encodings to generate per graph.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for graph walk sampling.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Round-trip check token decoding against reordered node types and adjacency.",
    )
    return parser.parse_args()


def load_old_vocab(path: Path):
    vocab = json.loads(path.read_text(encoding="utf-8"))
    combo_to_id = {
        tuple(ast.literal_eval(combo_key)): combo_id
        for combo_key, combo_id in vocab["combo_to_id"].items()
    }
    return vocab, combo_to_id


def combo_names_to_old_ids(combo_names):
    ids = sorted(ROOM_TYPE_IDS.get(name, 7) for name in combo_names)
    return tuple(ids or [7])


def sample_sent(adj, n, rng):
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
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


def sent_to_old_tokens(sent, node_combo_ids, tok_open, tok_close, tok_break, node_offset):
    tokens = []
    node_to_id = {}
    next_id = [1]
    visit_order = []

    def get_id(node_idx):
        if node_idx not in node_to_id:
            node_to_id[node_idx] = next_id[0]
            next_id[0] += 1
            visit_order.append(node_idx)
        return node_to_id[node_idx]

    for seg_idx, trail in enumerate(sent):
        if seg_idx > 0:
            tokens.append(tok_break)
        for node_idx, neighbors in trail:
            node_id = get_id(node_idx)
            neighbor_ids = sorted(get_id(nei) for nei in neighbors)
            tokens.append(node_offset + node_id)
            tokens.append(int(node_combo_ids[node_idx]))
            tokens.append(tok_open)
            tokens.extend(node_offset + nei_id for nei_id in neighbor_ids)
            tokens.append(tok_close)

    return tokens, visit_order


def reorder_by_visit(visit_order, coords, combo_ids, adj, n_nodes, max_nodes):
    coords_r = np.zeros((max_nodes, 2), dtype=np.float32)
    comboids_r = np.zeros(max_nodes, dtype=np.int32)
    adj_r = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    for new_idx, old_idx in enumerate(visit_order):
        coords_r[new_idx] = coords[old_idx]
        comboids_r[new_idx] = combo_ids[old_idx]

    for ni, oi in enumerate(visit_order):
        for nj, oj in enumerate(visit_order):
            adj_r[ni, nj] = adj[oi, oj]

    return coords_r, comboids_r, adj_r


def decode_old_tokens(tokens, vocab):
    bos_id = vocab["BOS_ID"]
    eos_id = vocab["EOS_ID"]
    pad_id = 0
    tok_open = vocab["TOK_OPEN"]
    tok_close = vocab["TOK_CLOSE"]
    tok_break = vocab["TOK_BREAK"]
    node_offset = vocab["NODE_OFFSET"]
    max_nodes = vocab["MAX_NODES"]

    seq = [t for t in tokens if t not in (pad_id, bos_id, eos_id)]
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    comboids = np.zeros(max_nodes, dtype=np.int32)

    current_node = None
    prev_node = None
    waiting_type = False
    in_bracket = False
    seen_nodes = set()

    for token in seq:
        if node_offset < token <= node_offset + max_nodes:
            node_id = token - node_offset
            if in_bracket:
                if current_node is None:
                    raise ValueError("Malformed sequence: neighbor seen before node.")
                u = current_node - 1
                v = node_id - 1
                adj[u, v] = 1
                adj[v, u] = 1
                seen_nodes.add(node_id)
            else:
                if prev_node is not None:
                    u = prev_node - 1
                    v = node_id - 1
                    adj[u, v] = 1
                    adj[v, u] = 1
                current_node = node_id
                prev_node = node_id
                waiting_type = True
                seen_nodes.add(node_id)
        elif waiting_type:
            comboids[current_node - 1] = token
            waiting_type = False
        elif token == tok_open:
            in_bracket = True
        elif token == tok_close:
            in_bracket = False
        elif token == tok_break:
            current_node = None
            prev_node = None
            waiting_type = False
            in_bracket = False

    return comboids, adj, len(seen_nodes)


def validate_round_trip(tokens, comboids_expected, adj_expected, n_nodes, vocab):
    comboids_decoded, adj_decoded, n_decoded = decode_old_tokens(tokens, vocab)
    if n_decoded != n_nodes:
        raise ValueError(f"Decoded n_nodes mismatch: {n_decoded} != {n_nodes}")
    if not np.array_equal(comboids_decoded[:n_nodes], comboids_expected[:n_nodes]):
        raise ValueError("Decoded node combo ids do not match reordered combo ids.")
    if not np.array_equal(adj_decoded[:n_nodes, :n_nodes], adj_expected[:n_nodes, :n_nodes]):
        raise ValueError("Decoded adjacency does not match reordered adjacency.")


def main():
    args = parse_args()
    input_path = Path(args.input)
    old_vocab_path = Path(args.old_vocab)
    output_path = Path(args.output)
    prompt_output_path = Path(args.prompt_output)

    vocab, combo_to_id = load_old_vocab(old_vocab_path)
    max_nodes = int(vocab["MAX_NODES"])
    max_len = int(args.max_len)

    all_tokens = []
    all_lengths = []
    all_coords = []
    all_node_types = []
    all_adj = []
    all_n_nodes = []
    all_prompts = []
    truncated = 0

    with input_path.open("r", encoding="utf-8") as f:
        for rec_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_nodes = int(record["n_nodes"])
            coords = np.array(record["node_coords"][:n_nodes], dtype=np.float32)
            adj = np.array(record["adj_matrix"], dtype=np.float32)[:n_nodes, :n_nodes].copy()

            combo_ids = []
            for combo_names in record["node_types"][:n_nodes]:
                combo_key = combo_names_to_old_ids(combo_names)
                if combo_key not in combo_to_id:
                    raise KeyError(f"Combo {combo_key} is missing from old vocab.")
                combo_ids.append(combo_to_id[combo_key])
            combo_ids = np.array(combo_ids, dtype=np.int32)

            np.fill_diagonal(adj, 0)

            for aug_idx in range(args.augment):
                rng = random.Random(args.seed + rec_idx * 1000 + aug_idx)
                sent = sample_sent(adj, n_nodes, rng)
                tokens, visit_order = sent_to_old_tokens(
                    sent,
                    combo_ids,
                    vocab["TOK_OPEN"],
                    vocab["TOK_CLOSE"],
                    vocab["TOK_BREAK"],
                    vocab["NODE_OFFSET"],
                )
                full_tokens = [vocab["BOS_ID"]] + tokens + [vocab["EOS_ID"]]

                if len(full_tokens) > max_len:
                    full_tokens = full_tokens[:max_len]
                    truncated += 1

                length = len(full_tokens)
                padded = np.zeros(max_len, dtype=np.int32)
                padded[:length] = full_tokens

                coords_r, comboids_r, adj_r = reorder_by_visit(
                    visit_order, coords, combo_ids, adj, n_nodes, max_nodes
                )
                if args.validate and length < max_len:
                    validate_round_trip(full_tokens, comboids_r, adj_r, n_nodes, vocab)

                all_tokens.append(padded)
                all_lengths.append(length)
                all_coords.append(coords_r)
                all_node_types.append(comboids_r)
                all_adj.append(adj_r)
                all_n_nodes.append(n_nodes)
                all_prompts.append(record["prompt"])

            if (rec_idx + 1) % 5000 == 0:
                print(f"processed {rec_idx + 1} graphs -> {len(all_tokens)} sequences")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        tokens=np.array(all_tokens, dtype=np.int32),
        lengths=np.array(all_lengths, dtype=np.int32),
        coords=np.array(all_coords, dtype=np.float32),
        node_types=np.array(all_node_types, dtype=np.int32),
        adj_matrix=np.array(all_adj, dtype=np.float32),
        n_nodes=np.array(all_n_nodes, dtype=np.int32),
    )

    with prompt_output_path.open("w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(prompt.replace("\n", " ").replace("\r", " ") + "\n")

    lengths_arr = np.array(all_lengths, dtype=np.int32)
    print(f"written sequences: {len(all_tokens)}")
    print(f"truncated sequences: {truncated}")
    print(
        f"length stats: min={lengths_arr.min()} max={lengths_arr.max()} "
        f"mean={lengths_arr.mean():.1f}"
    )
    print(f"saved npz -> {output_path}")
    print(f"saved prompts -> {prompt_output_path}")


if __name__ == "__main__":
    main()
