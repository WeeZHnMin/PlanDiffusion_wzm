"""
构建最终图结构数据集（v2）：支持多批次 mapping + multi-caption 文件。

输入（两批，各自独立处理）：
  data/viz_50000/mapping.jsonl          + data/jsonl/viz_50000_captions_multi.jsonl
  data/viz_100000/mapping.jsonl         + data/jsonl/viz_100000_captions_multi.jsonl
  data/Architext_v1/train_jsonl/        （两批共用）

输出：
  data/jsonl/final_graph_dataset_v2.jsonl
  data/processed/type_combo_vocab_v2.json

模型优先级：
  每张图从多个模型的描述中选质量最高的一条（跳过 OCR 类模型）。

用法：
  python data/scripts/unified_dataset/build_final_graph_dataset_v2.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

# ── 模型优先级（值越小越优先）────────────────────────────────────────────────
MODEL_RANK: dict[str, int] = {
    "qwen3.5-397b-a17b":          0,
    "qwen3.5-122b-a10b":          1,
    "qwen3-235b-a22b":            2,
    "qwen3.6-35b-a3b":            3,
    "qwen3.5-35b-a3b":            4,
    "qwen3.6-max-preview":        5,
    "qwen3-max":                  6,
    "qwen3-max-2026-01-23":       7,
    "qwen3-max-2025-09-23":       8,
    "qwen3-max-preview":          9,
    "deepseek-v3.2":             10,
    "deepseek-v3.1":             11,
    "deepseek-v3":               12,
    "Moonshot-Kimi-K2-Instruct": 13,
    "kimi-k2.6":                 14,
    "kimi-k2.5":                 15,
    "qwen3.6-plus":              16,
    "qwen3.6-plus-2026-04-02":   17,
    "qwen3.5-plus":              18,
    "qwen3.5-plus-2026-04-20":   19,
    "qwen3.5-plus-2026-02-15":   20,
    "qwen3.6-27b":               21,
    "qwen3.5-27b":               22,
    "qwen3-32b":                 23,
    "qwen3-30b-a3b":             24,
    "qwen3.6-flash":             25,
    "qwen3.6-flash-2026-04-16":  26,
    "qwen3.5-flash":             27,
    "qwen3.5-flash-2026-02-23":  28,
    "qwen3-vl-235b-a22b-instruct": 29,
    "qwen3-vl-32b-instruct":     30,
    "qwen3-vl-30b-a3b-instruct": 31,
    "qwen3-vl-plus":             32,
    "qwen3-vl-plus-2025-12-19":  33,
    "qwen3-vl-plus-2025-09-23":  34,
    "qwen3-vl-flash":            35,
    "qwen3-vl-flash-2026-01-22": 36,
    "qwen3-vl-flash-2025-10-15": 37,
    "qwen3-vl-8b-instruct":      38,
    "gui-plus":                  39,
    "gui-plus-2026-02-26":       40,
    "tongyi-xiaomi-analysis-pro":41,
    "qwen-vl-max":               42,
    "qwen-vl-plus":              43,
}

# OCR 类模型跳过（输出是 OCR 识别结果，不是描述）
def is_ocr_model(model: str) -> bool:
    return "ocr" in model.lower()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", default=[
        "data/viz_50000/mapping.jsonl:data/jsonl/viz_50000_captions_multi.jsonl",
        "data/viz_100000/mapping.jsonl:data/jsonl/viz_100000_captions_multi.jsonl",
    ], help="mapping:captions 对，用冒号分隔")
    p.add_argument("--src-dir",  default="data/Architext_v1/train_jsonl")
    p.add_argument("--output",   default="data/jsonl/final_graph_dataset_v2.jsonl")
    p.add_argument("--vocab-output", default="data/processed/type_combo_vocab_v2.json")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ── 工具函数（与 v1 相同）──────────────────────────────────────────────────────
MAX_NODES = 40
ROOM_TYPE_ORDER = ["bathroom","bedroom","living_room","kitchen","corridor","dining_room","other"]


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
            coord_to_types[tuple(coord)].add(room["type"])
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
    return [[int(round(x - cx)), int(round(y - cy))] for x, y in coords]


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
    TOK_OPEN = MAX_NODES + 1
    TOK_CLOSE = MAX_NODES + 2
    TOK_BREAK = MAX_NODES + 3
    BOS_ID = MAX_NODES + 4
    EOS_ID = MAX_NODES + 5
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
            json.dumps(list(combo), ensure_ascii=False): cid
            for combo, cid in sorted(combo_to_id.items(), key=lambda x: x[1])
        },
        "id_to_combo": {
            str(cid): list(combo)
            for combo, cid in sorted(combo_to_id.items(), key=lambda x: x[1])
        },
        "N_TYPES": max(combo_to_id.values()) if combo_to_id else 0,
        "ROOM_TYPE_ORDER": ROOM_TYPE_ORDER,
    }


# ── 加载 caption：每张图取优先级最高的模型的描述 ───────────────────────────────
def load_best_captions(captions_path: Path) -> dict[str, str]:
    """返回 {image_file: best_caption_text}"""
    best: dict[str, tuple[int, str]] = {}  # {file: (rank, caption)}
    with captions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("ok") or not row.get("caption"):
                continue
            model   = row.get("model", "")
            if is_ocr_model(model):
                continue
            rank    = MODEL_RANK.get(model, 999)
            imgfile = row["file"]
            if imgfile not in best or rank < best[imgfile][0]:
                best[imgfile] = (rank, row["caption"])
    return {k: v[1] for k, v in best.items()}


# ── 加载源数据（一次性全部读入内存，两批共用）─────────────────────────────────
def load_source_data(src_dir: Path) -> dict[tuple[str, int], dict]:
    print(f"加载源数据 {src_dir} ...")
    t0 = time.perf_counter()
    src: dict[tuple[str, int], dict] = {}
    for jsonl_file in sorted(src_dir.glob("*.jsonl")):
        with jsonl_file.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                src[(jsonl_file.name, line_no)] = json.loads(line)
    print(f"  载入 {len(src)} 条源记录，耗时 {time.perf_counter()-t0:.1f}s")
    return src


# ── 构建单条记录 ────────────────────────────────────────────────────────────────
def build_record(mapping_row, source_row, caption_text, seed_offset):
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
    return {
        "prompt":      caption_text,
        "image":       mapping_row["image"],
        "source_file": mapping_row["source_file"],
        "source_line": mapping_row["source_line"],
        "n_nodes":     n_nodes,
        "node_coords": node_coords,
        "node_types":  node_types,
        "node_mask":   node_mask,
        "adj_matrix":  adj_matrix,
        "tokens":      tokens,
        "length":      len(tokens),
    }


# ── 主流程 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    src_dir = Path(args.src_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 一次性加载所有源数据
    source_data = load_source_data(src_dir)

    total_written  = 0
    missing_cap    = 0
    missing_src    = 0
    all_records    = []

    for batch_spec in args.batches:
        mapping_path_str, captions_path_str = batch_spec.split(":", 1)
        mapping_path  = Path(mapping_path_str)
        captions_path = Path(captions_path_str)

        print(f"\n处理批次：{mapping_path.parent.name}")
        t0 = time.perf_counter()

        # 加载该批次的 caption
        captions = load_best_captions(captions_path)
        print(f"  载入 {len(captions)} 条描述（{captions_path.name}）")

        # 处理 mapping
        with mapping_path.open(encoding="utf-8") as f:
            mapping_rows = [json.loads(l) for l in f if l.strip()]

        batch_written = 0
        for idx, row in enumerate(mapping_rows):
            caption = captions.get(row["image"])
            if not caption:
                missing_cap += 1
                continue
            key = (row["source_file"], int(row["source_line"]))
            src_row = source_data.get(key)
            if src_row is None:
                missing_src += 1
                continue
            record = build_record(row, src_row, caption, args.seed + total_written + idx)
            all_records.append(record)
            batch_written += 1

            if (idx + 1) % 10000 == 0:
                print(f"  {idx+1}/{len(mapping_rows)} 已处理，写入 {batch_written} 条")

        total_written += batch_written
        print(f"  批次完成：{batch_written} 条，耗时 {time.perf_counter()-t0:.1f}s")

    # 构建 combo vocab 并写入 node_combo_ids
    print(f"\n构建 combo vocab...")
    combo_to_id = build_combo_vocab(all_records)
    for record in all_records:
        record["node_combo_ids"] = [
            combo_to_id[tuple(combo)] if combo else 0
            for combo in record["node_types"]
        ]

    # 写出
    print(f"写出 {total_written} 条记录 → {output_path}")
    with output_path.open("w", encoding="utf-8") as out:
        for record in all_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    vocab_path = Path(args.vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(serialize_vocab(combo_to_id), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n完成")
    print(f"  总写入：{total_written}")
    print(f"  缺描述：{missing_cap}")
    print(f"  缺源数据：{missing_src}")
    print(f"  vocab → {vocab_path}")


if __name__ == "__main__":
    main()
