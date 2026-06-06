"""
构建最终图结构数据集（v2）：支持多批次 mapping + multi-caption 文件。

输入（两批，各自独立处理）：
  data/viz_50000/mapping.jsonl          + data/jsonl/viz_50000_captions_multi.jsonl
  data/viz_100000/mapping.jsonl         + data/jsonl/viz_100000_captions_multi.jsonl
  data/Architext_v1/train_jsonl/        （两批共用）
  data/processed/type_combo_vocab_old.json  （固定 combo→ID 映射，32种类型）

输出：
  data/jsonl/final_graph_dataset_v2.jsonl
    每条记录包含 node_combo_ids（整数列表，对应32种combo类型ID）

模型优先级：
  每张图从多个模型的描述中选质量最高的一条（跳过 OCR 类模型）。

用法：
  python -m data.scripts.build_v2.build_final_graph_dataset
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
    # OCR 模型优先级最低，仅在无其他模型时使用
    "qwen-vl-ocr":              100,
    "qwen-vl-ocr-latest":       101,
    "qwen-vl-ocr-1028":         102,
    "qwen-vl-ocr-2025-04-13":   103,
    "qwen-vl-ocr-2025-08-28":   104,
    "qwen-vl-ocr-2025-11-20":   105,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", default=[
        "data/viz_50000/mapping.jsonl:data/jsonl/translated_en/viz_50000_captions_multi_en.jsonl",
        "data/viz_100000/mapping.jsonl:data/jsonl/translated_en/viz_100000_captions_multi_en.jsonl",
    ], help="mapping:captions 对，用冒号分隔")
    p.add_argument("--src-dir",   default="data/Architext_v1/train_jsonl")
    p.add_argument("--combo-vocab", default="data/processed/type_combo_vocab_old.json",
                   help="固定 combo→ID 映射文件（旧 vocab，32种类型）")
    p.add_argument("--output",    default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def load_combo_vocab(vocab_path: Path) -> dict[tuple, int]:
    """从 type_combo_vocab_old.json 加载固定的 combo→ID 映射"""
    import ast
    raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    combo_to_id = {}
    for key_str, cid in raw["combo_to_id"].items():
        bases = ast.literal_eval(key_str)   # "[1, 2]" → [1, 2]
        # 转换：数字ID → 房间类型名称
        id_to_name = {v: k for k, v in raw["base_type_names"].items()}
        # base_type_names 里 key 是字符串数字
        name_map = {int(k): v for k, v in raw["base_type_names"].items()}
        combo_names = tuple(name_map[b] for b in bases)
        combo_to_id[combo_names] = cid
    print(f"载入 combo vocab: {len(combo_to_id)} 种组合类型（共32种）")
    return combo_to_id


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


_ORDER_MAP = {name: idx for idx, name in enumerate(ROOM_TYPE_ORDER)}

def extract_node_type_combos(rooms, vertices, tol=0.5):
    """
    计算每个节点的 combo 类型：
    1. 多边形顶点归属：节点坐标出现在哪些房间的顶点列表里
    2. 墙段穿越检测：节点落在哪些房间的墙段内部（非端点）
    两者取并集，确保共享墙上的中间节点类型完整。
    """
    import numpy as _np
    coord_to_types = defaultdict(set)

    # Step 1: 多边形顶点归属
    for room in rooms:
        for coord in room["coords"]:
            coord_to_types[tuple(coord)].add(room["type"])

    # Step 2: 墙段穿越检测
    verts_arr = [_np.array(v, dtype=float) for v in vertices]
    for room in rooms:
        cs = room["coords"]
        nr = len(cs)
        for k in range(nr):
            p1 = _np.array(cs[k], dtype=float)
            p2 = _np.array(cs[(k + 1) % nr], dtype=float)
            seg_len = float(_np.linalg.norm(p2 - p1))
            if seg_len < 1:
                continue
            for vi, pt in enumerate(verts_arr):
                t = float(_np.dot(pt - p1, p2 - p1)) / seg_len ** 2
                if 1e-6 < t < 1 - 1e-6:
                    dist = float(_np.linalg.norm(pt - (p1 + t * (p2 - p1))))
                    if dist <= tol:
                        coord_to_types[tuple(vertices[vi])].add(room["type"])

    combos = []
    for vertex in vertices:
        types = sorted(coord_to_types.get(tuple(vertex), set()),
                       key=lambda x: _ORDER_MAP.get(x, 99))
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
def build_record(mapping_row, source_row, caption_text, seed_offset, combo_to_id):
    vertices = source_row["vertices"]
    rooms    = source_row["rooms"]
    n_nodes  = len(vertices)
    centered_coords = center_node_coords(vertices)
    node_coords = centered_coords + [[0, 0]] * (MAX_NODES - n_nodes)
    node_mask   = [1] * n_nodes + [0] * (MAX_NODES - n_nodes)
    node_types  = extract_node_type_combos(rooms, vertices)
    adj_matrix  = build_adj_matrix(source_row["vertex_adj"], MAX_NODES)

    # 直接用旧 vocab 映射为 ID，找不到则 fallback 到 other(7)
    other_id = combo_to_id.get(("other",), 7)
    node_combo_ids = [combo_to_id.get(tuple(t), other_id) for t in node_types]
    node_combo_ids += [0] * (MAX_NODES - n_nodes)   # 填充位用 0

    adj_n = [row[:n_nodes] for row in adj_matrix[:n_nodes]]
    for i in range(n_nodes):
        adj_n[i][i] = 0
    rng    = random.Random(seed_offset)
    tokens = sent_to_tokens(sample_sent(adj_n, n_nodes, rng))
    return {
        "prompt":         caption_text,
        "image":          mapping_row["image"],
        "source_file":    mapping_row["source_file"],
        "source_line":    mapping_row["source_line"],
        "n_nodes":        n_nodes,
        "node_coords":    node_coords,
        "node_types":     node_types + [[] for _ in range(MAX_NODES - n_nodes)],
        "node_combo_ids": node_combo_ids,
        "node_mask":      node_mask,
        "adj_matrix":     adj_matrix,
        "tokens":         tokens,
        "length":         len(tokens),
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

    # 加载固定 combo vocab（旧 vocab，32种，ID 与旧数据完全一致）
    combo_to_id = load_combo_vocab(Path(args.combo_vocab))

    total_written  = 0
    missing_cap    = 0
    missing_src    = 0
    all_records    = []

    for batch_spec in args.batches:
        if ":" not in batch_spec:
            raise SystemExit(
                f"Invalid --batches item: {batch_spec}\n"
                "Expected format: mapping_path:captions_path"
            )
        mapping_path_str, captions_path_str = batch_spec.split(":", 1)
        mapping_path  = Path(mapping_path_str)
        captions_path = Path(captions_path_str)

        if not mapping_path.exists():
            raise SystemExit(f"Mapping file not found: {mapping_path}")
        if not captions_path.exists():
            raise SystemExit(f"Captions file not found: {captions_path}")

        print(f"\n处理批次：{mapping_path.parent.name}")
        t0 = time.perf_counter()

        captions = load_best_captions(captions_path)
        print(f"  载入 {len(captions)} 条描述（{captions_path.name}）")

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
            record = build_record(row, src_row, caption, args.seed + total_written + idx, combo_to_id)
            all_records.append(record)
            batch_written += 1

            if (idx + 1) % 10000 == 0:
                print(f"  {idx+1}/{len(mapping_rows)} 已处理，写入 {batch_written} 条")

        total_written += batch_written
        print(f"  批次完成：{batch_written} 条，耗时 {time.perf_counter()-t0:.1f}s")

    # ── 写出数据集 ──────────────────────────────────────────────────────────────
    print(f"\n写出 {total_written} 条记录 → {output_path}")
    with output_path.open("w", encoding="utf-8") as out:
        for rec in all_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n完成")
    print(f"  总写入：{total_written}")
    print(f"  缺描述：{missing_cap}")
    print(f"  缺源数据：{missing_src}")
    print(f"  combo vocab：{args.combo_vocab}（{len(combo_to_id)} 种类型）")


if __name__ == "__main__":
    main()
