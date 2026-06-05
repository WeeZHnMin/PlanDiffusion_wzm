"""
修复 graph_dataset.npz 中错误的 node_combo_ids。

问题原因：
  build_final_graph_dataset.py 中 extract_node_type_combos 按字母序排序
  房间类型名，导致 ('corridor', 'living_room') 无法匹配 vocab 中的
  ('living_room', 'corridor')，fallback 成 7（other）。

修复方案：
  直接读 final_graph_dataset_v2.jsonl 中已存的 node_types（字符串列表），
  用正确的 ROOM_TYPE_ORDER 排序后重新查 vocab，重建 graph_dataset.npz。
  无需重建 JSONL，无需源数据。

用法：
  python -m data.scripts.build_v2.fix_combo_ids_npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOM_TYPE_ORDER = ["bathroom","bedroom","living_room","kitchen","corridor","dining_room","other"]
_ORDER_MAP = {name: idx for idx, name in enumerate(ROOM_TYPE_ORDER)}

MAX_NODES    = 40
MAX_TEXT_LEN = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",    default="data/jsonl/final_graph_dataset_v2.jsonl")
    p.add_argument("--vocab",    default="data/processed/type_combo_vocab_old.json")
    p.add_argument("--bpe",      default="node_diffusion/unified_vocab_wp/wp_tokenizer.json")
    p.add_argument("--output",   default="data/processed/node_diffusion/graph_dataset.npz")
    p.add_argument("--augment",  type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def load_combo_vocab(vocab_path: Path) -> dict[tuple, int]:
    raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    combo_to_id = {}
    import ast
    name_map = {int(k): v for k, v in raw["base_type_names"].items()}
    for key_str, cid in raw["combo_to_id"].items():
        bases = ast.literal_eval(key_str)
        # vocab key 按数字 ID 顺序，与 ROOM_TYPE_ORDER 一致
        combo_names = tuple(name_map[b] for b in bases)
        combo_to_id[combo_names] = cid
    print(f"载入 vocab: {len(combo_to_id)} 种 combo")
    return combo_to_id


def types_to_combo_id(types: list[str], combo_to_id: dict) -> int:
    if not types:
        return 0
    # 按 ROOM_TYPE_ORDER 排序（与 vocab key 顺序一致）
    sorted_types = tuple(sorted(types, key=lambda x: _ORDER_MAP.get(x, 99)))
    return combo_to_id.get(sorted_types, combo_to_id.get(("other",), 7))


def permute_graph(adj, combo_ids, coords, n, perm):
    full_perm = perm + list(range(n, MAX_NODES))
    return (adj[np.ix_(full_perm, full_perm)],
            combo_ids[full_perm],
            coords[full_perm])


def main():
    import random
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    combo_to_id = load_combo_vocab(Path(args.vocab))
    bpe = Tokenizer.from_file(args.bpe)

    adj_list    = []
    mask_list   = []
    ids_list    = []
    coords_list = []
    ptok_list   = []
    plen_list   = []
    nnodes_list = []

    fixed = 0
    total_nodes = 0
    t0 = time.perf_counter()

    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            prompt = rec.get("prompt", "").replace("\n", " ").strip()
            if len(bpe.encode(prompt).ids) > MAX_TEXT_LEN:
                continue

            n = int(rec["n_nodes"])
            adj_full = np.array(rec["adj_matrix"], dtype=np.int32)
            np.fill_diagonal(adj_full, 0)

            # 重新计算 node_combo_ids（用正确排序）
            node_types_raw = rec["node_types"][:MAX_NODES]
            new_combo_ids  = np.zeros(MAX_NODES, dtype=np.int32)
            old_combo_ids  = rec["node_combo_ids"][:MAX_NODES]
            for i, types in enumerate(node_types_raw):
                if not types:
                    new_combo_ids[i] = 0
                    continue
                new_id = types_to_combo_id(types, combo_to_id)
                new_combo_ids[i] = new_id
                if i < n:
                    total_nodes += 1
                    if new_id != old_combo_ids[i]:
                        fixed += 1

            coords = np.zeros((MAX_NODES, 2), dtype=np.int32)
            raw_coords = rec["node_coords"][:MAX_NODES]
            coords[:len(raw_coords)] = raw_coords

            mask = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n] = 1

            text_ids = bpe.encode(prompt).ids[:MAX_TEXT_LEN]
            text_len = len(text_ids)
            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            padded[:text_len] = text_ids

            base_perm = list(range(n))
            perms = [base_perm]
            for _ in range(args.augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_ids, new_coords = permute_graph(
                    adj_full, new_combo_ids, coords, n, perm)
                adj_list.append(new_adj)
                mask_list.append(mask)
                ids_list.append(new_ids)
                coords_list.append(new_coords)
                ptok_list.append(padded)
                plen_list.append(text_len)
                nnodes_list.append(n)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 条  修复节点 {fixed}/{total_nodes} ({fixed/max(total_nodes,1)*100:.2f}%)  {elapsed:.1f}s")

    print(f"\n处理完成: {len(adj_list)} 条记录")
    print(f"修复 combo_id: {fixed}/{total_nodes} 个节点 ({fixed/max(total_nodes,1)*100:.2f}%)")
    print("保存 NPZ...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        adj_matrix     = np.stack(adj_list,    axis=0),
        node_mask      = np.stack(mask_list,   axis=0),
        node_combo_ids = np.stack(ids_list,    axis=0),
        node_coords    = np.stack(coords_list, axis=0),
        prompt_tokens  = np.stack(ptok_list,   axis=0),
        prompt_lens    = np.array(plen_list,   dtype=np.int32),
        n_nodes        = np.array(nnodes_list, dtype=np.int32),
    )

    # 验证修复后的分布
    arr   = np.stack(ids_list, axis=0)
    valid = arr[np.stack(mask_list).astype(bool)]
    unique, counts = np.unique(valid, return_counts=True)
    print(f"\n修复后节点类型分布（TOP10）:")
    for uid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  combo_id={uid:2d}  count={cnt:8d}  ({cnt/len(valid)*100:.2f}%)")

    print(f"\n保存 → {out_path}  耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
