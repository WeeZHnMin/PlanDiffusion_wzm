"""
从 final_graph_dataset.jsonl 生成统一词表格式的训练数据。

输出（两个独立文件，供两阶段训练使用）：
  data/processed/unified_dataset/graph_only.npz   ← 第一阶段：纯图序列
  data/processed/unified_dataset/text_graph.npz   ← 第二阶段：文本+图序列

图序列格式（新统一ID）：
  BOS_G  node1 type1 OPEN [nbrs] CLOSE  node2 type2 ...  EOS_G

文本+图序列格式：
  bpe_token... BOS_G  node1 type1 OPEN [nbrs] CLOSE ...  EOS_G

输入：
  data/jsonl/final_graph_dataset.jsonl
  data/processed/unified_vocab/vocab_config.json
  data/processed/unified_vocab/bpe_tokenizer.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOM_TYPE_IDS = {
    "bathroom": 1, "bedroom": 2, "living_room": 3,
    "kitchen": 4, "corridor": 5, "dining_room": 6,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path,
                        default=Path("data/jsonl/final_graph_dataset.jsonl"))
    parser.add_argument("--vocab-config", type=Path,
                        default=Path("data/processed/unified_vocab/vocab_config.json"))
    parser.add_argument("--bpe-tokenizer", type=Path,
                        default=Path("data/processed/unified_vocab/bpe_tokenizer.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/unified_dataset"))
    parser.add_argument("--augment", type=int, default=5)
    parser.add_argument("--max-graph-len", type=int, default=256,
                        help="图序列最大长度（含BOS_G和EOS_G）")
    parser.add_argument("--max-text-len", type=int, default=128,
                        help="文本BPE序列最大长度（截断）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def room_type_id(name: str) -> int:
    return ROOM_TYPE_IDS.get(name, 7)


def combo_names_to_type_id(combo_names: list, combo_to_id: dict) -> int:
    ids = tuple(sorted(room_type_id(n) for n in combo_names)) or (7,)
    return combo_to_id.get(ids, combo_to_id.get((7,), 7))


def build_combo_to_id(old_vocab_path: Path) -> dict:
    """从旧vocab重建 frozenset->id 映射"""
    import ast
    old_vocab = json.loads(old_vocab_path.read_text(encoding="utf-8"))
    combo_to_id = {}
    for key_str, cid in old_vocab["combo_to_id"].items():
        key = tuple(ast.literal_eval(key_str))
        combo_to_id[key] = cid
    return combo_to_id


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


def sent_to_tokens(sent, node_combo_ids, cfg):
    """将SENT转换为新统一词表ID的图token序列（不含BOS_G/EOS_G）"""
    BOS_G    = cfg["BOS_ID"]
    TOK_OPEN = cfg["TOK_OPEN"]
    TOK_CLOSE= cfg["TOK_CLOSE"]
    TOK_BREAK= cfg["TOK_BREAK"]
    TYPE_START= cfg["TYPE_START"]   # TYPE_1 = TYPE_START+0
    NODE_START= cfg["NODE_START"]   # NODE_1 = NODE_START+0

    tokens = []
    node_to_id = {}
    next_id = [0]  # 0-based，NODE_START+0 = NODE_1
    visit_order = []

    def get_id(node_idx):
        if node_idx not in node_to_id:
            node_to_id[node_idx] = next_id[0]
            next_id[0] += 1
            visit_order.append(node_idx)
        return node_to_id[node_idx]

    for seg_idx, trail in enumerate(sent):
        if seg_idx > 0:
            tokens.append(TOK_BREAK)
        for node_idx, nbrs in trail:
            nid = get_id(node_idx)
            type_id = int(node_combo_ids[node_idx])
            nbr_ids = sorted(get_id(nei) for nei in nbrs)

            tokens.append(NODE_START + nid)           # 节点token
            tokens.append(TYPE_START + (type_id - 1)) # 类型token（type_id 1-based）
            tokens.append(TOK_OPEN)
            tokens.extend(NODE_START + nid2 for nid2 in nbr_ids)
            tokens.append(TOK_CLOSE)

    return tokens, visit_order


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.vocab_config.read_text(encoding="utf-8"))
    bpe = Tokenizer.from_file(str(args.bpe_tokenizer))

    # 旧combo_to_id映射
    old_vocab_path = Path("data/processed/type_combo_vocab_old.json")
    combo_to_id = build_combo_to_id(old_vocab_path)

    BOS_G = cfg["BOS_ID"]
    EOS_G = cfg["EOS_ID"]
    PAD   = cfg["PAD_ID"]
    MAX_NODES = cfg["MAX_NODES"]

    # 图序列：最大长度含BOS_G+EOS_G
    max_graph = args.max_graph_len
    # 文本+图：文本最大128，图最大256，总计384
    max_combined = args.max_text_len + args.max_graph_len

    # 收集数据
    graph_tokens_list   = []
    graph_lengths_list  = []
    combined_tokens_list= []
    combined_lengths_list=[]
    text_lens_list      = []
    n_nodes_list        = []
    truncated           = 0

    random.seed(args.seed)
    np.random.seed(args.seed)

    with args.jsonl.open("r", encoding="utf-8") as f:
        for rec_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            n_nodes = int(record["n_nodes"])
            adj = np.array(record["adj_matrix"], dtype=np.float32)[:n_nodes, :n_nodes].copy()
            np.fill_diagonal(adj, 0)

            # 获取节点类型ID（基于旧combo_to_id）
            combo_ids = []
            for combo_names in record["node_types"][:n_nodes]:
                ids = tuple(sorted(room_type_id(n) for n in combo_names)) or (7,)
                combo_ids.append(combo_to_id.get(ids, combo_to_id.get((7,), 7)))
            combo_ids = np.array(combo_ids, dtype=np.int32)

            # 编码提示词
            prompt = record.get("prompt", "").replace("\n", " ").strip()
            text_ids = bpe.encode(prompt).ids[:args.max_text_len]

            for aug_idx in range(args.augment):
                rng = random.Random(args.seed + rec_idx * 1000 + aug_idx)
                sent = sample_sent(adj, n_nodes, rng)
                graph_body, _ = sent_to_tokens(sent, combo_ids, cfg)

                # ── 图序列：BOS_G + graph_body + EOS_G ──
                g_seq = [BOS_G] + graph_body + [EOS_G]
                if len(g_seq) > max_graph:
                    g_seq = g_seq[:max_graph]
                    truncated += 1
                g_len = len(g_seq)
                g_padded = np.full(max_graph, PAD, dtype=np.int32)
                g_padded[:g_len] = g_seq

                # ── 文本+图序列：text_ids + BOS_G + graph_body + EOS_G ──
                c_seq = text_ids + [BOS_G] + graph_body + [EOS_G]
                if len(c_seq) > max_combined:
                    c_seq = c_seq[:max_combined]
                c_len = len(c_seq)
                c_padded = np.full(max_combined, PAD, dtype=np.int32)
                c_padded[:c_len] = c_seq

                graph_tokens_list.append(g_padded)
                graph_lengths_list.append(g_len)
                combined_tokens_list.append(c_padded)
                combined_lengths_list.append(c_len)
                text_lens_list.append(len(text_ids))
                n_nodes_list.append(n_nodes)

            if (rec_idx + 1) % 5000 == 0:
                print(f"处理 {rec_idx + 1} 张图 -> {len(graph_tokens_list)} 条序列")

    print(f"\n总序列数: {len(graph_tokens_list)}")
    print(f"截断序列: {truncated}")

    g_lengths = np.array(graph_lengths_list)
    print(f"图序列长度: min={g_lengths.min()} max={g_lengths.max()} mean={g_lengths.mean():.1f}")

    # 保存第一阶段数据（纯图序列）
    graph_out = args.output_dir / "graph_only.npz"
    np.savez_compressed(
        graph_out,
        tokens  = np.array(graph_tokens_list,  dtype=np.int32),
        lengths = np.array(graph_lengths_list, dtype=np.int32),
        n_nodes = np.array(n_nodes_list,       dtype=np.int32),
    )
    print(f"第一阶段数据 -> {graph_out}")

    # 保存第二阶段数据（文本+图序列）
    combined_out = args.output_dir / "text_graph.npz"
    np.savez_compressed(
        combined_out,
        tokens    = np.array(combined_tokens_list,  dtype=np.int32),
        lengths   = np.array(combined_lengths_list, dtype=np.int32),
        text_lens = np.array(text_lens_list,        dtype=np.int32),
        n_nodes   = np.array(n_nodes_list,          dtype=np.int32),
    )
    print(f"第二阶段数据 -> {combined_out}")


if __name__ == "__main__":
    main()
