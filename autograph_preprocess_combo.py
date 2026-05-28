"""
预处理带节点组合类型的平面布局图数据
节点类型 = 该顶点所属的所有房间类型的组合（frozenset）
两遍处理：第一遍建组合类型词表，第二遍生成序列

输入：data/jsonl/mapped_type_data_zh.jsonl
输出：
  data/processed/graph_tokens_combo_5w.npz
  data/processed/graph_prompts_combo_5w.txt
  data/processed/type_combo_vocab.json   ← 组合词表（供训练脚本加载）
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC_FILE  = Path('data/jsonl/mapped_type_data_zh.jsonl')
OUT_NPZ   = Path('data/processed/graph_tokens_combo_5w.npz')
OUT_TXT   = Path('data/processed/graph_prompts_combo_5w.txt')
OUT_VOCAB = Path('data/processed/type_combo_vocab.json')

ROOM_TYPE_IDS = {
    'bathroom': 1, 'bedroom': 2, 'living_room': 3,
    'kitchen': 4, 'corridor': 5, 'dining_room': 6,
}
def room_type_id(s):
    return ROOM_TYPE_IDS.get(s, 7)

MAX_NODES = 40
MAX_LEN   = 256
AUGMENT   = 5
PAD_ID    = 0


# ── 从 rooms 字段重建每个顶点的类型组合 ────────────────────────
def get_vertex_combos(data):
    """
    返回 frozenset 列表，长度 = len(data['vertices'])
    每个 frozenset 是该顶点所属的所有房间类型 ID 的集合
    """
    coord_to_types = defaultdict(set)
    for room in data['rooms']:
        tid = room_type_id(room['type'])
        for coord in room['coords']:
            coord_to_types[(coord[0], coord[1])].add(tid)

    combos = []
    for v in data['vertices']:
        key = (v[0], v[1])
        types = coord_to_types.get(key)
        combos.append(frozenset(types) if types else frozenset([7]))
    return combos


# ── 第一遍：建组合词表 ──────────────────────────────────────────
def build_combo_vocab(src_file):
    """
    单类型固定为 1-7（与原版一致），多类型组合从 8 开始。
    返回 dict: frozenset -> int
    """
    combo_to_id = {frozenset([t]): t for t in range(1, 8)}
    next_id = 8

    with src_file.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for combo in get_vertex_combos(data):
                if combo not in combo_to_id:
                    combo_to_id[combo] = next_id
                    next_id += 1

    return combo_to_id


# ── 游走算法（不变）──────────────────────────────────────────────
def sample_sent(adj, n):
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                neighbors[i].add(j)

    unvisited = set(range(n))
    all_nodes  = set(range(n))
    v = random.choice(list(unvisited))
    unvisited.remove(v)
    current_trail = [(v, set())]
    sent = []

    while unvisited:
        unvisited_nbrs = neighbors[v] & unvisited
        if not unvisited_nbrs:
            sent.append(current_trail)
            v = random.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            A = neighbors[v] & visited
            current_trail = [(v, A)]
        else:
            u = random.choice(list(unvisited_nbrs))
            unvisited.remove(u)
            visited = all_nodes - unvisited
            A = (neighbors[u] - {v}) & visited
            current_trail.append((u, A))
            v = u

    sent.append(current_trail)
    return sent


def sent_to_tokens(sent, node_combo_ids, TOK_OPEN, TOK_CLOSE, TOK_BREAK, NODE_OFFSET):
    """
    node_combo_ids: list[int], 每个节点的组合类型 ID（已从 combo_to_id 映射好）
    """
    tokens = []
    node_to_id = {}
    next_id = [1]
    visit_order = []

    def get_id(node):
        if node not in node_to_id:
            node_to_id[node] = next_id[0]
            next_id[0] += 1
            visit_order.append(node)
        return node_to_id[node]

    for seg_idx, trail in enumerate(sent):
        if seg_idx > 0:
            tokens.append(TOK_BREAK)
        for node, nbrs in trail:
            v_id   = get_id(node)
            v_type = node_combo_ids[node]
            nbr_ids = sorted(get_id(u) for u in nbrs)
            tokens.append(NODE_OFFSET + v_id)
            tokens.append(v_type)
            tokens.append(TOK_OPEN)
            tokens.extend(NODE_OFFSET + nid for nid in nbr_ids)
            tokens.append(TOK_CLOSE)

    return tokens, visit_order


def reorder_by_visit(visit_order, coords, combo_ids, adj, n):
    coords_r   = np.zeros((MAX_NODES, 2), dtype=np.float32)
    comboids_r = np.zeros(MAX_NODES, dtype=np.int32)
    for new_idx, orig_idx in enumerate(visit_order):
        coords_r[new_idx]   = coords[orig_idx]
        comboids_r[new_idx] = combo_ids[orig_idx]
    adj_r = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    for ni, oi in enumerate(visit_order):
        for nj, oj in enumerate(visit_order):
            adj_r[ni, nj] = adj[oi, oj]
    return coords_r, comboids_r, adj_r


# ── 主程序 ─────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    # 第一遍：建组合词表
    print('第一遍扫描，建组合类型词表...')
    combo_to_id = build_combo_vocab(SRC_FILE)

    N_TYPES     = max(combo_to_id.values())   # 实际最大类型 ID
    TOK_OPEN    = N_TYPES + 1
    TOK_CLOSE   = N_TYPES + 2
    TOK_BREAK   = N_TYPES + 3
    BOS_ID      = N_TYPES + 4
    EOS_ID      = N_TYPES + 5
    NODE_OFFSET = N_TYPES + 6
    VOCAB_SIZE  = NODE_OFFSET + MAX_NODES + 1  # 节点token最大为NODE_OFFSET+MAX_NODES，需+1

    # 统计多类型组合
    multi = {k: v for k, v in combo_to_id.items() if len(k) > 1}
    print(f'单类型: 7  多类型组合: {len(multi)}  总类型数: {N_TYPES}')
    print(f'Token 常量: TOK_OPEN={TOK_OPEN} TOK_CLOSE={TOK_CLOSE} '
          f'TOK_BREAK={TOK_BREAK} BOS={BOS_ID} EOS={EOS_ID} '
          f'NODE_OFFSET={NODE_OFFSET} VOCAB_SIZE={VOCAB_SIZE}')
    if multi:
        print('多类型组合列表:')
        for combo, cid in sorted(multi.items(), key=lambda x: x[1]):
            names = sorted(combo)
            print(f'  ID {cid}: {names}')

    # 保存词表（供训练脚本加载）
    vocab_out = {
        'combo_to_id': {str(sorted(list(k))): v for k, v in combo_to_id.items()},
        'N_TYPES':     N_TYPES,
        'TOK_OPEN':    TOK_OPEN,
        'TOK_CLOSE':   TOK_CLOSE,
        'TOK_BREAK':   TOK_BREAK,
        'BOS_ID':      BOS_ID,
        'EOS_ID':      EOS_ID,
        'NODE_OFFSET': NODE_OFFSET,
        'VOCAB_SIZE':  VOCAB_SIZE,
        'MAX_NODES':   MAX_NODES,
    }
    with OUT_VOCAB.open('w', encoding='utf-8') as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)
    print(f'已保存词表 -> {OUT_VOCAB}')

    # 第二遍：生成序列
    print('\n第二遍处理，生成 token 序列...')
    all_tokens     = []
    all_lengths    = []
    all_coords     = []
    all_node_types = []
    all_adj        = []
    all_n_nodes    = []
    all_prompts    = []
    truncated      = 0

    with SRC_FILE.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            n          = len(data['vertices'])
            coords_raw = np.array(data['vertices'],   dtype=np.float32)
            adj_raw    = np.array(data['adj_matrix'], dtype=np.float32)
            prompt     = data['prompt']

            adj_n = adj_raw[:n, :n].copy()
            np.fill_diagonal(adj_n, 0)

            # 用组合类型替代单类型
            combos = get_vertex_combos(data)
            combo_ids = np.array([combo_to_id[c] for c in combos], dtype=np.int32)

            for _ in range(AUGMENT):
                sent = sample_sent(adj_n, n)
                tokens, visit_order = sent_to_tokens(
                    sent, combo_ids, TOK_OPEN, TOK_CLOSE, TOK_BREAK, NODE_OFFSET
                )
                full_tokens = [BOS_ID] + tokens + [EOS_ID]

                if len(full_tokens) > MAX_LEN:
                    full_tokens = full_tokens[:MAX_LEN]
                    truncated += 1

                length = len(full_tokens)
                padded = np.zeros(MAX_LEN, dtype=np.int32)
                padded[:length] = full_tokens

                coords_r, comboids_r, adj_r = reorder_by_visit(
                    visit_order, coords_raw, combo_ids, adj_n, n
                )

                # 中心化：有效节点坐标减去重心后取整，padding位保持0
                centroid = coords_r[:n].mean(axis=0)
                coords_r[:n] = np.round(coords_r[:n] - centroid).astype(np.float32)

                all_tokens.append(padded)
                all_lengths.append(length)
                all_coords.append(coords_r)
                all_node_types.append(comboids_r)
                all_adj.append(adj_r)
                all_n_nodes.append(n)
                all_prompts.append(prompt)

            if (line_no + 1) % 5000 == 0:
                print(f'  处理 {line_no + 1} 张图，已生成 {len(all_tokens)} 条序列')

    print(f'\n总计：{len(all_tokens)} 条序列，截断 {truncated} 条')
    lengths_arr = np.array(all_lengths)
    print(f'序列长度：min={lengths_arr.min()}  max={lengths_arr.max()}  mean={lengths_arr.mean():.1f}')

    np.savez_compressed(
        OUT_NPZ,
        tokens     = np.array(all_tokens,     dtype=np.int32),
        lengths    = np.array(all_lengths,    dtype=np.int32),
        coords     = np.array(all_coords,     dtype=np.float32),
        node_types = np.array(all_node_types, dtype=np.int32),
        adj_matrix = np.array(all_adj,        dtype=np.float32),
        n_nodes    = np.array(all_n_nodes,    dtype=np.int32),
    )
    print(f'已保存 -> {OUT_NPZ}')

    with OUT_TXT.open('w', encoding='utf-8') as f:
        for p in all_prompts:
            f.write(p + '\n')
    print(f'已保存 -> {OUT_TXT}')


if __name__ == '__main__':
    main()
