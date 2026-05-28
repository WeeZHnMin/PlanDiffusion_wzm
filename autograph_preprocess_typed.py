"""
预处理带节点类型的平面布局图数据
输入：data/jsonl/mapped_type_data_zh.jsonl
输出：
  data/processed/graph_tokens_typed_5w.npz   — token序列+坐标+类型+邻接矩阵
  data/processed/graph_prompts_5w.txt        — 提示词，行号对应npz索引
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── 路径 ───────────────────────────────────────────────────
SRC_FILE  = Path('data/jsonl/mapped_type_data_zh.jsonl')
OUT_NPZ   = Path('data/processed/graph_tokens_typed_5w.npz')
OUT_TXT   = Path('data/processed/graph_prompts_5w.txt')

# ── token 编号 ─────────────────────────────────────────────
PAD_ID    = 0
TYPE_1    = 1   # bathroom
TYPE_2    = 2   # bedroom
TYPE_3    = 3   # living_room
TYPE_4    = 4   # kitchen
TYPE_5    = 5   # corridor
TYPE_6    = 6   # dining_room
TYPE_7    = 7   # other
TOK_OPEN  = 8   # <
TOK_CLOSE = 9   # >
TOK_BREAK = 10  # /
BOS_ID    = 11
EOS_ID    = 12
NODE_OFFSET = 13          # 节点编号从13开始，节点i对应token = i + NODE_OFFSET
MAX_NODES = 40
MAX_LEN   = 256
VOCAB_SIZE = NODE_OFFSET + MAX_NODES  # 53
AUGMENT   = 5   # 每张图游走次数

# ── 游走算法 ───────────────────────────────────────────────
def sample_sent(adj, n):
    """
    随机游走生成 SENT，返回 trail 列表
    每个 trail 是 [(node_idx, {visited_neighbors}), ...]
    """
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                neighbors[i].add(j)

    unvisited = set(range(n))
    all_nodes = set(range(n))

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


def sent_to_tokens_typed(sent, node_types):
    """
    把 SENT + 节点类型 转成 token 序列
    格式：节点编号 节点类型 < 邻居编号列表 >
    节点按首次出现顺序重编号（1-based）
    同时返回游走顺序（原始节点索引顺序）
    """
    tokens = []
    node_to_id = {}
    next_id = [1]
    visit_order = []  # 原始节点索引，按首次出现顺序

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
            v_id = get_id(node)
            v_type = node_types[node]
            nbr_ids = sorted(get_id(u) for u in nbrs)
            tokens.append(NODE_OFFSET + v_id)   # 节点编号token
            tokens.append(v_type)                # 节点类型token
            tokens.append(TOK_OPEN)
            tokens.extend(NODE_OFFSET + nid for nid in nbr_ids)
            tokens.append(TOK_CLOSE)

    return tokens, visit_order


def reorder_by_visit(visit_order, coords, node_types, adj, n):
    """
    按游走顺序重排坐标、类型、邻接矩阵
    """
    coords_reordered = np.zeros((MAX_NODES, 2), dtype=np.float32)
    types_reordered  = np.zeros(MAX_NODES, dtype=np.int32)

    for new_idx, orig_idx in enumerate(visit_order):
        coords_reordered[new_idx] = coords[orig_idx]
        types_reordered[new_idx]  = node_types[orig_idx]

    # 重排邻接矩阵
    perm = visit_order  # visit_order 是原始索引列表
    adj_reordered = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    for new_i, orig_i in enumerate(perm):
        for new_j, orig_j in enumerate(perm):
            adj_reordered[new_i, new_j] = adj[orig_i, orig_j]

    return coords_reordered, types_reordered, adj_reordered


# ── 主程序 ─────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    all_tokens     = []
    all_lengths    = []
    all_coords     = []
    all_node_types = []
    all_adj        = []
    all_n_nodes    = []
    all_prompts    = []

    truncated = 0

    with SRC_FILE.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            n          = len(data['vertices'])
            coords_raw = np.array(data['vertices'], dtype=np.float32)   # (n, 2)
            vtypes_raw = np.array(data['vertex_type'], dtype=np.int32)  # (n,)
            adj_raw    = np.array(data['adj_matrix'], dtype=np.float32) # (40, 40)
            prompt     = data['prompt']

            # 去对角线自环
            adj_n = adj_raw[:n, :n].copy()
            np.fill_diagonal(adj_n, 0)

            for _ in range(AUGMENT):
                sent = sample_sent(adj_n, n)
                tokens, visit_order = sent_to_tokens_typed(sent, vtypes_raw)
                full_tokens = [BOS_ID] + tokens + [EOS_ID]

                # 截断
                if len(full_tokens) > MAX_LEN:
                    full_tokens = full_tokens[:MAX_LEN]
                    truncated += 1

                length = len(full_tokens)

                # padding
                padded = np.zeros(MAX_LEN, dtype=np.int32)
                padded[:length] = full_tokens

                # 重排坐标、类型、邻接矩阵
                coords_r, types_r, adj_r = reorder_by_visit(
                    visit_order, coords_raw, vtypes_raw, adj_n, n
                )

                all_tokens.append(padded)
                all_lengths.append(length)
                all_coords.append(coords_r)
                all_node_types.append(types_r)
                all_adj.append(adj_r)
                all_n_nodes.append(n)
                all_prompts.append(prompt)

            if (line_no + 1) % 5000 == 0:
                print(f'  处理 {line_no + 1} 张图，已生成 {len(all_tokens)} 条序列')

    print(f'\n总计：{len(all_tokens)} 条序列，截断 {truncated} 条')

    # 统计序列长度
    lengths_arr = np.array(all_lengths)
    print(f'序列长度：min={lengths_arr.min()}  max={lengths_arr.max()}  mean={lengths_arr.mean():.1f}')

    # 保存 npz
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

    # 保存提示词
    with OUT_TXT.open('w', encoding='utf-8') as f:
        for p in all_prompts:
            f.write(p + '\n')
    print(f'已保存 -> {OUT_TXT}')


if __name__ == '__main__':
    main()
