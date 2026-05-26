"""
把 nodes_train_6k_norm.npz 里的邻接矩阵
转换成 AutoGraph 风格的 token 序列，保存为 .npz
"""

import numpy as np
import random
from collections import defaultdict

# ── 特殊 token ID ──────────────────────────────────────────
# 节点编号从 1 开始（按首次出现顺序重编号，最多 40 个节点）
# 特殊 token 放在节点编号之后
MAX_NODES   = 40
PAD_ID      = 0
TOK_OPEN    = MAX_NODES + 1   # <
TOK_CLOSE   = MAX_NODES + 2   # >
TOK_BREAK   = MAX_NODES + 3   # /（段与段之间的断开）
BOS_ID      = MAX_NODES + 4
EOS_ID      = MAX_NODES + 5
VOCAB_SIZE  = MAX_NODES + 6


# ── 游走算法（论文 Algorithm 1）────────────────────────────
def sample_sent(adj, n):
    """
    adj: n×n numpy 数组，无向图，无自环（已去对角线）
    n:   真实节点数
    返回: SENT，格式为 list of list of (node, neighbors)
          node 和 neighbors 都是 0-based 整数
    """
    # 建邻接表
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                neighbors[i].add(j)

    unvisited = set(range(n))
    all_nodes = set(range(n))

    # 随机选起点
    v = random.choice(list(unvisited))
    unvisited.remove(v)

    current_trail = [(v, set())]   # (节点, 已访问邻居集合)
    sent = []

    while unvisited:
        unvisited_neighbors = neighbors[v] & unvisited

        if not unvisited_neighbors:
            # 走不下去了，断开，开新段
            sent.append(current_trail)
            v = random.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            A = neighbors[v] & visited   # v 和已访问节点的连接
            current_trail = [(v, A)]
        else:
            # 走到下一个未访问邻居
            u = random.choice(list(unvisited_neighbors))
            unvisited.remove(u)
            visited = all_nodes - unvisited
            A = (neighbors[u] - {v}) & visited   # u 的已访问邻居，排除来路 v
            current_trail.append((u, A))
            v = u

    sent.append(current_trail)
    return sent


# ── Tokenize（论文 Section 2.4）────────────────────────────
def sent_to_tokens(sent):
    """
    sent: list of list of (node_0based, neighbor_set_0based)
    返回: list of int（节点从 1 开始重编号）
    """
    tokens = []

    # 按首次出现顺序给节点重编号
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
            # 邻居按重编号排序，保证确定性
            nbr_ids = sorted(get_id(u) for u in nbrs)
            tokens.append(v_id)
            tokens.append(TOK_OPEN)
            tokens.extend(nbr_ids)
            tokens.append(TOK_CLOSE)

    return tokens


# ── 主流程 ──────────────────────────────────────────────────
def preprocess(input_path, output_path, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = np.load(input_path, allow_pickle=True)
    adj_all  = data['adj_matrix']   # (N, 40, 40)
    mask_all = data['node_mask']    # (N, 40)
    N = len(adj_all)

    all_tokens = []   # list of list of int
    max_len = 0

    for i in range(N):
        n = int(mask_all[i].sum())
        adj = adj_all[i, :n, :n].copy()
        np.fill_diagonal(adj, 0)   # 去掉自环

        sent = sample_sent(adj, n)
        tokens = [BOS_ID] + sent_to_tokens(sent) + [EOS_ID]
        all_tokens.append(tokens)
        max_len = max(max_len, len(tokens))

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} 处理完毕，当前最长序列={max_len}")

    print(f"最长序列长度: {max_len}")
    print(f"词表大小: {VOCAB_SIZE}")

    # 对齐到 max_len，用 PAD_ID 填充
    token_array = np.full((N, max_len), PAD_ID, dtype=np.int32)
    length_array = np.zeros(N, dtype=np.int32)
    for i, toks in enumerate(all_tokens):
        token_array[i, :len(toks)] = toks
        length_array[i] = len(toks)

    np.savez(output_path,
             tokens=token_array,
             lengths=length_array)

    print(f"已保存到 {output_path}.npz")
    print(f"tokens shape: {token_array.shape}")
    print(f"序列长度统计: min={length_array.min()}, max={length_array.max()}, mean={length_array.mean():.1f}")

    # 打印第一张图的样子
    print("\n── 第一张图示例 ──")
    toks = list(token_array[0, :length_array[0]])
    readable = []
    for t in toks:
        if t == BOS_ID:      readable.append('[BOS]')
        elif t == EOS_ID:    readable.append('[EOS]')
        elif t == TOK_OPEN:  readable.append('<')
        elif t == TOK_CLOSE: readable.append('>')
        elif t == TOK_BREAK: readable.append('/')
        else:                readable.append(str(t))
    print(' '.join(readable))


if __name__ == '__main__':
    preprocess(
        input_path  = 'data/processed/nodes_train_6k_norm.npz',
        output_path = 'data/processed/graph_tokens_train',
    )
