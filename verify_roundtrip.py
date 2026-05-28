"""
验证：邻接矩阵 → token 序列 → 还原图结构，是否无损
由于 token 序列对节点重编号，验证图的结构属性（边数、度序列），而非逐位对比
"""
import numpy as np
import random
from autograph_preprocess import sample_sent, sent_to_tokens

MAX_NODES = 40
BOS_ID = 44; EOS_ID = 45; TOK_OPEN = 41; TOK_CLOSE = 42; TOK_BREAK = 43; PAD_ID = 0


def decode_to_edges(token_list):
    """从 token 序列还原边集合（节点编号为 token 中的重编号，1-based）"""
    edges = set()
    toks = [t for t in token_list if t not in (BOS_ID, EOS_ID, PAD_ID)]
    prev_node = None
    in_bracket = False
    bracket_node = None

    for t in toks:
        if t == TOK_BREAK:
            prev_node = None
            in_bracket = False
            bracket_node = None
        elif t == TOK_OPEN:
            in_bracket = True
            bracket_node = prev_node
        elif t == TOK_CLOSE:
            in_bracket = False
            bracket_node = None
        elif 1 <= t <= MAX_NODES:
            if in_bracket:
                if bracket_node is not None:
                    edges.add((min(bracket_node, t), max(bracket_node, t)))
            else:
                if prev_node is not None:
                    edges.add((min(prev_node, t), max(prev_node, t)))
                prev_node = t
    return edges


def degree_sequence(edges, n_nodes):
    deg = [0] * (n_nodes + 1)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return sorted(deg[1:n_nodes+1])


def verify(n_samples=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = np.load('data/processed/nodes_train_6k_norm.npz', allow_pickle=True)
    adj_all  = data['adj_matrix']
    mask_all = data['node_mask']

    success = 0
    for i in range(n_samples):
        n = int(mask_all[i].sum())
        adj_orig = adj_all[i, :n, :n].copy().astype(np.float32)
        np.fill_diagonal(adj_orig, 0)

        # 原始边数和度序列
        orig_edges = int(adj_orig.sum()) // 2
        orig_deg = sorted(adj_orig.sum(axis=1).astype(int).tolist())

        # 正向：邻接矩阵 → token 序列
        sent = sample_sent(adj_orig, n)
        tokens = [BOS_ID] + sent_to_tokens(sent) + [EOS_ID]

        # 反向：token 序列 → 边集合
        recon_edge_set = decode_to_edges(tokens)
        recon_edges = len(recon_edge_set)
        recon_nodes = len(set(v for e in recon_edge_set for v in e))
        recon_deg = degree_sequence(recon_edge_set, recon_nodes)

        # 比较：边数 + 度序列
        edge_ok = (orig_edges == recon_edges)
        deg_ok  = (orig_deg == recon_deg)

        if edge_ok and deg_ok:
            success += 1
        else:
            print(f'图{i}: 节点={n}  原始边={orig_edges} 还原边={recon_edges}  '
                  f'度序列匹配={deg_ok}')

    print(f'\n结果：{success}/{n_samples} 张图结构完全一致')


if __name__ == '__main__':
    verify(n_samples=100)
