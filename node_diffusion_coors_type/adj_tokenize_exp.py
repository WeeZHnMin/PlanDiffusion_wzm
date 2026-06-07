"""
Experiment: encode adjacency matrix as 8-bit token sequence.

40 nodes → upper triangle (i<j) → 780 bits → pad to 784 → 98 tokens (0~255)

Usage:
    python -m node_diffusion.adj_tokenize_exp \
        --npz data/processed/nodes_train_6k_norm.npz \
        --n 100
"""

import argparse
import numpy as np


MAX_NODES = 40
# upper triangle indices (i < j) for 40 nodes
UPPER_I, UPPER_J = np.triu_indices(MAX_NODES, k=1)   # 780 pairs
N_BITS   = len(UPPER_I)                               # 780
PAD_BITS = (8 - N_BITS % 8) % 8                      # 4 padding bits
SEQ_LEN  = (N_BITS + PAD_BITS) // 8                  # 98 tokens


def adj_to_tokens(adj):
    """adj [40,40] float → tokens [98] int (0~255)"""
    bits = adj[UPPER_I, UPPER_J].astype(np.int32)     # [780]
    bits = np.concatenate([bits, np.zeros(PAD_BITS, dtype=np.int32)])  # [784]
    bits = bits.reshape(-1, 8)                         # [98, 8]
    tokens = (bits * (2 ** np.arange(7, -1, -1))).sum(axis=1).astype(np.int32)
    return tokens                                      # [98]


def tokens_to_adj(tokens, n_nodes):
    """tokens [98] int → adj [40,40] float (symmetric, diagonal=1)"""
    bits = np.unpackbits(tokens.astype(np.uint8))      # [784]
    bits = bits[:N_BITS]                               # [780]

    adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    adj[UPPER_I, UPPER_J] = bits
    adj = adj + adj.T                                  # symmetrize
    np.fill_diagonal(adj, 1)                           # self-loops

    # zero out padding nodes
    adj[n_nodes:, :] = 0
    adj[:, n_nodes:] = 0
    np.fill_diagonal(adj, 0)
    adj[:n_nodes, :n_nodes] += np.eye(n_nodes)        # restore valid self-loops

    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/processed/nodes_train_6k_norm.npz')
    parser.add_argument('--n',   type=int, default=100)
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    adj_all  = d['adj_matrix'][:args.n]    # [100, 40, 40]
    mask_all = d['node_mask'][:args.n]     # [100, 40]

    print(f"loaded {len(adj_all)} samples")
    print(f"upper triangle bits : {N_BITS}")
    print(f"padding bits        : {PAD_BITS}")
    print(f"sequence length     : {SEQ_LEN} tokens  (vocab size 256)")
    print()

    # encode & decode, verify roundtrip
    errors = 0
    for i in range(len(adj_all)):
        adj     = adj_all[i]
        n_nodes = int(mask_all[i].sum())

        tokens  = adj_to_tokens(adj)
        adj_rec = tokens_to_adj(tokens, n_nodes)

        if not np.allclose(adj, adj_rec):
            errors += 1

    print(f"roundtrip check: {len(adj_all) - errors}/{len(adj_all)} passed")
    if errors:
        print(f"  {errors} samples had reconstruction errors!")
    print()

    # show one example
    adj     = adj_all[0]
    n_nodes = int(mask_all[0].sum())
    tokens  = adj_to_tokens(adj)
    print(f"sample 0  (n_nodes={n_nodes})")
    print(f"  tokens[:10] = {tokens[:10].tolist()}")
    print(f"  tokens range: [{tokens.min()}, {tokens.max()}]")
    print(f"  token distribution (first 20):")
    vals, cnts = np.unique(tokens, return_counts=True)
    for v, c in zip(vals[:20], cnts[:20]):
        print(f"    token {v:3d} ({v:08b}) : {c}x")


if __name__ == '__main__':
    main()
