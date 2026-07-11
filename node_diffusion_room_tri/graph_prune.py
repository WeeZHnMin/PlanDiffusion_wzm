"""Graph cleanup helpers shared by tri-stream training and evaluation."""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def prune_dangling_nodes(
    coords: np.ndarray,
    adj: np.ndarray,
    node_types: Optional[Sequence] = None,
    min_degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Optional[List], np.ndarray]:
    """
    Iteratively remove active nodes whose graph degree is below min_degree.

    This removes isolated nodes and leaf-like dangling branches together with
    their incident edges. Self loops are ignored for degree counting.
    """
    coords = np.asarray(coords)
    adj = np.asarray(adj).copy()
    n = min(len(coords), adj.shape[0], adj.shape[1])
    coords = coords[:n]
    adj = adj[:n, :n]
    np.fill_diagonal(adj, 0)

    keep = np.ones(n, dtype=bool)
    while keep.any():
        active = np.where(keep)[0]
        sub_adj = adj[np.ix_(active, active)]
        deg = sub_adj.sum(axis=1)
        remove = active[deg < min_degree]
        if len(remove) == 0:
            break
        keep[remove] = False

    kept_idx = np.where(keep)[0]
    pruned_coords = coords[kept_idx]
    pruned_adj = adj[np.ix_(kept_idx, kept_idx)]
    pruned_types = None
    if node_types is not None:
        pruned_types = [node_types[i] for i in kept_idx]
    return pruned_coords, pruned_adj, pruned_types, kept_idx
