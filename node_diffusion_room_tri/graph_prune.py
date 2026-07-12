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


def _bridge_edges(adj: np.ndarray) -> set:
    n = adj.shape[0]
    tin = [-1] * n
    low = [0] * n
    bridges = set()
    timer = 0

    def dfs(v: int, parent: int) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to in np.where(adj[v] != 0)[0]:
            to = int(to)
            if to == parent:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.add((min(v, to), max(v, to)))

    for v in range(n):
        if tin[v] == -1:
            dfs(v, -1)
    return bridges


def prune_non_cycle_nodes(
    adj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iteratively keep only nodes that belong to at least one cycle.

    In an undirected graph, an edge is part of a cycle iff it is not a bridge.
    A node is kept if it is incident to any non-bridge edge. This removes
    dangling trees and bridge chains between cyclic components, then repeats
    until stable.
    """
    adj = np.asarray(adj).copy()
    n = min(adj.shape[0], adj.shape[1])
    adj = adj[:n, :n]
    adj = ((adj + adj.T) > 0).astype(np.int32)
    np.fill_diagonal(adj, 0)

    keep_global = np.arange(n)
    while len(keep_global) > 0:
        m = adj.shape[0]
        bridges = _bridge_edges(adj)
        keep = np.zeros(m, dtype=bool)
        for i in range(m):
            for j in np.where(adj[i] != 0)[0]:
                edge = (min(i, int(j)), max(i, int(j)))
                if edge not in bridges:
                    keep[i] = True
                    break
        if keep.all():
            break
        keep_global = keep_global[keep]
        adj = adj[np.ix_(keep, keep)]

    return adj, keep_global
