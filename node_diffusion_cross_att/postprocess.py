"""
Post-processing for node coordinate diffusion output.

snap_nodes_to_walls:
    If a node is within `threshold` pixels of a wall (an edge between two
    connected nodes), project it onto that wall and split the wall edge so
    the node is topologically ON the wall.
"""

import numpy as np


def _point_segment_closest(p, a, b):
    """
    Closest point on segment [a, b] to point p, and the distance.
    Returns (proj, dist, t) where t in [0,1] is the parameter along ab.
    """
    ab   = b - a
    ab2  = np.dot(ab, ab)
    if ab2 < 1e-10:
        return a.copy(), np.linalg.norm(p - a), 0.0
    t    = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
    proj = a + t * ab
    return proj, np.linalg.norm(p - proj), t


def snap_nodes_to_walls(coords, adj, node_mask, threshold=8.0,
                        endpoint_tol=0.05):
    """
    Snap free nodes that are close to wall segments onto those walls.

    Parameters
    ----------
    coords       : np.ndarray [N, 2]   node coordinates (pixel space)
    adj          : np.ndarray [N, N]   symmetric 0/1 adjacency matrix
    node_mask    : np.ndarray [N]      1 = valid node
    threshold    : float               distance threshold in pixels
    endpoint_tol : float               if projection t is within this of
                                       0 or 1, treat node as near-endpoint
                                       (skip — merging is a separate step)

    Returns
    -------
    new_coords : np.ndarray [N, 2]   updated coordinates
    new_adj    : np.ndarray [N, N]   updated adjacency matrix
    snapped    : list of (node_i, wall_j, wall_k, proj)  what was snapped
    """
    coords   = coords.copy().astype(np.float32)
    adj      = adj.copy().astype(np.float32)
    valid    = np.where(node_mask > 0.5)[0].tolist()
    snapped  = []

    # build list of unique edges among valid nodes
    edges = [(j, k) for idx_j, j in enumerate(valid)
                    for k in valid[idx_j + 1:]
                    if adj[j, k] > 0.5]

    already_snapped = set()

    for i in valid:
        if i in already_snapped:
            continue

        best_dist = threshold
        best_proj = None
        best_edge = None
        best_t    = None

        for (j, k) in edges:
            if i == j or i == k:
                continue

            proj, dist, t = _point_segment_closest(coords[i], coords[j], coords[k])

            # skip if the projection is essentially at an endpoint
            if t < endpoint_tol or t > 1.0 - endpoint_tol:
                continue

            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_edge = (j, k)
                best_t    = t

        if best_proj is not None:
            j, k = best_edge
            coords[i] = best_proj

            # split wall j-k into j-i and i-k
            adj[j, k] = 0.0
            adj[k, j] = 0.0
            adj[j, i] = 1.0
            adj[i, j] = 1.0
            adj[i, k] = 1.0
            adj[k, i] = 1.0

            # refresh edge list: remove j-k, add j-i and i-k
            edges = [(a, b) for (a, b) in edges if not (a == j and b == k)]
            edges.append((min(j, i), max(j, i)))
            edges.append((min(i, k), max(i, k)))

            snapped.append((i, j, k, best_proj.copy()))
            already_snapped.add(i)

    return coords, adj, snapped


def apply_snap(sample, threshold=8.0):
    """
    Convenience wrapper for a single inference sample.

    Parameters
    ----------
    sample : dict with keys
               'coords'    np.ndarray [N, 2]
               'adj'       np.ndarray [N, N]
               'node_mask' np.ndarray [N]
    threshold : float

    Returns
    -------
    Updated sample dict (coords and adj replaced in-place copy).
    """
    new_coords, new_adj, snapped = snap_nodes_to_walls(
        sample['coords'], sample['adj'], sample['node_mask'],
        threshold=threshold,
    )
    if snapped:
        print(f'[snap] {len(snapped)} node(s) snapped to wall: '
              + ', '.join(f'v{i}→wall({j},{k})' for i, j, k, _ in snapped))
    return {**sample, 'coords': new_coords, 'adj': new_adj}
