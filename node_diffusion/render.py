"""
Render floor plan polygons from structured graph output.

Pipeline per sample:
  1. Half-edge traversal on the 2D-embedded planar graph → find all bounded faces
  2. Each bounded face = one room polygon
  3. Majority vote on room type using the node_combo_ids of nodes in the face
     Tie-break: look one hop outside the face; the most common external type
     is excluded from candidates (adjacent rooms tend to differ).
  4. Draw colored polygons with matplotlib, save as PNG.

Usage:
    python node_diffusion/render.py \
        --input  data/jsonl/test_graph_dataset_8k.jsonl \
        --vocab  data/processed/type_combo_vocab.json \
        --out-dir data/rendered \
        --n 20
"""

import argparse
from shapely.geometry import Polygon as ShapelyPolygon
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# ── Constants ─────────────────────────────────────────────────────────────────

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]

ROOM_COLORS = {
    "bathroom":    "#AED6F1",
    "bedroom":     "#D7BDE2",
    "living_room": "#FAD7A0",
    "kitchen":     "#A9DFBF",
    "corridor":    "#CCD1D1",
    "dining_room": "#F9E79F",
    "other":       "#EAEDED",
}

ROOM_LABELS = {
    "bathroom":    "Bath",
    "bedroom":     "Bed",
    "living_room": "Living",
    "kitchen":     "Kitchen",
    "corridor":    "Corridor",
    "dining_room": "Dining",
    "other":       "Other",
}


# ── Vocab ─────────────────────────────────────────────────────────────────────

def load_vocab(path: Path) -> Dict[int, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload["id_to_combo"].items()}


# ── Half-edge face finder ─────────────────────────────────────────────────────

def _build_sorted_neighbors(
    coords: List[Tuple[float, float]],
    adj: List[List[int]],
    n: int,
) -> Dict[int, List[int]]:
    """For each node, sort neighbors CCW by angle."""
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(
                coords[w][1] - coords[i][1],
                coords[w][0] - coords[i][0],
            ),
        )
    return nbrs


def _next_half_edge(
    u: int, v: int, sorted_nbrs: Dict[int, List[int]]
) -> Optional[int]:
    """
    For directed edge u→v, return w such that v→w is the next half-edge
    in the same face (most clockwise turn at v after arriving from u).

    Implementation: in the CCW-sorted neighbor list of v, find u, then
    take the previous entry (one step clockwise).
    """
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face: List[int], coords: List[Tuple[float, float]]) -> float:
    pts = [coords[i] for i in face]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(
    coords: List[Tuple[float, float]],
    adj: List[List[int]],
) -> List[List[int]]:
    """
    Return all bounded faces of the planar graph as lists of node indices.
    The outer (unbounded) face is identified by having the largest absolute
    area and is excluded from the result.
    """
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)

    visited: set = set()
    faces: List[List[int]] = []

    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face: List[int] = []
            cu, cv = u, v
            steps = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv))
                face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw
                steps += 1
            if len(face) >= 3:
                faces.append(face)

    if not faces:
        return []

    # Remove the outer (unbounded) face — it has the largest absolute area.
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


# ── Room type voting ──────────────────────────────────────────────────────────

def vote_room_type(
    face: List[int],
    node_types: List[List[str]],
    all_nbrs: Dict[int, List[int]],
) -> str:
    """
    Determine room type for a face using a specificity score.

    Raw vote counts are unreliable because boundary nodes carry combo types
    that bleed across faces (e.g. a corridor+living_room node inflates
    'corridor' even for the living room face).

    Specificity score for each type t:
        score(t) = face_count(t) / (ext_count(t) + 1)

    A type concentrated inside the face but rare outside scores high.
    A type like 'corridor' that saturates all neighboring faces scores low.

    Final fallback (score tie): pick earliest in ROOM_TYPE_ORDER.
    """
    face_set = set(face)

    # Count each type in face nodes
    face_counts: Counter = Counter()
    for node in face:
        for t in node_types[node]:
            face_counts[t] += 1

    if not face_counts:
        return "other"

    # Count each type in one-hop external neighbors
    ext_counts: Counter = Counter()
    for node in face:
        for w in all_nbrs[node]:
            if w not in face_set:
                for t in node_types[w]:
                    ext_counts[t] += 1

    # Specificity: how concentrated is this type in the face vs outside
    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1)
              for t in face_counts}

    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]

    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_sample(
    sample: Dict,
    id_to_combo: Dict[int, List[str]],
    out_path: Path,
) -> None:
    n = sample["n_nodes"]
    raw_coords = sample["node_coords"][:n]
    combo_ids  = sample["node_combo_ids"][:n]
    adj        = [row[:n] for row in sample["adj_matrix"][:n]]
    prompt     = sample.get("prompt", "")

    coords: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in raw_coords]
    node_types = [id_to_combo.get(cid, ["other"]) for cid in combo_ids]

    # Build adjacency list (for tie-break lookups)
    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                all_nbrs[i].append(j)

    faces = find_faces(coords, adj)
    face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]

    # Normalize coordinates to [0,1] with margin
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def norm(x: float, y: float) -> Tuple[float, float]:
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw room polygons
    for face, room_type in zip(faces, face_types):
        pts = [norm(*coords[i]) for i in face]
        poly = MplPolygon(
            pts, closed=True,
            facecolor=ROOM_COLORS.get(room_type, "#EAEDED"),
            edgecolor="#555555", linewidth=1.0, alpha=0.88, zorder=1,
        )
        ax.add_patch(poly)
        # Use shapely representative_point() — guaranteed to be inside
        # the polygon even for concave/L-shaped/frame corridors.
        try:
            rp = ShapelyPolygon(pts).representative_point()
            cx, cy = rp.x, rp.y
        except Exception:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, ROOM_LABELS.get(room_type, room_type),
                ha="center", va="center", fontsize=6.5,
                color="#222222", zorder=3)

    # Draw edges
    for i in range(n):
        for j in all_nbrs[i]:
            if j > i:
                x0, y0 = norm(*coords[i])
                x1, y1 = norm(*coords[j])
                ax.plot([x0, x1], [y0, y1],
                        color="#888888", linewidth=0.8, zorder=2)

    # Draw nodes
    for i in range(n):
        x, y = norm(*coords[i])
        ax.plot(x, y, "o", color="#333333", markersize=3.5, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title = (prompt[:90] + "…") if len(prompt) > 90 else prompt
    ax.set_title(title, fontsize=6, pad=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=Path, default=Path("data/jsonl/test_graph_dataset_8k.jsonl"))
    parser.add_argument("--vocab",   type=Path, default=Path("data/processed/type_combo_vocab.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/rendered"))
    parser.add_argument("--n",       type=int,  default=20, help="Number of samples to render (0 = all)")
    parser.add_argument("--seed",    type=int,  default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_to_combo = load_vocab(args.vocab)

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))

    if args.n > 0:
        random.seed(args.seed)
        samples = random.sample(rows, min(args.n, len(rows)))
    else:
        samples = rows

    ok = err = 0
    for i, sample in enumerate(samples):
        name = sample.get("image", f"sample_{i}").replace(".png", "")
        out_path = args.out_dir / f"{name}.png"
        try:
            render_sample(sample, id_to_combo, out_path)
            ok += 1
            print(f"[{i+1}/{len(samples)}] {out_path.name}")
        except Exception as e:
            err += 1
            print(f"[{i+1}/{len(samples)}] ERROR {name}: {e}")

    print(f"\nDone. OK={ok} ERR={err} → {args.out_dir}")


if __name__ == "__main__":
    main()
