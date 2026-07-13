"""Visualize JSONL graph layouts with prompts.

Each output image contains one GT layout rendered from node_coords,
node_mask and adj_matrix, with node indices shown on the graph and the prompt
wrapped on the side for manual inspection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]

ROOM_COLORS = {
    "bathroom": "#AED6F1",
    "bedroom": "#D7BDE2",
    "living_room": "#FAD7A0",
    "kitchen": "#A9DFBF",
    "corridor": "#CCD1D1",
    "dining_room": "#F9E79F",
    "other": "#EAEDED",
}

ROOM_LABELS = {
    "bathroom": "Bath",
    "bedroom": "Bed",
    "living_room": "Living",
    "kitchen": "Kitchen",
    "corridor": "Corridor",
    "dining_room": "Dining",
    "other": "Other",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="outputs/tri_from_llm_graph_ddim500.jsonl")
    p.add_argument("--out_dir", default="outputs/val_jsonl_layout_prompt")
    p.add_argument("--n", type=int, default=8, help="Number of rows to visualize when --rows is omitted")
    p.add_argument("--rows", nargs="*", type=int, default=None, help="0-based row indices to visualize")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def read_selected(path: str, rows: List[int] | None, n: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    wanted = set(rows) if rows is not None else None
    yielded = 0
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if wanted is not None and idx not in wanted:
                continue
            yield idx, json.loads(line)
            yielded += 1
            if wanted is None and yielded >= n:
                break


def valid_indices(row: Dict[str, Any]) -> List[int]:
    mask = row.get("node_mask")
    if "gt_node_coords" in row and "gt_n_nodes" in row:
        n_nodes = int(row.get("gt_n_nodes", 0))
    else:
        n_nodes = int(row.get("n_nodes", 0))
    if mask is None:
        return list(range(n_nodes))
    return [i for i, v in enumerate(mask) if float(v) > 0.5]


def layout_name(row_index: int, row: Dict[str, Any]) -> str:
    image = row.get("image") or Path(str(row.get("image_path", ""))).name
    src = row.get("source_file", "?")
    line = row.get("source_line", "?")
    return f"val row {row_index} | image {image} | source {src}:{line}"


def _build_sorted_neighbors(coords: List[Tuple[float, float]], adj: List[List[int]], n: int) -> Dict[int, List[int]]:
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(coords[w][1] - coords[i][1], coords[w][0] - coords[i][0]),
        )
    return nbrs


def _next_half_edge(u: int, v: int, sorted_nbrs: Dict[int, List[int]]) -> int | None:
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    return nbrs[(nbrs.index(u) - 1) % len(nbrs)]


def _signed_area(face: List[int], coords: List[Tuple[float, float]]) -> float:
    pts = [coords[i] for i in face]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords: List[Tuple[float, float]], adj: List[List[int]]) -> List[List[int]]:
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited, faces = set(), []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face, cu, cv, steps = [], u, v, 0
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
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


def vote_room_type(face: List[int], node_types: List[List[str]], all_nbrs: Dict[int, List[int]]) -> str:
    face_set = set(face)
    face_counts: Counter = Counter()
    for node in face:
        for t in node_types[node]:
            face_counts[t] += 1
    if not face_counts:
        return "other"

    ext_counts: Counter = Counter()
    for node in face:
        for w in all_nbrs[node]:
            if w not in face_set:
                for t in node_types[w]:
                    ext_counts[t] += 1

    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1) for t in face_counts}
    best = max(scores.values())
    winners = [t for t, s in scores.items() if s == best]
    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))


def normalize_coords(coords: List[Tuple[float, float]]):
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

    return norm


def draw_graph(ax, row: Dict[str, Any]) -> None:
    if "node_coords" in row:
        coords_key, adj_key, types_key = "node_coords", "adj_matrix", "node_types"
        n_key = "n_nodes"
    elif "gt_node_coords" in row:
        coords_key, adj_key, types_key = "gt_node_coords", "gt_adj_matrix", "gt_node_types"
        n_key = "gt_n_nodes" if "gt_n_nodes" in row else "n_nodes"
    else:
        coords_key, adj_key, types_key = "pred_node_coords", "adj_matrix", "gt_node_types"
        n_key = "n_nodes"

    coords = np.asarray(row[coords_key], dtype=float)
    adj = np.asarray(row[adj_key], dtype=float)
    valid = valid_indices(row)
    if not valid:
        ax.text(0.5, 0.5, "No valid nodes", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    n = int(row.get(n_key, len(valid)))
    n = min(n, len(valid))
    coords_list = [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
    adj_list = [[int(adj[i, j]) for j in range(n)] for i in range(n)]
    node_types = [
        (t if isinstance(t, list) else [t])
        for t in row.get(types_key, [])[:n]
    ]
    if len(node_types) < n:
        node_types.extend([["other"]] * (n - len(node_types)))
    all_nbrs = _build_sorted_neighbors(coords_list, adj_list, n)
    faces = find_faces(coords_list, adj_list)
    face_types = [vote_room_type(face, node_types, all_nbrs) for face in faces]
    norm = normalize_coords(coords_list)

    for face, room_type in zip(faces, face_types):
        poly_pts = [norm(*coords_list[i]) for i in face]
        patch = MplPolygon(
            poly_pts, closed=True,
            facecolor=ROOM_COLORS.get(room_type, ROOM_COLORS["other"]),
            edgecolor="#555555", linewidth=1.0, alpha=0.88, zorder=1,
        )
        ax.add_patch(patch)
        try:
            rp = ShapelyPolygon(poly_pts).representative_point()
            tx, ty = rp.x, rp.y
        except Exception:
            tx = sum(p[0] for p in poly_pts) / len(poly_pts)
            ty = sum(p[1] for p in poly_pts) / len(poly_pts)
        ax.text(tx, ty, ROOM_LABELS.get(room_type, room_type),
                ha="center", va="center", fontsize=8.5, color="#222222", zorder=3)

    pos = {i: norm(*coords_list[i]) for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                pi, pj = pos[i], pos[j]
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#777777", lw=0.9, zorder=2)

    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=260, c="#2F80C1", edgecolors="#222222", linewidths=0.9, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", color="white",
                fontsize=7, fontweight="bold", zorder=5)

    ax.set_title(f"Rendered layout from {coords_key} / {types_key} / {adj_key}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x -> right/east")
    ax.set_ylabel("y -> up/north")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_text(ax, row_index: int, row: Dict[str, Any]) -> None:
    ax.axis("off")
    valid = valid_indices(row)
    prompt = row.get("prompt", "")
    original = row.get("prompt_original", "")

    def block(label: str, text: str, width: int = 78) -> List[str]:
        wrapped = textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        return [f"{label}:"] + [f"  {line}" for line in wrapped]

    info = [
        f"LAYOUT : {layout_name(row_index, row)}",
        f"NODES  : n_nodes={row.get('n_nodes')} | valid_mask_nodes={len(valid)} | adj_size={len(row.get('adj_matrix', []))}",
        "",
        *block("PROMPT", prompt),
    ]
    if original and original != prompt:
        info.extend(["", *block("PROMPT_ORIGINAL", original)])
    ax.text(
        0.02, 0.98, "\n".join(info),
        ha="left", va="top", fontsize=8.8,
        transform=ax.transAxes, family="DejaVu Sans Mono",
        linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F8F8F4", edgecolor="#CCCCCC"),
    )


def save_one(row_index: int, row: Dict[str, Any], out_dir: Path, dpi: int) -> Path:
    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25])
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    fig.suptitle(layout_name(row_index, row), fontsize=13, fontweight="bold")
    draw_graph(ax_graph, row)
    draw_text(ax_text, row_index, row)

    out_path = out_dir / f"{row_index:05d}_{Path(str(row.get('image', 'layout'))).stem}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for row_index, row in read_selected(args.jsonl, args.rows, args.n):
        out_path = save_one(row_index, row, out_dir, args.dpi)
        print(f"saved row={row_index} -> {out_path}")
        count += 1
    print(f"done: {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()
