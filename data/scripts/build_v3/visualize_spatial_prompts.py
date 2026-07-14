"""Visualize spatial prompt generation for manual inspection.

Each output image contains:
  1. The rendered layout from node_coords + adj_matrix, with node indices.
  2. Face-level room labels inferred by generate_spatial_prompts.py.
  3. The spatial prompt before [SEP] and the original natural-language prompt.

Usage:
  python -m data.scripts.build_v3.visualize_spatial_prompts \
      --jsonl data/jsonl/graph_160k_spatial_val.jsonl --n 8
  python -m data.scripts.build_v3.visualize_spatial_prompts \
      --jsonl data/jsonl/graph_160k_spatial_val.jsonl --rows 0 3 8
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from data.scripts.build_v3.generate_spatial_prompts import (
    TYPE_DISPLAY,
    find_bounded_faces,
    generate_prompt,
    vote_face_type,
)


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
    p.add_argument("--jsonl", default="data/jsonl/graph_160k_spatial_val.jsonl")
    p.add_argument("--out_dir", default="outputs/spatial_prompt_check")
    p.add_argument("--n", type=int, default=8, help="Number of rows to visualize when --rows is omitted")
    p.add_argument("--rows", nargs="*", type=int, default=None, help="0-based row indices to visualize")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def read_selected(path: str, rows: List[int] | None, n: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    wanted = set(rows) if rows is not None else None
    yielded = 0
    with open(path, encoding="utf-8-sig") as f:
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


def split_prompt(prompt: str) -> Tuple[str, str]:
    before, sep, after = prompt.partition(" [SEP] ")
    if sep:
        return before.strip(), after.strip()
    return "", prompt.strip()


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


def layout_name(row_index: int, row: Dict[str, Any]) -> str:
    image = row.get("image") or Path(str(row.get("image_path", ""))).name
    source_file = row.get("source_file", "?")
    source_line = row.get("source_line", "?")
    return f"row {row_index} | image {image} | source {source_file}:{source_line}"


def draw_layout(ax, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(row["n_nodes"])
    coords_arr = np.asarray(row["node_coords"][:n], dtype=float)
    adj = np.asarray(row["adj_matrix"], dtype=float)[:n, :n]
    coords = [(float(coords_arr[i, 0]), float(coords_arr[i, 1])) for i in range(n)]
    node_types = [
        t if isinstance(t, list) else [t]
        for t in row.get("node_types", [])[:n]
    ]
    if len(node_types) < n:
        node_types.extend([["other"]] * (n - len(node_types)))

    faces = find_bounded_faces(coords_arr, adj)
    norm = normalize_coords(coords)
    rooms: List[Dict[str, Any]] = []

    for face in faces:
        room_type = vote_face_type(face, node_types, adj) or "other"
        poly_pts = [norm(*coords[i]) for i in face]
        patch = MplPolygon(
            poly_pts,
            closed=True,
            facecolor=ROOM_COLORS.get(room_type, ROOM_COLORS["other"]),
            edgecolor="#555555",
            linewidth=1.0,
            alpha=0.88,
            zorder=1,
        )
        ax.add_patch(patch)
        try:
            point = ShapelyPolygon(poly_pts).representative_point()
            tx, ty = point.x, point.y
        except Exception:
            tx = sum(x for x, _ in poly_pts) / len(poly_pts)
            ty = sum(y for _, y in poly_pts) / len(poly_pts)
        label = ROOM_LABELS.get(room_type, room_type)
        ax.text(tx, ty, label, ha="center", va="center", fontsize=8.5, color="#222222", zorder=3)
        rooms.append({"type": room_type, "nodes": face})

    pos = {i: norm(*coords[i]) for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                pi, pj = pos[i], pos[j]
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#777777", lw=0.9, zorder=2)

    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=250, c="#2F80C1", edgecolors="#222222", linewidths=0.9, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Layout + inferred room faces + node indices", fontsize=11)
    return rooms


def draw_text(ax, row_index: int, row: Dict[str, Any], rooms: List[Dict[str, Any]]) -> None:
    ax.axis("off")
    prompt = row.get("prompt", "")
    stored_spatial, original_prompt = split_prompt(prompt)
    generated_spatial = generate_prompt(
        row["node_types"],
        row["node_coords"],
        row["adj_matrix"],
        int(row["n_nodes"]),
    )

    room_lines = [
        f"{i:02d}. {ROOM_LABELS.get(r['type'], r['type'])}: nodes={r['nodes']}"
        for i, r in enumerate(rooms)
    ]
    if not room_lines:
        room_lines = ["<no bounded faces found>"]

    def wrap_block(title: str, text: str, width: int = 78) -> List[str]:
        lines = textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False) or [""]
        return [f"{title}:"] + [f"  {line}" for line in lines]

    info = [
        f"LAYOUT : {layout_name(row_index, row)}",
        f"NODES  : n_nodes={row.get('n_nodes')} | faces={len(rooms)}",
        "",
        "ROOM FACES:",
        *[f"  {line}" for line in room_lines],
        "",
        *wrap_block("GENERATED_SPATIAL", generated_spatial),
    ]
    if stored_spatial:
        info.extend(["", *wrap_block("STORED_SPATIAL", stored_spatial)])
    info.extend(["", *wrap_block("ORIGINAL_PROMPT", original_prompt or prompt)])

    ax.text(
        0.02,
        0.98,
        "\n".join(info),
        ha="left",
        va="top",
        fontsize=8.6,
        transform=ax.transAxes,
        family="DejaVu Sans Mono",
        linespacing=1.16,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F8F8F4", edgecolor="#CCCCCC"),
    )


def save_one(row_index: int, row: Dict[str, Any], out_dir: Path, dpi: int) -> Path:
    fig = plt.figure(figsize=(14.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])
    ax_layout = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    fig.suptitle(layout_name(row_index, row), fontsize=13, fontweight="bold")
    rooms = draw_layout(ax_layout, row)
    draw_text(ax_text, row_index, row, rooms)

    stem = Path(str(row.get("image", f"row_{row_index}"))).stem
    out_path = out_dir / f"{row_index:05d}_{stem}.png"
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
