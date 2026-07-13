"""Visualize JSONL graph layouts with prompts.

Each output image contains one GT layout rendered from node_coords,
node_mask and adj_matrix, with node indices shown on the graph and the prompt
wrapped on the side for manual inspection.
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
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/jsonl/graph_160k_spatial_val.jsonl")
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
    n_nodes = int(row.get("n_nodes", 0))
    if mask is None:
        return list(range(n_nodes))
    return [i for i, v in enumerate(mask) if float(v) > 0.5]


def layout_name(row_index: int, row: Dict[str, Any]) -> str:
    image = row.get("image") or Path(str(row.get("image_path", ""))).name
    src = row.get("source_file", "?")
    line = row.get("source_line", "?")
    return f"val row {row_index} | image {image} | source {src}:{line}"


def draw_graph(ax, row: Dict[str, Any]) -> None:
    coords = np.asarray(row["node_coords"], dtype=float)
    adj = np.asarray(row["adj_matrix"], dtype=float)
    valid = valid_indices(row)
    if not valid:
        ax.text(0.5, 0.5, "No valid nodes", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    pts = coords[valid]
    center = pts.mean(axis=0, keepdims=True)
    span = max(float(np.abs(pts - center).max()), 1.0)
    draw_pts = (pts - center) / span
    pos = {node: draw_pts[i] for i, node in enumerate(valid)}

    for a_i, i in enumerate(valid):
        for j in valid[a_i + 1:]:
            if adj[i, j] > 0.5:
                pi, pj = pos[i], pos[j]
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#8F8F8F", lw=1.1, zorder=1)

    for i in valid:
        x, y = pos[i]
        ax.scatter([x], [y], s=430, c="#2F80C1", edgecolors="#222222", linewidths=0.9, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=4)

    ax.set_title("GT layout from node_coords / node_mask / adj_matrix", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x -> right/east")
    ax.set_ylabel("y -> up/north")
    ax.grid(True, color="#ECECEC", linewidth=0.8)


def draw_text(ax, row_index: int, row: Dict[str, Any]) -> None:
    ax.axis("off")
    valid = valid_indices(row)
    prompt = row.get("prompt", "")
    original = row.get("prompt_original", "")
    info = [
        layout_name(row_index, row),
        f"n_nodes={row.get('n_nodes')} | valid_mask_nodes={len(valid)} | adj_size={len(row.get('adj_matrix', []))}",
        "",
        "PROMPT:",
        textwrap.fill(prompt, width=74),
    ]
    if original and original != prompt:
        info.extend(["", "PROMPT_ORIGINAL:", textwrap.fill(original, width=74)])
    ax.text(
        0.0, 1.0, "\n".join(info),
        ha="left", va="top", fontsize=9.5,
        transform=ax.transAxes, family="DejaVu Sans",
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
