"""
viz_labeled.py — 可视化 label_rooms_mimo 的输出结果

用法：
  python -m node_diffusion_room_tri.viz_labeled \
      --jsonl   outputs/tri_ddim200_full.jsonl \
      --labeled outputs/tri_ddim200_labeled.jsonl \
      --out     outputs/tri_ddim200_viz \
      --limit   10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

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


def render(coords, faces, face_types, prompt, out_path):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def norm(x, y):
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    for face, ftype in zip(faces, face_types):
        pts = [norm(*coords[i]) for i in face]
        poly = MplPolygon(
            pts, closed=True,
            facecolor=ROOM_COLORS.get(ftype, "#EAEDED"),
            edgecolor="#555555", linewidth=1.0, alpha=0.88, zorder=1,
        )
        ax.add_patch(poly)
        if HAS_SHAPELY:
            try:
                rp = ShapelyPolygon(pts).representative_point()
                cx, cy = rp.x, rp.y
            except Exception:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
        else:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, ROOM_LABELS.get(ftype, ftype),
                ha="center", va="center", fontsize=6.5,
                color="#222222", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title = (prompt[:90] + "…") if len(prompt) > 90 else prompt
    ax.set_title(title, fontsize=6, pad=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",   default="outputs/tri_ddim200_full.jsonl")
    p.add_argument("--labeled", default="outputs/tri_ddim200_labeled.jsonl")
    p.add_argument("--out",     default="outputs/tri_ddim200_viz")
    p.add_argument("--limit",   type=int, default=0, help="0=全量")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                full_rows[i] = json.loads(line)

    labeled = {}
    with open(args.labeled, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ok") and row.get("faces"):
                labeled[row["idx"]] = row

    idxs = sorted(labeled.keys())
    if args.limit > 0:
        idxs = idxs[:args.limit]

    print(f"渲染 {len(idxs)} 张...")
    ok = err = 0
    for idx in idxs:
        src = full_rows.get(idx)
        lab = labeled[idx]
        if src is None:
            print(f"[WARN] idx={idx} 在 full.jsonl 中找不到")
            continue
        try:
            n      = int(src["n_nodes"])
            coords = [(float(c[0]), float(c[1])) for c in src["pred_node_coords"][:n]]
            render(coords, lab["faces"], lab["face_types"],
                   src.get("prompt", ""), out_dir / f"{idx:05d}.png")
            ok += 1
            print(f"  [{ok}/{len(idxs)}] {idx:05d}.png")
        except Exception as e:
            err += 1
            print(f"  [ERR] idx={idx}: {e}")

    print(f"完成: ok={ok} err={err} → {out_dir}/")


if __name__ == "__main__":
    main()
