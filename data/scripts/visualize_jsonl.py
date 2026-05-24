"""
Sample N records from train_jsonl and render one PNG per record using Pillow.

Usage:
    python data/scripts/visualize_jsonl.py          # default 50000
    python data/scripts/visualize_jsonl.py 10000    # custom count
"""

import json
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
SEED = 42
IMG_SIZE = 640
MARGIN = 36

ROOM_COLORS = {
    "bathroom": "#AED6F1",
    "bedroom": "#A9DFBF",
    "living_room": "#F9E79F",
    "kitchen": "#F1948A",
    "corridor": "#D7BDE2",
    "dining_room": "#FAD7A0",
}
ROOM_ABBR = {
    "bathroom": "Bath",
    "bedroom": "Bed",
    "living_room": "Living",
    "kitchen": "Kitchen",
    "corridor": "Corridor",
    "dining_room": "Dining",
}

DATA_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = DATA_DIR / "Architext_v1" / "train_jsonl"
OUT_DIR = DATA_DIR / f"viz_{N_TOTAL}"
OUT_DIR.mkdir(exist_ok=True)
MAP_FILE = OUT_DIR / "mapping.jsonl"


def has_same_type_chain(rec, min_chain=3):
    rooms = rec["rooms"]
    adj = rec["adjacency"]
    types = [r["type"] for r in rooms]

    def dfs(node, rtype, length, visited):
        if length >= min_chain:
            return True
        for nb in adj[node]:
            if nb not in visited and types[nb] == rtype:
                visited.add(nb)
                if dfs(nb, rtype, length + 1, visited):
                    return True
                visited.remove(nb)
        return False

    for s in range(len(rooms)):
        if dfs(s, types[s], 1, {s}):
            return True
    return False


def load_all():
    records = []
    files = sorted(SRC_DIR.glob("train_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    for jf in files:
        with open(jf, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                rec = json.loads(line)
                if not has_same_type_chain(rec):
                    rec["_src_file"] = jf.name
                    rec["_src_line"] = line_no
                    records.append(rec)
    return records


def make_fonts():
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 16), ImageFont.truetype(name, 13)
        except Exception:
            pass
    fallback = ImageFont.load_default()
    return fallback, fallback


def polygon_centroid(points):
    n = len(points)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for k in range(n):
        x0, y0 = points[k]
        x1, y1 = points[(k + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    return cx / (6 * area), cy / (6 * area)


def render_one(args):
    idx, rec, out_path = args
    out_path = Path(out_path)
    if out_path.exists():
        return

    rooms = rec["rooms"]
    adj = rec["adjacency"]
    all_x = [c[0] for r in rooms for c in r["coords"]]
    all_y = [c[1] for r in rooms for c in r["coords"]]
    if not all_x:
        return

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = span * 0.10 + 3

    world_x0 = xmin - pad
    world_x1 = xmax + pad
    world_y0 = ymin - pad
    world_y1 = ymax + pad
    world_w = max(world_x1 - world_x0, 1e-6)
    world_h = max(world_y1 - world_y0, 1e-6)
    scale = min((IMG_SIZE - 2 * MARGIN) / world_w, (IMG_SIZE - 2 * MARGIN) / world_h)

    canvas_w = int(world_w * scale + 2 * MARGIN)
    canvas_h = int(world_h * scale + 2 * MARGIN)
    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font, title_font = make_fonts()

    def to_px(x, y):
        px = MARGIN + (x - world_x0) * scale
        py = MARGIN + (world_y1 - y) * scale
        return px, py

    centroids = []
    for room in rooms:
        rtype = room["type"]
        color = ROOM_COLORS.get(rtype, "#DDDDDD")
        pts_world = [(c[0], c[1]) for c in room["coords"]]
        pts_px = [to_px(x, y) for x, y in pts_world]
        draw.polygon(pts_px, fill=color, outline="#444444")

        cx, cy = polygon_centroid(pts_world)
        centroids.append((cx, cy))
        label = ROOM_ABBR.get(rtype, rtype)
        tx, ty = to_px(cx, cy)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw = right - left
        th = bottom - top
        draw.text((tx - tw / 2, ty - th / 2), label, fill="#222222", font=font)

    for i, neighbors in enumerate(adj):
        for j in neighbors:
            if j > i:
                x0, y0 = to_px(centroids[i][0], centroids[i][1])
                x1, y1 = to_px(centroids[j][0], centroids[j][1])
                draw.line((x0, y0, x1, y1), fill=(153, 153, 153, 160), width=1)

    prompt = rec.get("prompt", "")
    suffix = "..." if len(prompt) > 60 else ""
    title = f"[{idx + 1}] {prompt[:60]}{suffix}"
    draw.text((8, 8), title, fill="#333333", font=title_font)

    img.save(str(out_path), format="PNG", optimize=True)


if __name__ == "__main__":
    print("Loading and filtering records...")
    all_records = load_all()
    print(f"Valid records after filter: {len(all_records):,}")

    if len(all_records) < N_TOTAL:
        print(f"Warning: only {len(all_records)} valid records, less than requested {N_TOTAL}.")
        sampled = all_records
    else:
        random.seed(SEED)
        sampled = random.sample(all_records, N_TOTAL)
    print(f"Sampled: {len(sampled):,}, start rendering...\n")

    tasks = [(i, rec, str(OUT_DIR / f"{i + 1:05d}.png")) for i, rec in enumerate(sampled)]

    with open(MAP_FILE, "w", encoding="utf-8") as mf:
        for i, rec in enumerate(sampled, start=1):
            row = {
                "image": f"{i:05d}.png",
                "source_file": rec.get("_src_file", ""),
                "source_line": rec.get("_src_line", 0),
                "prompt": rec.get("prompt", ""),
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Mapping written: {MAP_FILE}")

    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} processes.")

    done = 0
    with Pool(n_workers) as pool:
        for _ in pool.imap_unordered(render_one, tasks, chunksize=40):
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)}")

    total_kb = sum(f.stat().st_size for f in OUT_DIR.glob("*.png")) // 1024
    print(f"\nDone: {len(sampled)} images, total {total_kb // 1024} MB")
    print(f"Output: {OUT_DIR}")
