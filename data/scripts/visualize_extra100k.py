"""
Generate 100,000 additional images from train_jsonl, skipping any record
already used in viz_50000 (matched by source_file + source_line).

Usage:
    python data/scripts/visualize_extra100k.py
"""

import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

N_TOTAL = 100000
SEED = 123
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
EXISTING_MAP = DATA_DIR / "viz_50000" / "mapping.jsonl"
OUT_DIR = DATA_DIR / "viz_100000"
OUT_DIR.mkdir(exist_ok=True)
MAP_FILE = OUT_DIR / "mapping.jsonl"


def load_used_set():
    used = set()
    if not EXISTING_MAP.exists():
        print(f"Warning: {EXISTING_MAP} not found, no exclusions applied.")
        return used
    with open(EXISTING_MAP, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            used.add((row["source_file"], row["source_line"]))
    print(f"Loaded {len(used):,} already-used records to exclude.")
    return used


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


def load_all(used_set):
    records = []
    files = sorted(SRC_DIR.glob("train_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    for jf in files:
        with open(jf, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if (jf.name, line_no) in used_set:
                    continue
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

    for room in rooms:
        rtype = room["type"]
        color = ROOM_COLORS.get(rtype, "#DDDDDD")
        pts_world = [(c[0], c[1]) for c in room["coords"]]
        pts_px = [to_px(x, y) for x, y in pts_world]
        draw.polygon(pts_px, fill=color, outline="#444444")

        cx, cy = polygon_centroid(pts_world)
        label = ROOM_ABBR.get(rtype, rtype)
        tx, ty = to_px(cx, cy)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw = right - left
        th = bottom - top
        draw.text((tx - tw / 2, ty - th / 2), label, fill="#222222", font=font)

    img.save(str(out_path), format="PNG", optimize=True)


if __name__ == "__main__":
    used_set = load_used_set()

    print("Loading and filtering records (excluding already-used)...")
    all_records = load_all(used_set)
    print(f"Available records after exclusion and filter: {len(all_records):,}")

    if len(all_records) < N_TOTAL:
        print(f"Warning: only {len(all_records)} available, less than requested {N_TOTAL}.")
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
            if done % 5000 == 0:
                print(f"  {done}/{len(tasks)}")

    total_kb = sum(f.stat().st_size for f in OUT_DIR.glob("*.png")) // 1024
    print(f"\nDone: {len(sampled)} images, total {total_kb // 1024} MB")
    print(f"Output: {OUT_DIR}")
