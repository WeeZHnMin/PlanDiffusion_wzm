"""
Re-render all images in viz_50000 using the clean style:
- No title
- No adjacency lines
- Only colored room polygons with room name labels

Reads mapping.jsonl to locate each record's source, re-renders and overwrites the PNG.

Usage:
    python data/scripts/rerender_viz50000.py
"""

import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
IMG_DIR = DATA_DIR / "viz_50000"
MAP_FILE = IMG_DIR / "mapping.jsonl"


def load_source_index():
    """Load each train_jsonl file into a dict keyed by (filename, line_no)."""
    index = {}
    files = sorted(SRC_DIR.glob("train_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    for jf in files:
        with open(jf, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                index[(jf.name, line_no)] = json.loads(line)
    return index


def make_font():
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except Exception:
            pass
    return ImageFont.load_default()


def polygon_centroid(points):
    n = len(points)
    area = cx = cy = 0.0
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
    img_name, rec = args
    out_path = IMG_DIR / img_name

    rooms = rec.get("rooms", [])
    if not rooms:
        return

    all_x = [c[0] for r in rooms for c in r["coords"]]
    all_y = [c[1] for r in rooms for c in r["coords"]]
    if not all_x:
        return

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = span * 0.10 + 3

    world_x0, world_x1 = xmin - pad, xmax + pad
    world_y0, world_y1 = ymin - pad, ymax + pad
    world_w = max(world_x1 - world_x0, 1e-6)
    world_h = max(world_y1 - world_y0, 1e-6)
    scale = min((IMG_SIZE - 2 * MARGIN) / world_w, (IMG_SIZE - 2 * MARGIN) / world_h)

    canvas_w = int(world_w * scale + 2 * MARGIN)
    canvas_h = int(world_h * scale + 2 * MARGIN)
    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font = make_font()

    def to_px(x, y):
        return MARGIN + (x - world_x0) * scale, MARGIN + (world_y1 - y) * scale

    for room in rooms:
        rtype = room["type"]
        color = ROOM_COLORS.get(rtype, "#DDDDDD")
        pts_px = [to_px(c[0], c[1]) for c in room["coords"]]
        draw.polygon(pts_px, fill=color, outline="#444444")

        cx, cy = polygon_centroid([(c[0], c[1]) for c in room["coords"]])
        label = ROOM_ABBR.get(rtype, rtype)
        tx, ty = to_px(cx, cy)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text((tx - (right - left) / 2, ty - (bottom - top) / 2), label, fill="#222222", font=font)

    img.save(str(out_path), format="PNG", optimize=True)


if __name__ == "__main__":
    print("Loading source index...")
    source_index = load_source_index()
    print(f"Source records loaded: {len(source_index):,}")

    print(f"Reading mapping: {MAP_FILE}")
    tasks = []
    missing = 0
    with open(MAP_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["source_file"], row["source_line"])
            rec = source_index.get(key)
            if rec is None:
                missing += 1
                continue
            tasks.append((row["image"], rec))

    print(f"Tasks: {len(tasks):,}  Missing source: {missing}")

    n_workers = max(1, cpu_count() - 1)
    print(f"Re-rendering with {n_workers} processes...")

    done = 0
    with Pool(n_workers) as pool:
        for _ in pool.imap_unordered(render_one, tasks, chunksize=40):
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(tasks)}")

    print(f"\nDone: {done} images re-rendered in {IMG_DIR}")
