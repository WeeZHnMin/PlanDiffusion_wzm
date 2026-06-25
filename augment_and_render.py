"""
从 JSONL 数据集抽样，做坐标增强（旋转90°/180°/水平镜像），
渲染平面图 PNG（含房间标签，PIL 实现），输出中间 JSONL（prompt 留空待 captioning）。

流程：
    1. 抽 n_orig 条原始样本（默认 16667）
    2. 每条做 3 种变换：rot90 / rot180 / flip_h → 共约 5 万条
    3. 渲染成 PNG，房间多边形上标注房间类型（Bath/Bed/Kitchen 等）
    4. 保存中间 JSONL（含变换后 node_coords，prompt=""）

用法：
    python augment_and_render.py \\
        --data      data/jsonl/final_graph_dataset_v3.jsonl \\
        --out_img   data/rendered_aug \\
        --out_jsonl data/jsonl/aug_50k_graph.jsonl \\
        --n_orig 16667 --seed 42 --workers 8
"""

import argparse
import json
import math
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon as ShapelyPolygon

# ── 房间颜色（RGB）────────────────────────────────────────────────────────────
ROOM_COLORS_RGB = {
    "bathroom":    (174, 214, 241),
    "bedroom":     (215, 189, 226),
    "living_room": (250, 215, 160),
    "kitchen":     (169, 223, 191),
    "corridor":    (204, 209, 209),
    "dining_room": (249, 231, 159),
    "other":       (234, 237, 237),
}

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]

ROOM_LABELS = {
    "bathroom":    "Bath",
    "bedroom":     "Bed",
    "living_room": "Living",
    "kitchen":     "Kitchen",
    "corridor":    "Corridor",
    "dining_room": "Dining",
    "other":       "Other",
}

# 尝试加载系统字体，失败则用 PIL 默认字体
def _load_font(size=16):
    candidates = [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

_FONT = None

def get_font():
    global _FONT
    if _FONT is None:
        _FONT = _load_font(16)
    return _FONT


# ── 坐标增强 ──────────────────────────────────────────────────────────────────

def transform_coords(coords: List, mode: str) -> List:
    """
    mode: 'r90'  旋转 90° CCW:  (x,y) → (-y, x)
          'r180' 旋转 180°:     (x,y) → (-x, -y)
          'flip' 水平镜像:      (x,y) → (-x,  y)
    """
    result = []
    for xy in coords:
        x, y = float(xy[0]), float(xy[1])
        if mode == 'r90':
            result.append([-y, x])
        elif mode == 'r180':
            result.append([-x, -y])
        else:  # flip
            result.append([-x, y])
    return result


# ── 平面图拓扑（纯 Python）───────────────────────────────────────────────────

def _build_sorted_neighbors(coords, adj, n):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(coords[w][1] - coords[i][1],
                                     coords[w][0] - coords[i][0]),
        )
    return nbrs


def _next_half_edge(u, v, sorted_nbrs):
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face, coords):
    pts = [coords[i] for i in face]
    n = len(pts); area = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords, adj):
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited = set(); faces = []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face = []; cu, cv = u, v; steps = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv)); face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw; steps += 1
            if len(face) >= 3:
                faces.append(face)
    if not faces:
        return []
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


def vote_room_type(face, node_types, all_nbrs):
    face_set = set(face)
    face_counts = Counter()
    for node in face:
        for t in node_types[node]:
            face_counts[t] += 1
    if not face_counts:
        return "other"
    ext_counts = Counter()
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


# ── PIL 渲染（无文字）────────────────────────────────────────────────────────

def render_pil(coords_raw, adj_raw, node_types, n, out_path: Path, img_size=400):
    coords = [(float(xy[0]), float(xy[1])) for xy in coords_raw[:n]]
    adj    = [row[:n] for row in adj_raw[:n]]
    all_nbrs = _build_sorted_neighbors(coords, adj, n)

    try:
        faces      = find_faces(coords, adj)
        face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]
    except Exception:
        faces, face_types = [], []

    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    mn_x, mx_x = min(xs), max(xs); mn_y, mx_y = min(ys), max(ys)
    span = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def to_px(x, y):
        nx =     (x - mn_x + margin) / (span + 2 * margin)
        ny = 1 - (y - mn_y + margin) / (span + 2 * margin)  # PIL y 轴朝下
        return (int(nx * img_size), int(ny * img_size))

    img  = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = get_font()

    # 房间多边形 + 标签
    for face, room_type in zip(faces, face_types):
        pts   = [to_px(*coords[i]) for i in face]
        color = ROOM_COLORS_RGB.get(room_type, (234, 237, 237))
        draw.polygon(pts, fill=color, outline=(85, 85, 85))
        # 用 shapely representative_point 找标签位置（保证在多边形内）
        try:
            rp = ShapelyPolygon(pts).representative_point()
            cx, cy = int(rp.x), int(rp.y)
        except Exception:
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
        label = ROOM_LABELS.get(room_type, room_type)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 6, 12
        draw.text((cx - tw // 2, cy - th // 2), label, fill=(34, 34, 34), font=font)

    # 边
    for i in range(n):
        for j in all_nbrs[i]:
            if j > i:
                draw.line([to_px(*coords[i]), to_px(*coords[j])],
                          fill=(136, 136, 136), width=1)

    # 节点
    r = 4
    for i in range(n):
        x, y = to_px(*coords[i])
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(51, 51, 51))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ── 多进程工作函数 ────────────────────────────────────────────────────────────

def _worker(task):
    i, row, mode, img_path, = task
    n           = int(row['n_nodes'])
    orig_coords = row['node_coords']
    aug_coords  = transform_coords(orig_coords[:n], mode)
    full_coords = aug_coords + [[0, 0]] * (len(orig_coords) - n)

    node_types = []
    for k in range(n):
        t = row['node_types'][k]
        node_types.append(t if isinstance(t, list) else [t])

    try:
        render_pil(full_coords, row['adj_matrix'], node_types, n, Path(img_path))
        ok = True
    except Exception as e:
        ok = False
        print(f'  [WARN] render failed i={i} mode={mode}: {e}')

    jrow = {k: v for k, v in row.items()}
    jrow['node_coords'] = full_coords
    jrow['prompt']      = ""
    jrow['image']       = Path(img_path).name
    return ok, jrow


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',      default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--out_img',   default='data/rendered_aug')
    p.add_argument('--out_jsonl', default='data/jsonl/aug_50k_graph.jsonl')
    p.add_argument('--n_orig',    type=int, default=16667)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--workers',   type=int, default=0, help='0=CPU核心数-1')
    p.add_argument('--img_size',  type=int, default=400)
    return p.parse_args()


def main():
    args      = parse_args()
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    out_img   = Path(args.out_img)
    out_img.mkdir(parents=True, exist_ok=True)

    print(f'读取: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    print(f'共 {len(all_rows)} 条，抽取 {args.n_orig} 条')

    random.seed(args.seed)
    sampled = random.sample(all_rows, min(args.n_orig, len(all_rows)))

    MODES = ['r90', 'r180', 'flip']
    tasks = [
        (i, row, mode, str(out_img / f'{i:05d}_{mode}.png'))
        for i, row in enumerate(sampled)
        for mode in MODES
    ]
    total = len(tasks)
    print(f'共 {total} 条增强样本，workers={n_workers}')

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    with Pool(processes=n_workers) as pool, \
         open(out_jsonl, 'w', encoding='utf-8') as fout:
        for done, (ok, jrow) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=64)):
            if ok:
                fout.write(json.dumps(jrow, ensure_ascii=False) + '\n')
                ok_count += 1
            if (done + 1) % 2000 == 0:
                print(f'  {done+1}/{total}  ok={ok_count}', flush=True)

    print(f'\n完成：ok={ok_count}/{total}')
    print(f'PNG  → {out_img}/')
    print(f'JSONL→ {out_jsonl}')
    print(f'\n下一步：')
    print(f'  python caption_floorplans_mimo.py --img-dir {out_img} --out-file data/jsonl/aug_50k_captions.jsonl')


if __name__ == '__main__':
    main()
