"""
将 final_graph_dataset_v3.jsonl 转换为 ChatHouseDiffusion 训练所需格式。

输出目录结构：
  data/chathousediffusion/chat_train/
    images/00000.png   ← 64×64 语义分割图（像素值=房间类型id）
    masks/00000.png    ← 64×64 建筑外轮廓二值图
    texts/00000.json   ← 房间 JSON（name/type/link/location/size）

房间类型映射（对齐 ChatHouseDiffusion 的 graph_encoder.py）：
  0=Unknown, 1=LivingRoom, 2=MasterRoom, 3=Kitchen, 4=Bathroom,
  5=DiningRoom, 6=CommonRoom（corridor）, 7=Storage, 8=Balcony, 9=Bedroom

用法：
  python convert_to_chathousediffusion.py \\
      --data  data/jsonl/final_graph_dataset_v3.jsonl \\
      --out   data/chathousediffusion/chat_train \\
      --workers 8
"""

import argparse
import json
import math
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

# ── 房间类型映射 ───────────────────────────────────────────────────────────────

# 我们数据集的类型名 → ChatHouseDiffusion 的类型名
OUR2CHAT = {
    "living_room": "LivingRoom",
    "bedroom":     "MasterRoom",   # 统一用 MasterRoom（最常见）
    "kitchen":     "Kitchen",
    "bathroom":    "Bathroom",
    "dining_room": "DiningRoom",
    "corridor":    "CommonRoom",
    "other":       "Storage",
}

# ChatHouseDiffusion 类型 → 像素值（id）
CHAT_TYPE2ID = {
    "Unknown":    0,
    "LivingRoom": 1,
    "MasterRoom": 2,
    "Kitchen":    3,
    "Bathroom":   4,
    "DiningRoom": 5,
    "CommonRoom": 6,
    "Storage":    7,
    "Balcony":    8,
}

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]

# 9方位
LOCATIONS = ["north", "northeast", "east", "southeast",
             "south", "southwest", "west", "northwest", "center"]

SIZE_THRESHOLDS = [0.05, 0.12, 0.22, 0.35]  # XS/S/M/L/XL 分界

IMG_SIZE = 64


# ── 平面图拓扑（复用 render.py 逻辑）────────────────────────────────────────────

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
    n = len(pts)
    area = 0.0
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


# ── 辅助计算 ──────────────────────────────────────────────────────────────────

def face_centroid(face, coords):
    xs = [coords[i][0] for i in face]
    ys = [coords[i][1] for i in face]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def face_area(face, coords):
    return abs(_signed_area(face, coords))


def compute_location(cx, cy, all_coords):
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span_x = mx_x - mn_x or 1
    span_y = mx_y - mn_y or 1
    rx = (cx - mn_x) / span_x  # 0=left 1=right
    ry = (cy - mn_y) / span_y  # 0=bottom 1=top

    # 映射到 9 方位
    if rx < 0.33:
        hdir = "west"
    elif rx < 0.67:
        hdir = "center"
    else:
        hdir = "east"

    if ry > 0.67:
        vdir = "north"
    elif ry > 0.33:
        vdir = "center"
    else:
        vdir = "south"

    if vdir == "center" and hdir == "center":
        return "center"
    if vdir == "center":
        return hdir
    if hdir == "center":
        return vdir
    return vdir + hdir  # e.g. "northwest"


def compute_size(area, total_area):
    ratio = area / (total_area or 1)
    if ratio < SIZE_THRESHOLDS[0]:
        return "XS"
    elif ratio < SIZE_THRESHOLDS[1]:
        return "S"
    elif ratio < SIZE_THRESHOLDS[2]:
        return "M"
    elif ratio < SIZE_THRESHOLDS[3]:
        return "L"
    return "XL"


def faces_adjacent(face_a, face_b, adj):
    """两个面是否共享相邻节点（即通过边相连）。"""
    for u in face_a:
        for v in face_b:
            if u != v and adj[u][v] == 1:
                return True
    return False


# ── 渲染 ──────────────────────────────────────────────────────────────────────

def to_px_fn(coords, img_size=IMG_SIZE):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.08

    def to_px(x, y):
        nx =     (x - mn_x + margin) / (span + 2 * margin)
        ny = 1 - (y - mn_y + margin) / (span + 2 * margin)
        return (int(nx * img_size), int(ny * img_size))
    return to_px


def render_semantic(coords, adj, faces, face_types, img_size=IMG_SIZE):
    """渲染语义分割图：每像素=房间类型id。"""
    to_px = to_px_fn(coords, img_size)
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    for face, room_type in zip(faces, face_types):
        chat_type = OUR2CHAT.get(room_type, "Storage")
        type_id = CHAT_TYPE2ID.get(chat_type, 0)
        pts = [to_px(*coords[i]) for i in face]
        draw.polygon(pts, fill=type_id)
    return img


def render_boundary(coords, adj, faces, img_size=IMG_SIZE):
    """渲染外轮廓 mask：建筑内部=0，外部=255。
    ChatHouseDiffusion feature_to_mask 对 <0.9 的像素返回1（扩散区域），
    所以内部必须是黑(0)，外部才能是白(255)被排除在外。
    """
    to_px = to_px_fn(coords, img_size)
    img = Image.new('L', (img_size, img_size), 255)  # 外部白色
    draw = ImageDraw.Draw(img)
    for face in faces:
        pts = [to_px(*coords[i]) for i in face]
        draw.polygon(pts, fill=0)  # 内部黑色
    return img


# ── 主转换函数 ────────────────────────────────────────────────────────────────

def convert_row(task):
    idx, row, out_img_dir, out_mask_dir, out_text_dir = task
    try:
        n = int(row['n_nodes'])
        coords = [(float(c[0]), float(c[1])) for c in row['node_coords'][:n]]
        adj    = [row['adj_matrix'][i][:n] for i in range(n)]
        node_types = []
        for k in range(n):
            t = row['node_types'][k]
            node_types.append(t if isinstance(t, list) else [t])

        faces = find_faces(coords, adj)
        if not faces:
            return False

        all_nbrs = _build_sorted_neighbors(coords, adj, n)
        face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]

        # ── 像素图 + mask ───────────────────────────────────────────────────
        sem_img  = render_semantic(coords, adj, faces, face_types)
        mask_img = render_boundary(coords, adj, faces)

        stem = f'{idx:05d}'
        sem_img .save(out_img_dir  / f'{stem}.png')
        mask_img.save(out_mask_dir / f'{stem}.png')

        # ── JSON ────────────────────────────────────────────────────────────
        total_area = sum(face_area(f, coords) for f in faces) or 1.0
        all_coords_list = list(coords)

        # 为每个面生成唯一名字
        type_counter = Counter()
        room_list = []
        for fi, (face, room_type) in enumerate(zip(faces, face_types)):
            chat_type = OUR2CHAT.get(room_type, "Storage")
            type_counter[chat_type] += 1
            name = f"{chat_type}_{type_counter[chat_type]}"

            cx, cy = face_centroid(face, coords)
            location = compute_location(cx, cy, all_coords_list)
            size     = compute_size(face_area(face, coords), total_area)

            room_list.append({
                'face_idx': fi,
                'name':     name,
                'type':     chat_type,
                'location': location,
                'size':     size,
            })

        # 计算 link（房间间邻接）
        for i, room in enumerate(room_list):
            links = []
            for j, other in enumerate(room_list):
                if i != j and faces_adjacent(faces[room['face_idx']],
                                              faces[other['face_idx']], adj):
                    links.append(other['name'])
            room['link'] = links

        # 按房间类型分组输出（graph_encoder.get_nodes 期望的格式）
        # {"LivingRoom": {"rooms": [...]}, "Kitchen": {"rooms": [...]}, ...}
        grouped: Dict[str, list] = {}
        for room in room_list:
            t = room['type']
            grouped.setdefault(t, []).append({
                'name':     room['name'],
                'link':     room['link'],
                'location': room['location'],
                'size':     room['size'],
            })
        json_out = {t: {'rooms': rooms} for t, rooms in grouped.items()}

        with open(out_text_dir / f'{stem}.json', 'w', encoding='utf-8') as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f'  [WARN] idx={idx} failed: {e}')
        return False


# ── 入口 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',    default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--out',     default='data/chathousediffusion/chat_train')
    p.add_argument('--suffix',  default='', help='子目录后缀，如 _test 生成验证集')
    p.add_argument('--workers', type=int, default=0, help='0=CPU核心数-1')
    p.add_argument('--max',     type=int, default=0,  help='最多处理条数，0=全量')
    return p.parse_args()


def main():
    args = parse_args()
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    out_root     = Path(args.out)
    out_img_dir  = out_root / f'images{args.suffix}'
    out_mask_dir = out_root / f'masks{args.suffix}'
    out_text_dir = out_root / f'texts{args.suffix}'
    for d in [out_img_dir, out_mask_dir, out_text_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f'读取: {args.data}')
    rows = []
    with open(args.data, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max > 0:
        rows = rows[:args.max]
    print(f'共 {len(rows)} 条，workers={n_workers}')

    tasks = [
        (i, row, out_img_dir, out_mask_dir, out_text_dir)
        for i, row in enumerate(rows)
    ]

    ok = 0
    results = []
    with Pool(processes=n_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(convert_row, tasks, chunksize=64)):
            if result:
                ok += 1
            results.append(result)
            if (done + 1) % 5000 == 0:
                print(f'  {done+1}/{len(tasks)}  ok={ok}', flush=True)

    # 生成 texts.csv（filename → json字符串）
    csv_path = out_text_dir / 'texts.csv'
    import csv as csv_mod
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv_mod.writer(csvf)
        writer.writerow(['0', '1'])
        for i in range(len(rows)):
            json_path = out_text_dir / f'{i:05d}.json'
            if json_path.exists():
                with open(json_path, encoding='utf-8') as jf:
                    content = jf.read().strip()
                writer.writerow([f'{i:05d}.png', content])
    print(f'CSV → {csv_path}')

    print(f'\n完成: ok={ok}/{len(tasks)}')
    print(f'输出 → {out_root}/')


if __name__ == '__main__':
    main()
