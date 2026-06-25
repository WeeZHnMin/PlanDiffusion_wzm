"""
从 JSONL 数据集抽样，做坐标增强（旋转90°/180°/水平镜像），
渲染平面图 PNG（无文字），输出中间 JSONL（prompt 留空，待后续 captioning 填充）。

流程：
    1. 抽 n_orig 条原始样本（默认 16667）
    2. 每条做 3 种变换：rot90 / rot180 / flip_h → 共约 5 万条
    3. 渲染成 PNG，不含文字标注
    4. 保存中间 JSONL（含变换后 node_coords，prompt=""）

用法：
    python augment_and_render.py \\
        --data   data/jsonl/final_graph_dataset_v3.jsonl \\
        --out_img  data/rendered_aug \\
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
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon

# ── 房间颜色（与 render.py 一致）──────────────────────────────────────────────
ROOM_COLORS = {
    "bathroom":    "#AED6F1",
    "bedroom":     "#D7BDE2",
    "living_room": "#FAD7A0",
    "kitchen":     "#A9DFBF",
    "corridor":    "#CCD1D1",
    "dining_room": "#F9E79F",
    "other":       "#EAEDED",
}

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]


# ── 坐标增强 ──────────────────────────────────────────────────────────────────

def transform_coords(coords: List[List[float]], mode: str) -> List[List[float]]:
    """
    对有效节点坐标做增强变换（坐标已中心化）。
    mode: 'r90'   旋转 90° CCW:  (x,y) → (-y, x)
          'r180'  旋转 180°:     (x,y) → (-x, -y)
          'flip'  水平镜像:      (x,y) → (-x,  y)
    """
    result = []
    for xy in coords:
        x, y = float(xy[0]), float(xy[1])
        if mode == 'r90':
            result.append([-y, x])
        elif mode == 'r180':
            result.append([-x, -y])
        elif mode == 'flip':
            result.append([-x, y])
        else:
            result.append([x, y])
    return result


# ── 平面图渲染（无文字）──────────────────────────────────────────────────────

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
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords, adj):
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited = set()
    faces = []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face = []
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
    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]
    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))


def render_no_text(coords_raw, adj_raw, node_types, n, out_path: Path):
    """渲染平面图，不含任何文字标注。"""
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
    span   = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def norm(x, y):
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal"); ax.axis("off")

    for face, room_type in zip(faces, face_types):
        pts  = [norm(*coords[i]) for i in face]
        poly = MplPolygon(pts, closed=True,
                          facecolor=ROOM_COLORS.get(room_type, "#EAEDED"),
                          edgecolor="#555555", linewidth=0.9,
                          alpha=0.88, zorder=1)
        ax.add_patch(poly)

    for i in range(n):
        for j in all_nbrs[i]:
            if j > i:
                x0, y0 = norm(*coords[i])
                x1, y1 = norm(*coords[j])
                ax.plot([x0, x1], [y0, y1], color="#888888", lw=0.7, zorder=2)

    for i in range(n):
        x, y = norm(*coords[i])
        ax.plot(x, y, "o", color="#333333", markersize=3.0, zorder=4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 多进程工作函数 ────────────────────────────────────────────────────────────

def _worker(task):
    idx, row, mode, img_path, jsonl_row = task
    n          = int(row['n_nodes'])
    orig_coords = row['node_coords']            # [40, 2]（含 padding）
    aug_coords  = transform_coords(orig_coords[:n], mode)  # 只变换有效节点
    # padding 部分保持 [0,0]
    full_coords = aug_coords + [[0, 0]] * (len(orig_coords) - n)

    node_types  = [row['node_types'][i] if isinstance(row['node_types'][i], list)
                   else [row['node_types'][i]] for i in range(n)]

    try:
        render_no_text(full_coords, row['adj_matrix'], node_types, n, Path(img_path))
        ok = True
    except Exception as e:
        ok = False
        print(f'  [WARN] render failed idx={idx} mode={mode}: {e}')

    jsonl_row['node_coords'] = full_coords
    jsonl_row['prompt']      = ""
    jsonl_row['image']       = Path(img_path).name
    return ok, jsonl_row, img_path


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',      default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--out_img',   default='data/rendered_aug')
    p.add_argument('--out_jsonl', default='data/jsonl/aug_50k_graph.jsonl')
    p.add_argument('--n_orig',    type=int, default=16667, help='原始样本数（×3 = 总增强数）')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--workers',   type=int, default=0, help='0=CPU核心数')
    return p.parse_args()


def main():
    args    = parse_args()
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    out_img = Path(args.out_img)
    out_img.mkdir(parents=True, exist_ok=True)

    print(f'读取数据: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    print(f'共 {len(all_rows)} 条，抽取 {args.n_orig} 条')

    random.seed(args.seed)
    sampled = random.sample(all_rows, min(args.n_orig, len(all_rows)))

    MODES = ['r90', 'r180', 'flip']
    MODE_SUFFIX = {'r90': 'r90', 'r180': 'r180', 'flip': 'flip'}

    tasks = []
    for i, row in enumerate(sampled):
        for mode in MODES:
            img_name  = f'{i:05d}_{MODE_SUFFIX[mode]}.png'
            img_path  = out_img / img_name
            jsonl_row = {k: v for k, v in row.items()}   # shallow copy
            tasks.append((i, row, mode, str(img_path), jsonl_row))

    total = len(tasks)
    print(f'共 {total} 条增强样本，workers={n_workers}')

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    with Pool(processes=n_workers) as pool, \
         open(out_jsonl, 'w', encoding='utf-8') as fout:
        for done, (ok, jrow, _) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=32)):
            if ok:
                fout.write(json.dumps(jrow, ensure_ascii=False) + '\n')
                ok_count += 1
            if (done + 1) % 1000 == 0:
                print(f'  {done+1}/{total}  ok={ok_count}', flush=True)

    print(f'\n完成：ok={ok_count}/{total}')
    print(f'PNG → {out_img}/')
    print(f'JSONL → {out_jsonl}')
    print(f'\n下一步：运行 caption_floorplans_mimo.py --img-dir {out_img} --out-file data/jsonl/aug_50k_captions.jsonl')


if __name__ == '__main__':
    main()
