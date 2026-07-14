"""
根据节点坐标生成空间位置描述文本，替换原有的拓扑叙述 prompt。

核心逻辑：
  1. 每个节点有 node_types（可多类型）和坐标
  2. 用 node_coords + adj_matrix 做半边遍历，提取所有有界面 → 每个面代表一个「房间实例」
  3. 根据面内外 node_types 投票确定房间类型
  4. 计算每个房间实例的质心，映射到 5×5 网格 → 位置标签
  5. 生成: "Living room: center. Kitchen: top-right. Bedroom 1: top-left. Bedroom 2: bottom-right."

用法：
  python generate_spatial_prompts.py
  python generate_spatial_prompts.py --input data/jsonl/final_graph_dataset_v3.jsonl \
      --output data/jsonl/final_graph_dataset_v3_spatial.jsonl
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ── 房间类型优先级与显示名 ─────────────────────────────────────────────────────
TYPE_PRIORITY = ['living_room', 'kitchen', 'bedroom', 'corridor', 'bathroom', 'dining_room']
TYPE_DISPLAY  = {
    'bedroom':     'Bedroom',
    'bathroom':    'Bathroom',
    'kitchen':     'Kitchen',
    'living_room': 'Living room',
    'corridor':    'Corridor',
    'dining_room': 'Dining room',
}

# ── 5×5 位置标签 ──────────────────────────────────────────────────────────────
ROW_LABELS = ['top', 'upper', 'mid', 'lower', 'bottom']
COL_LABELS = ['far-left', 'left', 'center', 'right', 'far-right']


def position_label(nx, ny):
    """nx, ny ∈ [0,1]，左上角=(0,0)。"""
    col_i = min(int(nx * 5), 4)
    row_i = min(int(ny * 5), 4)
    row, col = ROW_LABELS[row_i], COL_LABELS[col_i]
    if row == 'mid' and col == 'center':
        return 'center'
    if row == 'mid':
        return col
    if col == 'center':
        return f'{row}-center'
    return f'{row}-{col}'


# ── 连通分量 ──────────────────────────────────────────────────────────────────

def connected_components(node_indices, adj):
    """
    在 adj 的子图（仅保留 node_indices 中的节点）中找连通分量。
    返回 list of list（每个子列表是一组连通节点索引）。
    """
    node_set = set(node_indices)
    visited   = set()
    comps     = []
    for start in node_indices:
        if start in visited:
            continue
        comp  = []
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb, connected in enumerate(adj[cur]):
                if connected and nb in node_set and nb not in visited:
                    stack.append(nb)
        comps.append(comp)
    return comps


def _build_sorted_neighbors(coords, adj, n):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > 0.5:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i].sort(
            key=lambda w: math.atan2(
                coords[w, 1] - coords[i, 1],
                coords[w, 0] - coords[i, 0],
            )
        )
    return nbrs


def _next_half_edge(u, v, sorted_nbrs):
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face, coords):
    area = 0.0
    for i, u in enumerate(face):
        v = face[(i + 1) % len(face)]
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_bounded_faces(coords, adj):
    """
    用半边遍历提取平面图有界面。
    一个有界面才对应一个真实房间候选，最大面积面视为外轮廓并剔除。
    """
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

    abs_areas = [abs(_signed_area(face, coords)) for face in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [face for i, face in enumerate(faces) if i != outer_idx]


def vote_face_type(face, node_types, adj):
    """
    给一个房间面投票确定类型。
    共享节点会带入相邻房间类型，因此用 face 内计数 / face 外一跳计数做去噪。
    """
    face_set = set(face)
    face_counts = Counter()
    for node in face:
        for t in node_types[node]:
            if t in TYPE_DISPLAY:
                face_counts[t] += 1

    if not face_counts:
        return None

    ext_counts = Counter()
    for node in face:
        for nb, connected in enumerate(adj[node]):
            if connected > 0.5 and nb not in face_set:
                for t in node_types[nb]:
                    if t in TYPE_DISPLAY:
                        ext_counts[t] += 1

    scores = {
        t: face_counts[t] / (ext_counts.get(t, 0) + 1)
        for t in face_counts
    }
    best = max(scores.values())
    winners = [t for t, score in scores.items() if score == best]
    order = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    return min(winners, key=lambda t: order.get(t, len(TYPE_PRIORITY)))


def fallback_room_instances(node_types, nx_arr, ny_arr, adj, n):
    """
    面提取失败时兜底。
    与旧逻辑不同，这里保留共享节点的全部类型，不再只取 primary_type。
    """
    type_groups = {}
    for i in range(n):
        for t in node_types[i]:
            if t in TYPE_DISPLAY:
                type_groups.setdefault(t, []).append(i)

    room_instances = []
    for pt in TYPE_PRIORITY:
        if pt not in type_groups:
            continue
        comps = connected_components(type_groups[pt], adj)
        for comp in comps:
            xs = [nx_arr[i] for i in comp]
            ys = [ny_arr[i] for i in comp]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            room_instances.append((pt, cx, cy))
    return room_instances


# ── 核心：生成描述 ────────────────────────────────────────────────────────────

def generate_prompt(node_types, node_coords, adj_matrix, n):
    coords = np.array(node_coords[:n], dtype=np.float32)
    adj    = np.array(adj_matrix, dtype=np.float32)[:n, :n]
    np.fill_diagonal(adj, 0)

    # 归一化坐标到 [0,1]（y 小→上方）
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    nx_arr = (coords[:, 0] - x_min) / x_span
    ny_arr = (y_max - coords[:, 1]) / y_span

    # 正确逻辑：先从平面图提取有界面，每个有界面才是一个房间实例。
    room_instances = []   # (room_type, centroid_nx, centroid_ny)
    for face in find_bounded_faces(coords, adj):
        t = vote_face_type(face, node_types, adj)
        if t is None:
            continue
        xs = [nx_arr[i] for i in face]
        ys = [ny_arr[i] for i in face]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        room_instances.append((t, cx, cy))

    if not room_instances:
        room_instances = fallback_room_instances(node_types, nx_arr, ny_arr, adj, n)

    # 类型顺序固定；同类型内部按上到下、左到右排序，便于 Bedroom 1/2 命名稳定。
    type_order = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    room_instances.sort(key=lambda item: (
        type_order.get(item[0], len(TYPE_PRIORITY)),
        item[2],
        item[1],
    ))

    # 生成文字
    type_counts = {}
    for t, _, _ in room_instances:
        type_counts[t] = type_counts.get(t, 0) + 1

    type_rank = {}
    parts = []
    for t, cx, cy in room_instances:
        disp  = TYPE_DISPLAY.get(t, t.replace('_', ' ').title())
        label = position_label(cx, cy)
        if type_counts[t] == 1:
            parts.append(f'{disp}: {label}')
        else:
            rank = type_rank.get(t, 1)
            type_rank[t] = rank + 1
            parts.append(f'{disp} {rank}: {label}')

    return '. '.join(parts) + '.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='data/jsonl/final_graph_dataset_v3.jsonl')
    parser.add_argument('--output',  default='data/jsonl/final_graph_dataset_v3_spatial.jsonl')
    parser.add_argument('--preview', type=int, default=5, help='打印前 N 条对比，0=不打印')
    parser.add_argument('--bert',    default='models/bert-base-uncased')
    parser.add_argument('--stats',   action='store_true', help='统计 token 长度分布')
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stats:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert)
    else:
        tokenizer = None

    t0 = time.perf_counter()
    n_written = 0
    token_lens = []

    with open(in_path, encoding='utf-8-sig') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n   = int(rec['n_nodes'])

            spatial_prompt         = generate_prompt(
                rec['node_types'], rec['node_coords'], rec['adj_matrix'], n)
            original_prompt        = rec.get('prompt', '')
            if ' [SEP] ' in original_prompt:
                original_prompt = original_prompt.split(' [SEP] ', 1)[1]
            full_prompt            = spatial_prompt + ' [SEP] ' + original_prompt
            rec['prompt_original'] = original_prompt
            rec['prompt']          = full_prompt
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_written += 1

            if tokenizer is not None:
                token_lens.append(len(tokenizer(full_prompt)['input_ids']))

            if args.preview > 0 and line_no < args.preview:
                print(f'=== sample {line_no} (n={n}) ===')
                print(full_prompt)
                print()

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f'  {line_no+1} 条  ({elapsed:.1f}s)', flush=True)

    elapsed = time.perf_counter() - t0
    print(f'完成: {n_written} 条 -> {out_path}  ({elapsed:.1f}s)')

    if token_lens:
        arr = np.array(token_lens)
        print(f'\nToken 长度统计 ({n_written} 条):')
        print(f'  mean={arr.mean():.1f}  median={np.median(arr):.0f}')
        print(f'  p95={np.percentile(arr, 95):.0f}  p99={np.percentile(arr, 99):.0f}  max={arr.max()}')


if __name__ == '__main__':
    main()
