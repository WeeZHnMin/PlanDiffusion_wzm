"""
根据节点坐标生成空间位置描述文本，替换原有的拓扑叙述 prompt。

核心逻辑：
  1. 每个节点有 node_types（可多类型）和坐标
  2. 按 primary_type 分组，用邻接矩阵找同类型连通分量 → 每个分量代表一个「房间实例」
  3. 计算每个房间实例的质心，映射到 5×5 网格 → 位置标签
  4. 生成: "Living room: center. Kitchen: top-right. Bedroom 1: top-left. Bedroom 2: bottom-right."

用法：
  python generate_spatial_prompts.py
  python generate_spatial_prompts.py --input data/jsonl/final_graph_dataset_v3.jsonl \
      --output data/jsonl/final_graph_dataset_v3_spatial.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

# ── 房间类型优先级与显示名 ─────────────────────────────────────────────────────
TYPE_PRIORITY = ['living_room', 'kitchen', 'bedroom', 'corridor', 'bathroom']
TYPE_DISPLAY  = {
    'bedroom':     'Bedroom',
    'bathroom':    'Bathroom',
    'kitchen':     'Kitchen',
    'living_room': 'Living room',
    'corridor':    'Corridor',
}


def primary_type(types):
    for t in TYPE_PRIORITY:
        if t in types:
            return t
    return types[0] if types else None


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
    ny_arr = (coords[:, 1] - y_min) / y_span

    # 按 primary_type 分组节点
    type_groups = {}
    for i in range(n):
        pt = primary_type(node_types[i])
        if pt is None:
            continue
        type_groups.setdefault(pt, []).append(i)

    # 每个类型内找连通分量（房间实例），计算质心
    room_instances = []   # (primary_type, centroid_nx, centroid_ny)
    for pt in TYPE_PRIORITY:
        if pt not in type_groups:
            continue
        comps = connected_components(type_groups[pt], adj)
        # 按质心位置排序（先 y 后 x，从上到下、从左到右）
        comp_centroids = []
        for comp in comps:
            xs = [nx_arr[i] for i in comp]
            ys = [ny_arr[i] for i in comp]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            comp_centroids.append((cy, cx, pt))
        comp_centroids.sort()
        for cy, cx, t in comp_centroids:
            room_instances.append((t, cx, cy))

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

    tokenizer = BertTokenizer.from_pretrained(args.bert) if args.stats else None

    t0 = time.perf_counter()
    n_written = 0
    token_lens = []

    with open(in_path, encoding='utf-8') as fin, \
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
