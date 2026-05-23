"""
Parse 测试数据.txt → type_data.jsonl
- 只保留 prompt 唯一的条目
- 每个房间前插入 SEP token（type=0，coord=(0,0)），用于分隔同类型房间
- 类别：0=SEP, 1=bathroom, 2=bedroom, 3=living_room, 4=kitchen, 5=corridor, 6=dining_room, 7=PAD
- 输出 type_seq、coord_seq（含 SEP 处的 (0,0)）、adjacency
"""

import re
import json
from pathlib import Path

SRC = Path(__file__).parent.parent / "测试数据.txt"
DST = Path(__file__).parent / "type_data.jsonl"

# 0=SEP，房间类别从 1 开始
SEP_TYPE = 0
ROOM_TYPES = {
    "bathroom":    1,
    "bedroom":     2,
    "living_room": 3,
    "kitchen":     4,
    "corridor":    5,
    "dining_room": 6,
}
PAD_TYPE   = 7
N_CLASSES  = 8   # 0=SEP, 1-6=rooms, 7=PAD

ROOM_TYPE_NAMES = ["SEP", "bathroom", "bedroom", "living_room",
                   "kitchen", "corridor", "dining_room", "PAD"]


def parse_layout(layout_str):
    rooms = []
    parts = re.split(r',\s*(?=[a-z_]+\s*:)', layout_str.strip())
    for part in parts:
        m = re.match(r'([a-z_]+)\s*:(.*)', part.strip())
        if not m:
            continue
        room_type = m.group(1)
        coord_str = m.group(2)
        coords = [[int(a), int(b)] for a, b in re.findall(r'\((\d+),(\d+)\)', coord_str)]
        if coords:
            rooms.append({"type": room_type, "coords": coords})
    return rooms


def infer_adjacency(rooms):
    """两个房间共享至少一个顶点 → 相邻"""
    n = len(rooms)
    coord_sets = [set(tuple(c) for c in room["coords"]) for room in rooms]
    adjacency  = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if coord_sets[i] & coord_sets[j]:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def point_on_segment(p, u, v):
    """
    判断点 p 是否严格在线段 uv 内部（不含端点）。
    所有坐标为整数，cross product 精确。
    """
    px, py = p; ux, uy = u; vx, vy = v
    # 共线检查
    cross = (vx - ux) * (py - uy) - (vy - uy) * (px - ux)
    if cross != 0:
        return False
    # 在线段范围内（严格内部）
    if ux != vx:
        t_num = px - ux
        t_den = vx - ux
    else:
        t_num = py - uy
        t_den = vy - uy
    if t_den == 0:
        return False
    # 0 < t < 1 严格内部
    return (0 < t_num < t_den) if t_den > 0 else (t_den < t_num < 0)


def build_vertex_adjacency(rooms):
    """
    构建坐标点级别的邻接关系。
    节点：所有唯一坐标点（去重）
    边规则：
      1. 多边形连续顶点之间相连（含首尾闭合）
      2. 若点 p 严格在边 (u,v) 内部，则 p 与 u、p 与 v 相连
    返回：
      vertices  : List[[x,y]]  唯一节点列表
      vertex_map: Dict[tuple→int]  坐标→节点索引
      seq_to_vtx: List[int]  coord_seq 中每个非SEP位置对应的节点索引（按房间顶点顺序）
      adj       : List[List[int]]  邻接表
    """
    # 1. 收集唯一节点
    coord_order = []
    for room in rooms:
        for c in room["coords"]:
            t = tuple(c)
            if t not in [tuple(x) for x in coord_order]:
                coord_order.append(list(c))
    # 去重保序
    seen_v = {}
    vertices = []
    for c in coord_order:
        t = tuple(c)
        if t not in seen_v:
            seen_v[t] = len(vertices)
            vertices.append(list(c))
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}
    n = len(vertices)

    # seq_to_vtx：按 coord_seq 顺序（跳过 SEP）映射到节点索引
    seq_to_vtx = []
    for room in rooms:
        for c in room["coords"]:
            seq_to_vtx.append(vertex_map[tuple(c)])

    # 2. 收集所有多边形边
    edges = set()
    for room in rooms:
        coords = room["coords"]
        m = len(coords)
        for k in range(m):
            u = tuple(coords[k])
            v = tuple(coords[(k + 1) % m])
            ui, vi = vertex_map[u], vertex_map[v]
            if ui != vi:
                edges.add((min(ui, vi), max(ui, vi)))

    # 3. 对每条边，找出所有严格在其内部的节点，按 t 值排序后分段连接
    #    这样边 A-B 若有中间节点 P，会被拆成 A-P 和 P-B，不保留直接的 A-B
    adj = [set() for _ in range(n)]
    for (ui, vi) in edges:
        u = vertices[ui]; v = vertices[vi]
        ux, uy = u;       vx, vy = v

        # 找所有严格在 (u,v) 内部的节点，并计算其 t 值用于排序
        intermediates = []
        for pi, p in enumerate(vertices):
            if pi == ui or pi == vi:
                continue
            if point_on_segment(p, u, v):
                # t ∈ (0,1)：p = u + t*(v-u)
                t = (p[0] - ux) / (vx - ux) if ux != vx else (p[1] - uy) / (vy - uy)
                intermediates.append((t, pi))

        # 按 t 值从小到大排序，构成链：ui → p1 → p2 → ... → vi
        intermediates.sort()
        chain = [ui] + [pi for _, pi in intermediates] + [vi]
        for k in range(len(chain) - 1):
            adj[chain[k]].add(chain[k + 1])
            adj[chain[k + 1]].add(chain[k])

    adj_list = [sorted(list(s)) for s in adj]
    return vertices, vertex_map, seq_to_vtx, adj_list


# ── 解析全部记录 ──────────────────────────────────────────────────────────────
all_records = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'\[User prompt\]\s*(.*?)\s*\[Layout\]\s*(.*)', line)
        if not m:
            continue
        prompt = m.group(1).strip()
        rooms  = parse_layout(m.group(2).strip())
        all_records.append({"prompt": prompt, "rooms": rooms})

# ── 去重 ──────────────────────────────────────────────────────────────────────
seen, unique_records = set(), []
for r in all_records:
    if r["prompt"] not in seen:
        seen.add(r["prompt"])
        unique_records.append(r)

print(f"Total: {len(all_records)}  →  Unique prompts: {len(unique_records)}")

# ── 构建带 SEP 的序列 ─────────────────────────────────────────────────────────
# 每个房间前插一个 SEP token（type=0，coord=(0,0)）
for r in unique_records:
    type_seq  = []
    coord_seq = []   # [[x, y], ...]，SEP 处为 [0, 0]
    for room in r["rooms"]:
        tid = ROOM_TYPES.get(room["type"], PAD_TYPE)
        # SEP token
        type_seq.append(SEP_TYPE)
        coord_seq.append([0, 0])
        # 房间顶点
        type_seq.extend([tid] * len(room["coords"]))
        coord_seq.extend(room["coords"])

    r["type_seq"]   = type_seq
    r["coord_seq"]  = coord_seq
    r["n_tokens"]   = len(type_seq)
    r["n_rooms"]    = len(r["rooms"])
    r["adjacency"]  = infer_adjacency(r["rooms"])

    # 坐标点级别的邻接图
    vertices, vertex_map, seq_to_vtx, adj_list = build_vertex_adjacency(r["rooms"])
    r["vertices"]    = vertices    # 唯一节点坐标列表
    r["seq_to_vtx"]  = seq_to_vtx # coord_seq 非SEP位置 → 节点索引
    r["vertex_adj"]  = adj_list    # 邻接表：adj_list[i] = 节点i的邻居索引列表

MAX_LEN = max(r["n_tokens"] for r in unique_records)
N_MAX   = max(len(r["vertices"]) for r in unique_records)
print(f"MAX_LEN (tokens with SEP) = {MAX_LEN},  N_MAX (nodes) = {N_MAX}")

# 为每条记录构建 N_MAX×N_MAX 的 01 邻接矩阵（含自环），存入 jsonl
for r in unique_records:
    n   = len(r["vertices"])
    adj = [[0] * N_MAX for _ in range(N_MAX)]
    for i in range(n):
        adj[i][i] = 1  # 自环
        for j in r["vertex_adj"][i]:
            adj[i][j] = 1
    r["adj_matrix"] = adj   # N_MAX×N_MAX list[list[int]]

for r in unique_records:
    pad = MAX_LEN - r["n_tokens"]
    r["type_seq_padded"]  = r["type_seq"]  + [PAD_TYPE] * pad
    r["coord_seq_padded"] = r["coord_seq"] + [[0, 0]]   * pad
    # mask=1 表示真实 token（含 SEP），mask=0 表示 padding
    r["token_mask"]       = [1] * r["n_tokens"] + [0] * pad
    # coord_mask=1 表示需要扩散的坐标（非 SEP、非 padding）
    r["coord_mask"]       = [1 if t not in (SEP_TYPE, PAD_TYPE) else 0
                              for t in r["type_seq_padded"]]

# ── 保存 ──────────────────────────────────────────────────────────────────────
DST.parent.mkdir(exist_ok=True)
with open(DST, "w", encoding="utf-8") as f:
    for r in unique_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(unique_records)} records → {DST}")
for i, r in enumerate(unique_records):
    print(f"  [{i+1:2d}] tokens={r['n_tokens']:2d}  rooms={r['n_rooms']}  {r['prompt'][:55]}")
