"""
把 Architext_v1/train/ 下的每个 train_N.txt 转换为 train_N.jsonl
按 layout 坐标全局去重（跨文件），相同坐标只保留第一条。

去重 key：将每个房间的顶点集合 + 类型 做规范化后取 hash，
          与房间顺序、顶点顺序无关。

每条记录包含：
  prompt      : str
  rooms       : [{type, coords}]
  type_seq    : [int]  含SEP(0)
  coord_seq   : [[x,y]]
  n_tokens    : int
  n_rooms     : int
  adjacency   : [[int]]  房间级邻接表
  vertices    : [[x,y]]  去重后的节点坐标
  vertex_adj  : [[int]]  节点级邻接表（已切分中间节点）
"""

import re
import json
import hashlib
from pathlib import Path

SEP_TYPE   = 0
ROOM_TYPES = {
    "bathroom":    1,
    "bedroom":     2,
    "living_room": 3,
    "kitchen":     4,
    "corridor":    5,
    "dining_room": 6,
}
PAD_TYPE = 7

DATA_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = DATA_DIR / "Architext_v1" / "train"
DST_DIR = DATA_DIR / "Architext_v1" / "train_jsonl"
DST_DIR.mkdir(exist_ok=True)


# ── 去重 key：与房间顺序、顶点顺序无关 ────────────────────────────────────────

def layout_hash(rooms):
    """
    规范化 layout：每个房间 → (type, frozenset of coord tuples)
    对所有房间排序后序列化，取 MD5 前 16 字节作为去重 key。
    """
    canonical = sorted(
        (room["type"], tuple(sorted(tuple(c) for c in room["coords"])))
        for room in rooms
    )
    return hashlib.md5(str(canonical).encode()).digest()


# ── 解析 ──────────────────────────────────────────────────────────────────────

def parse_layout(layout_str):
    rooms = []
    parts = re.split(r',\s*(?=[a-z_]+\s*:)', layout_str.strip())
    for part in parts:
        m = re.match(r'([a-z_]+)\s*:(.*)', part.strip())
        if not m:
            continue
        room_type = m.group(1)
        coords = [[int(a), int(b)]
                  for a, b in re.findall(r'\((\d+),(\d+)\)', m.group(2))]
        if coords:
            rooms.append({"type": room_type, "coords": coords})
    return rooms


def infer_adjacency(rooms):
    n = len(rooms)
    coord_sets = [set(tuple(c) for c in r["coords"]) for r in rooms]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if coord_sets[i] & coord_sets[j]:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def point_on_segment(p, u, v):
    px, py = p; ux, uy = u; vx, vy = v
    cross = (vx - ux) * (py - uy) - (vy - uy) * (px - ux)
    if cross != 0:
        return False
    t_num = (px - ux) if ux != vx else (py - uy)
    t_den = (vx - ux) if ux != vx else (vy - uy)
    if t_den == 0:
        return False
    return (0 < t_num < t_den) if t_den > 0 else (t_den < t_num < 0)


def build_vertex_graph(rooms):
    seen, vertices = {}, []
    for room in rooms:
        for c in room["coords"]:
            t = tuple(c)
            if t not in seen:
                seen[t] = len(vertices)
                vertices.append(list(c))
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}
    n = len(vertices)

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

    adj = [set() for _ in range(n)]
    for (ui, vi) in edges:
        u, v = vertices[ui], vertices[vi]
        ux, uy = u; vx, vy = v
        intermediates = []
        for pi, p in enumerate(vertices):
            if pi == ui or pi == vi:
                continue
            if point_on_segment(p, u, v):
                t = (p[0] - ux) / (vx - ux) if ux != vx else (p[1] - uy) / (vy - uy)
                intermediates.append((t, pi))
        intermediates.sort()
        chain = [ui] + [pi for _, pi in intermediates] + [vi]
        for k in range(len(chain) - 1):
            adj[chain[k]].add(chain[k + 1])
            adj[chain[k + 1]].add(chain[k])

    return vertices, [sorted(list(s)) for s in adj]


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    m = re.match(r'\[User prompt\]\s*(.*?)\s*\[Layout\]\s*(.*)', line)
    if not m:
        return None
    prompt = m.group(1).strip()
    rooms  = parse_layout(m.group(2).strip())
    if not rooms:
        return None

    type_seq, coord_seq = [], []
    for room in rooms:
        tid = ROOM_TYPES.get(room["type"], PAD_TYPE)
        type_seq.append(SEP_TYPE)
        coord_seq.append([0, 0])
        type_seq.extend([tid] * len(room["coords"]))
        coord_seq.extend(room["coords"])

    vertices, vertex_adj = build_vertex_graph(rooms)

    return {
        "prompt":     prompt,
        "rooms":      rooms,
        "type_seq":   type_seq,
        "coord_seq":  coord_seq,
        "n_tokens":   len(type_seq),
        "n_rooms":    len(rooms),
        "adjacency":  infer_adjacency(rooms),
        "vertices":   vertices,
        "vertex_adj": vertex_adj,
    }


# ── 主循环（全局去重 set）────────────────────────────────────────────────────

txt_files = sorted(SRC_DIR.glob("train_*.txt"),
                   key=lambda p: int(re.search(r'\d+', p.stem).group()))

print(f"找到 {len(txt_files)} 个 txt 文件，开始转换（全局 layout 去重）...\n")

seen_layouts = set()   # 跨文件全局去重，存 MD5 bytes（每条 16 字节）
total_ok = total_dup = total_skip = 0

for txt_path in txt_files:
    dst_path = DST_DIR / (txt_path.stem + ".jsonl")
    ok = dup = skip = 0

    with open(txt_path, encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = parse_line(line)
            if rec is None:
                skip += 1
                continue

            h = layout_hash(rec["rooms"])
            if h in seen_layouts:
                dup += 1
                continue
            seen_layouts.add(h)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

    total_ok += ok; total_dup += dup; total_skip += skip
    print(f"  {txt_path.name}  写出 {ok:,}  重复跳过 {dup:,}  解析失败 {skip}")

print(f"\n完成！")
print(f"  写出：{total_ok:,} 条")
print(f"  去重：{total_dup:,} 条")
print(f"  失败：{total_skip} 条")
print(f"  输出：{DST_DIR}")
