"""
可视化一条数据的多次增强结果，验证 tokens / coords / types / adj 对齐正确。
运行：python verify_augment_visual.py
输出：verify_augment_visual.png
"""

import json
import random
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

# Windows 中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sys.stdout.reconfigure(encoding='utf-8')

# ── 从词表文件加载常量 ────────────────────────────────────────
_vocab = json.load(open('data/processed/type_combo_vocab_old.json', encoding='utf-8'))
PAD_ID      = 0
TOK_OPEN    = _vocab['TOK_OPEN']
TOK_CLOSE   = _vocab['TOK_CLOSE']
TOK_BREAK   = _vocab['TOK_BREAK']
BOS_ID      = _vocab['BOS_ID']
EOS_ID      = _vocab['EOS_ID']
NODE_OFFSET = _vocab['NODE_OFFSET']
MAX_NODES   = _vocab['MAX_NODES']
MAX_LEN     = 256

# combo_to_id: frozenset -> int
COMBO_TO_ID = {frozenset(map(int, k.strip('[]').split(', '))): v
               for k, v in _vocab['combo_to_id'].items()}

BASE_NAMES  = {int(k): v for k, v in _vocab['base_type_names'].items()}

# 节点颜色：按所属房间类型集合混合（单类型直接映射，多类型取第一个主色加虚线边框）
BASE_COLORS = {1:'#4FC3F7', 2:'#FFB74D', 3:'#AED581',
               4:'#F06292', 5:'#CE93D8', 6:'#FF8A65', 7:'#B0BEC5'}

ROOM_TYPE_IDS = {'bathroom':1,'bedroom':2,'living_room':3,
                 'kitchen':4,'corridor':5,'dining_room':6}
def room_type_id(s): return ROOM_TYPE_IDS.get(s, 7)


def get_vertex_combos(data):
    coord_to_types = defaultdict(set)
    for room in data['rooms']:
        tid = room_type_id(room['type'])
        for coord in room['coords']:
            coord_to_types[(coord[0], coord[1])].add(tid)
    combos = []
    for v in data['vertices']:
        key = (v[0], v[1])
        types = coord_to_types.get(key)
        combos.append(frozenset(types) if types else frozenset([7]))
    return combos


def combo_color(combo):
    """多类型组合取第一个基础类型的颜色，边框加深表示共享"""
    return BASE_COLORS[min(combo)]


def combo_label(combo):
    if len(combo) == 1:
        return BASE_NAMES[next(iter(combo))][:3]
    return '+'.join(BASE_NAMES[t][:3] for t in sorted(combo))


def sample_sent(adj, n, seed=None):
    if seed is not None:
        random.seed(seed)
    neighbors = defaultdict(set)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                neighbors[i].add(j)
    unvisited = set(range(n))
    all_nodes = set(range(n))
    v = random.choice(list(unvisited))
    unvisited.remove(v)
    current_trail = [(v, set())]
    sent = []
    while unvisited:
        unvisited_nbrs = neighbors[v] & unvisited
        if not unvisited_nbrs:
            sent.append(current_trail)
            v = random.choice(list(unvisited))
            unvisited.remove(v)
            visited = all_nodes - unvisited
            A = neighbors[v] & visited
            current_trail = [(v, A)]
        else:
            u = random.choice(list(unvisited_nbrs))
            unvisited.remove(u)
            visited = all_nodes - unvisited
            A = (neighbors[u] - {v}) & visited
            current_trail.append((u, A))
            v = u
    sent.append(current_trail)
    return sent


def sent_to_tokens_typed(sent, combo_ids):
    """combo_ids: list[int], 每个节点的组合类型 token ID"""
    tokens = []
    node_to_id = {}
    next_id = [1]
    visit_order = []

    def get_id(node):
        if node not in node_to_id:
            node_to_id[node] = next_id[0]
            next_id[0] += 1
            visit_order.append(node)
        return node_to_id[node]

    for seg_idx, trail in enumerate(sent):
        if seg_idx > 0:
            tokens.append(TOK_BREAK)
        for node, nbrs in trail:
            v_id   = get_id(node)
            v_type = combo_ids[node]
            nbr_ids = sorted(get_id(u) for u in nbrs)
            tokens.append(NODE_OFFSET + v_id)
            tokens.append(v_type)
            tokens.append(TOK_OPEN)
            tokens.extend(NODE_OFFSET + nid for nid in nbr_ids)
            tokens.append(TOK_CLOSE)
    return tokens, visit_order


def reorder_by_visit(visit_order, coords, combo_ids, adj, n):
    coords_r   = np.zeros((n, 2), dtype=np.float32)
    comboids_r = np.zeros(n, dtype=np.int32)
    for new_idx, orig_idx in enumerate(visit_order):
        coords_r[new_idx]   = coords[orig_idx]
        comboids_r[new_idx] = combo_ids[orig_idx]
    adj_r = np.zeros((n, n), dtype=np.float32)
    for ni, oi in enumerate(visit_order):
        for nj, oj in enumerate(visit_order):
            adj_r[ni, nj] = adj[oi, oj]
    return coords_r, comboids_r, adj_r


def tokens_to_str(tokens):
    """把token序列转成可读字符串（只显示结构部分）"""
    sym = {TOK_OPEN:'<', TOK_CLOSE:'>', TOK_BREAK:'/', BOS_ID:'[BOS]', EOS_ID:'[EOS]'}
    parts = []
    for t in tokens:
        if t == BOS_ID:
            parts.append('[BOS]')
        elif t == EOS_ID:
            parts.append('[EOS]')
        elif t == TOK_OPEN:
            parts.append('<')
        elif t == TOK_CLOSE:
            parts.append('>')
        elif t == TOK_BREAK:
            parts.append('/')
        elif 1 <= t <= 7:
            parts.append(f'T{t}')
        elif t > NODE_OFFSET:
            parts.append(str(t - NODE_OFFSET))
    return ' '.join(parts)


# ── 加载第一条数据 ─────────────────────────────────────────────
SRC = Path('data/jsonl/mapped_type_data_zh.jsonl')
with SRC.open(encoding='utf-8') as f:
    data = json.loads(f.readline())

n          = len(data['vertices'])
coords_raw = np.array(data['vertices'],   dtype=np.float32)
adj_raw    = np.array(data['adj_matrix'], dtype=np.float32)
adj_n      = adj_raw[:n, :n].copy()
np.fill_diagonal(adj_n, 0)

# 用 combo 方式重建每个顶点的类型
combos_raw = get_vertex_combos(data)
combo_ids  = np.array([COMBO_TO_ID[c] for c in combos_raw], dtype=np.int32)

print(f"节点数: {n}")
print(f"prompt: {data['prompt'][:60]}...")

# ── 跑 3 次增强 ────────────────────────────────────────────────
N_AUG = 3
random.seed(0)
results = []
for aug_i in range(N_AUG):
    sent = sample_sent(adj_n, n)
    tokens, visit_order = sent_to_tokens_typed(sent, combo_ids)
    full_tokens = [BOS_ID] + tokens + [EOS_ID]
    coords_r, comboids_r, adj_r = reorder_by_visit(visit_order, coords_raw, combo_ids, adj_n, n)
    results.append({
        'visit_order': visit_order,
        'tokens':      full_tokens,
        'coords':      coords_r,
        'combo_ids':   comboids_r,
        'combos':      [combos_raw[i] for i in visit_order],
        'adj':         adj_r,
    })

# ── 画图 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_AUG + 1, figsize=(5 * (N_AUG + 1), 6))
fig.suptitle(f'数据增强验证  (n={n} 节点)\n{data["prompt"][:50]}...', fontsize=10)


def draw_graph(ax, coords, combos, adj, n, title, label_prefix='orig'):
    """画节点+边，节点按组合类型着色（多类型节点用双圈标注）"""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        'k-', linewidth=0.8, alpha=0.5, zorder=1)
    for i in range(n):
        combo  = combos[i]
        color  = combo_color(combo)
        is_multi = len(combo) > 1
        edgecol  = 'red' if is_multi else 'k'
        lw       = 1.5  if is_multi else 0.5
        ax.scatter(coords[i, 0], coords[i, 1], s=200, c=color,
                   edgecolors=edgecol, linewidths=lw, zorder=2)
        label = str(i + 1) if label_prefix == 'new' else str(i)
        ax.text(coords[i, 0], coords[i, 1], label,
                ha='center', va='center', fontsize=6, zorder=3, fontweight='bold')
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')


# 原始图（用原始节点编号）
draw_graph(axes[0], coords_raw, combos_raw, adj_n, n,
           '原始图\n(红圈=多类型共享节点)', label_prefix='orig')

# 图例
legend_patches = [mpatches.Patch(color=BASE_COLORS[k], label=BASE_NAMES[k])
                  for k in sorted(BASE_COLORS)]
legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='red',
                                     linewidth=1.5, label='多类型共享'))
axes[0].legend(handles=legend_patches, loc='lower left', fontsize=5, framealpha=0.7)

# 每次增强
for aug_i, res in enumerate(results):
    ax = axes[aug_i + 1]
    draw_graph(ax, res['coords'], res['combos'], res['adj'], n,
               f'增强 #{aug_i + 1}\n游走顺序: {res["visit_order"][:5]}...',
               label_prefix='new')
    # 在底部显示token序列（截断显示）
    tok_str = tokens_to_str(res['tokens'])
    if len(tok_str) > 80:
        tok_str = tok_str[:80] + '...'
    ax.text(0.5, -0.02, tok_str, transform=ax.transAxes,
            fontsize=5, ha='center', va='top', wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('verify_augment_visual.png', dpi=150, bbox_inches='tight')
print('已保存 -> verify_augment_visual.png')

# ── 文字验证：三次增强的边集合是否一致 ───────────────────────────
def adj_to_edges(adj, n):
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.add((i, j))
    return edges

orig_edges = adj_to_edges(adj_n, n)
print(f'\n原始边数: {len(orig_edges)}')
for aug_i, res in enumerate(results):
    aug_edges = adj_to_edges(res['adj'], n)
    match = 'OK' if len(aug_edges) == len(orig_edges) else 'NG'
    print(f'增强 #{aug_i+1}: 边数={len(aug_edges)} {match}  '
          f'visit_order前5={res["visit_order"][:5]}')

# 验证 coords 和 types 对应关系
print('\n节点对应关系验证（增强#1为例）:')
res0 = results[0]
for new_idx, orig_idx in enumerate(res0['visit_order']):
    c_orig  = coords_raw[orig_idx]
    c_new   = res0['coords'][new_idx]
    id_orig = combo_ids[orig_idx]
    id_new  = res0['combo_ids'][new_idx]
    coord_ok = np.allclose(c_orig, c_new)
    type_ok  = id_orig == id_new
    status   = 'OK' if (coord_ok and type_ok) else 'NG'
    combo    = combos_raw[orig_idx]
    names    = '+'.join(BASE_NAMES[t] for t in sorted(combo))
    print(f'  新编号{new_idx+1} <- 原始{orig_idx}  {status}  '
          f'({c_orig[0]:.0f},{c_orig[1]:.0f})  [{names}]')
