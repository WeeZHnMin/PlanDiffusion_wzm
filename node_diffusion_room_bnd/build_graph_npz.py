"""
构建 node_diffusion_room_bnd（四流注意力 + 建筑轮廓条件版）训练用 NPZ。

相比 node_diffusion_room_tri 额外保存：
  is_boundary  : (N, MAX_NODES)  float32，1=轮廓节点（外边界），0=内部节点

外边界节点通过半边（half-edge）算法找到：遍历所有面，有向面积最大的面即外轮廓面，
其节点集合即为轮廓节点。

用法：
  python -m node_diffusion_room_bnd.build_graph_npz
  python -m node_diffusion_room_bnd.build_graph_npz --augment 8 --workers 8
  # 小训练集
  python -m node_diffusion_room_bnd.build_graph_npz \\
      --max_samples 5000 \\
      --output data/processed/node_diffusion_room_bnd/graph_dataset_5k.npz
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

from .model import _assign_room_membership_single, MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192


# ── 半边算法辅助函数 ──────────────────────────────────────────────────────────

def _build_sorted_neighbors(coords, adj, n):
    """按极角对每个节点的邻居排序，返回 dict {u: [v0, v1, ...]}。"""
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j]:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(
                coords[w][1] - coords[i][1],
                coords[w][0] - coords[i][0],
            ),
        )
    return nbrs


def _next_half_edge(u, v, sorted_nbrs):
    """半边 (u→v) 的下一条半边：在 v 的邻居中找 u，向左转一步。"""
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face, coords):
    pts  = [coords[i] for i in face]
    n    = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_outer_face_nodes(coords_list, adj, n):
    """
    用半边算法枚举平面图所有面，找出有向面积绝对值最大的外轮廓面，
    返回其节点集合的 bool 数组 [n]。

    coords_list : list of (x, y)，长度 n
    adj         : [n, n] int 0/1 邻接矩阵（numpy 或 list of list）
    n           : 有效节点数
    """
    if n < 3:
        return np.ones(n, dtype=bool)

    # 确保 adj 是 list-of-list 形式供索引
    if hasattr(adj, 'tolist'):
        adj_ll = adj.tolist()
    else:
        adj_ll = adj

    sorted_nbrs = _build_sorted_neighbors(coords_list, adj_ll, n)

    visited   = set()
    all_faces = []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face  = []
            cu, cv = u, v
            steps  = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv))
                face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw
                steps += 1
            if len(face) >= 3:
                all_faces.append(face)

    if not all_faces:
        return np.ones(n, dtype=bool)

    abs_areas = [abs(_signed_area(f, coords_list)) for f in all_faces]
    outer_idx  = int(np.argmax(abs_areas))
    outer_set  = set(all_faces[outer_idx])
    return np.array([i in outer_set for i in range(n)], dtype=bool)


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",       default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--bert",        default="models/bert-base-uncased")
    p.add_argument("--output",      default="data/processed/node_diffusion_room_bnd/graph_dataset.npz")
    p.add_argument("--augment",     type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0,
                   help="最多读取的原始图数量，0=全量")
    return p.parse_args()


def permute_graph(adj, combo_ids, coords, is_bnd, n, perm):
    full_perm    = perm + list(range(n, MAX_NODES))
    new_adj      = adj[np.ix_(full_perm, full_perm)]
    new_ids      = combo_ids[full_perm]
    new_coords   = coords[full_perm]
    new_is_bnd   = is_bnd[full_perm]
    return new_adj, new_ids, new_coords, new_is_bnd


def _compute_room_membership(args_tuple):
    """multiprocessing worker：计算单条记录的 room_membership [MAX_NODES, MAX_ROOMS]。"""
    idx, adj_row, mask_row = args_tuple
    n    = int(mask_row.sum())
    full = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    if n >= 3:
        m = _assign_room_membership_single(adj_row[:n, :n].astype(bool), n)
        full[:n, :] = m
    return idx, full


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert)

    mask_list   = []
    ids_list    = []
    coords_list = []
    ptok_list   = []
    pmask_list  = []
    plen_list   = []
    nnodes_list = []
    adj_list    = []
    bnd_list    = []   # is_boundary per sample

    t0        = time.perf_counter()
    n_graphs  = 0
    n_skipped = 0

    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            rec    = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            if args.max_samples > 0 and n_graphs >= args.max_samples:
                break

            n        = int(rec["n_nodes"])
            n_graphs += 1

            adj_full = np.array(rec["adj_matrix"], dtype=np.int32)
            np.fill_diagonal(adj_full, 0)

            combo_ids = np.array(rec["node_combo_ids"][:MAX_NODES], dtype=np.int32)

            raw_coords = rec["node_coords"][:MAX_NODES]
            coords     = np.zeros((MAX_NODES, 2), dtype=np.float32)
            coords[:len(raw_coords)] = raw_coords

            mask     = np.zeros(MAX_NODES, dtype=np.int32)
            mask[:n] = 1

            padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
            tlen     = len(enc['input_ids'])
            padded[:tlen]   = enc['input_ids']
            attn_msk[:tlen] = enc['attention_mask']

            # 计算轮廓节点（仅对有效节点）
            coords_ll = [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
            bnd_n     = find_outer_face_nodes(coords_ll, adj_full[:n, :n], n)
            # 填充到 MAX_NODES（padding 节点视为非轮廓）
            is_bnd_full          = np.zeros(MAX_NODES, dtype=np.float32)
            is_bnd_full[:n]      = bnd_n.astype(np.float32)

            base_perm = list(range(n))
            perms     = [base_perm]
            for _ in range(args.augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_ids, new_coords, new_is_bnd = permute_graph(
                    adj_full, combo_ids, coords, is_bnd_full, n, perm)
                mask_list.append(mask)
                ids_list.append(new_ids)
                coords_list.append(new_coords)
                ptok_list.append(padded)
                pmask_list.append(attn_msk)
                plen_list.append(tlen)
                nnodes_list.append(n)
                adj_list.append(new_adj)
                bnd_list.append(new_is_bnd)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 张图 → {len(adj_list)} 条记录  ({elapsed:.1f}s)")

    total = len(adj_list)
    print(f"\n共 {n_graphs} 张图（跳过 {n_skipped} 条），增强后 {total} 条记录")

    # ── 并行计算 room_membership ──────────────────────────────────────────────
    print("计算 room_membership（并行）...")
    adj_arr  = np.stack(adj_list,  axis=0)   # [total, MAX_NODES, MAX_NODES]
    mask_arr = np.stack(mask_list, axis=0)   # [total, MAX_NODES]
    bnd_arr  = np.stack(bnd_list,  axis=0)   # [total, MAX_NODES]
    del adj_list, mask_list, bnd_list

    n_workers    = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    tasks        = [(i, adj_arr[i], mask_arr[i]) for i in range(total)]
    membership_out = np.zeros((total, MAX_NODES, MAX_ROOMS), dtype=np.float32)

    with Pool(processes=n_workers) as pool:
        for done, (idx, m) in enumerate(
            pool.imap_unordered(_compute_room_membership, tasks, chunksize=256)
        ):
            membership_out[idx] = m
            if (done + 1) % 50000 == 0:
                print(f"  room_membership: {done+1}/{total}", flush=True)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        node_mask        = mask_arr,
        node_combo_ids   = np.stack(ids_list,    axis=0),
        node_coords      = np.stack(coords_list, axis=0),
        prompt_tokens    = np.stack(ptok_list,   axis=0),
        prompt_mask      = np.stack(pmask_list,  axis=0),
        prompt_lens      = np.array(plen_list,   dtype=np.int32),
        n_nodes          = np.array(nnodes_list, dtype=np.int32),
        room_membership  = membership_out,
        adj_matrix       = adj_arr.astype(np.uint8),
        is_boundary      = bnd_arr.astype(np.float32),   # [total, MAX_NODES]
    )

    elapsed = time.perf_counter() - t0
    print(f"保存 → {out_path}  (耗时 {elapsed:.1f}s)")
    density = membership_out.sum(axis=2)
    print(f"room_membership: 平均每节点属于 {density[mask_arr==1].mean():.2f} 个环")
    bnd_ratio = bnd_arr[mask_arr==1].mean()
    print(f"is_boundary    : 轮廓节点占有效节点比例 = {bnd_ratio:.2%}")


if __name__ == "__main__":
    main()
