"""
从推理 NPZ 计算 Micro/Macro IoU。

输入 NPZ 由 eval_infer.py 生成，包含 GT 和预测的节点坐标、类型、邻接矩阵。

质心对齐：
    GT 坐标在数据集构建时已以原点为中心。
    预测坐标整体平移，使有效节点质心归零，两者进入同一坐标空间。

IoU 计算：
    对每个样本，用 find_faces 从节点坐标 + 邻接图提取房间多边形，
    按房间类型分组后用 shapely 计算面积交集/并集，跨所有样本累加。

    Micro-IoU = sum_r(I_r) / sum_r(U_r)
    Macro-IoU = (1/R) * sum_r(I_r / U_r)

Usage (from project root):
    python -m node_diffusion_cross_att.eval_iou \\
        --npz   outputs/eval/infer_500.npz \\
        --vocab node_diffusion_cross_att/type_combo_vocab_old.json \\
        --out   outputs/eval/iou_500.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from .render import load_vocab, find_faces, vote_room_type, ROOM_TYPE_ORDER


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',   default='', help='NPZ 文件路径（与 --jsonl 二选一）')
    p.add_argument('--jsonl', default='', help='best_roll.jsonl 路径（与 --npz 二选一）')
    p.add_argument('--vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--out',   default='outputs/eval/iou_result.json')
    return p.parse_args()


def extract_polygons(
    coords: np.ndarray,          # [n, 2]  已截断到 n 个有效节点
    adj:    np.ndarray,          # [n, n]
    node_types: List[List[str]], # length n
) -> Dict[str, List[ShapelyPolygon]]:
    """
    从有效节点坐标 + 邻接矩阵提取房间多边形，按房间类型分组。
    """
    n = len(coords)
    coords_list = [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
    adj_list    = adj.tolist()

    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj_list[i][j] == 1:
                all_nbrs[i].append(j)

    faces = find_faces(coords_list, adj_list)

    polys: Dict[str, List[ShapelyPolygon]] = defaultdict(list)
    for face in faces:
        if len(face) < 3:
            continue
        room_type = vote_room_type(face, node_types, all_nbrs)
        pts = [(coords_list[i][0], coords_list[i][1]) for i in face]
        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 0:
                polys[room_type].append(poly)
        except Exception:
            continue

    return polys


def load_samples(args, id_to_combo):
    """统一加载样本，返回 list of dict，每个 dict 包含 n, adj, gt_c, gt_types, pred_c, pred_types。"""
    samples = []

    if args.jsonl:
        print(f'读取 JSONL: {args.jsonl}')
        with open(args.jsonl, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n   = int(row['n_nodes'])
                adj = np.array(row['adj_matrix'], dtype=np.float32)   # [n, n]

                gt_c = np.array(row['gt_coords'], dtype=np.float32)   # [n, 2]
                gt_types = [id_to_combo.get(int(row['gt_combo_ids'][k]), ['other'])
                            for k in range(n)]

                pred_c = np.array(row['pred_coords'], dtype=np.float32)  # [n, 2]
                pred_c -= pred_c.mean(axis=0)  # 质心归零
                pred_types = [id_to_combo.get(int(row['pred_combo_ids'][k]), ['other'])
                              for k in range(n)]

                samples.append(dict(n=n, adj=adj,
                                    gt_c=gt_c, gt_types=gt_types,
                                    pred_c=pred_c, pred_types=pred_types))
    else:
        print(f'读取 NPZ: {args.npz}')
        data = np.load(args.npz)
        gt_coords      = data['gt_coords']
        gt_combo_ids   = data['gt_combo_ids']
        gt_adj         = data['gt_adj']
        gt_n_nodes     = data['gt_n_nodes']
        pred_coords    = data['pred_coords']
        pred_combo_ids = data['pred_combo_ids']
        K = int(data['rolls']) if 'rolls' in data else 1

        for i in range(len(gt_n_nodes)):
            n   = int(gt_n_nodes[i])
            adj = gt_adj[i, :n, :n]

            gt_c     = gt_coords[i, :n]
            gt_types = [id_to_combo.get(int(gt_combo_ids[i, k]), ['other'])
                        for k in range(n)]

            # NPZ 模式：取第 0 个 roll
            pc = pred_coords[i] if pred_coords.ndim == 3 else pred_coords[i, 0]
            pred_c = pc[:n].copy()
            pred_c -= pred_c.mean(axis=0)
            pred_types = [id_to_combo.get(int(pred_combo_ids[i, 0, k] if pred_combo_ids.ndim == 3 else pred_combo_ids[i, k]), ['other'])
                          for k in range(n)]

            samples.append(dict(n=n, adj=adj,
                                gt_c=gt_c, gt_types=gt_types,
                                pred_c=pred_c, pred_types=pred_types))

    print(f'共 {len(samples)} 条样本')
    return samples


def main():
    args = parse_args()
    if not args.jsonl and not args.npz:
        raise SystemExit('请指定 --jsonl 或 --npz')
    id_to_combo = load_vocab(Path(args.vocab))

    samples = load_samples(args, id_to_combo)
    N = len(samples)

    sum_inter: Dict[str, float] = defaultdict(float)
    sum_union: Dict[str, float] = defaultdict(float)
    skipped = 0

    for i, s in enumerate(samples):
        n          = s['n']
        adj        = s['adj']
        gt_c       = s['gt_c']
        gt_types   = s['gt_types']
        pred_c     = s['pred_c']
        pred_types = s['pred_types']

        try:
            gt_polys   = extract_polygons(gt_c,   adj, gt_types)
            pred_polys = extract_polygons(pred_c, adj, pred_types)
        except Exception:
            skipped += 1
            continue

        # 按类型累加交集/并集面积
        all_types = set(gt_polys) | set(pred_polys)
        for rt in all_types:
            gt_merged   = unary_union(gt_polys[rt])   if gt_polys.get(rt)   else None
            pred_merged = unary_union(pred_polys[rt]) if pred_polys.get(rt) else None

            if gt_merged is None and pred_merged is None:
                continue
            elif gt_merged is None:
                sum_union[rt] += pred_merged.area
            elif pred_merged is None:
                sum_union[rt] += gt_merged.area
            else:
                sum_inter[rt] += gt_merged.intersection(pred_merged).area
                sum_union[rt] += gt_merged.union(pred_merged).area

        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{N}', flush=True)

    print(f'Skipped {skipped}/{N} samples (polygon errors)')

    # ── Micro / Macro IoU ─────────────────────────────────────────────────────
    per_type_iou: Dict[str, float] = {}
    for rt in ROOM_TYPE_ORDER:
        u = sum_union.get(rt, 0.0)
        if u > 0:
            per_type_iou[rt] = sum_inter.get(rt, 0.0) / u

    micro_iou = sum(sum_inter.values()) / max(sum(sum_union.values()), 1e-9)
    macro_iou = sum(per_type_iou.values()) / max(len(per_type_iou), 1)

    print('\n=== IoU Results ===')
    print(f'Micro-IoU : {micro_iou * 100:.2f}%')
    print(f'Macro-IoU : {macro_iou * 100:.2f}%')
    print('\nPer-type IoU:')
    for rt in ROOM_TYPE_ORDER:
        if rt in per_type_iou:
            print(f'  {rt:12s}: {per_type_iou[rt]*100:.2f}%'
                  f'  (I={sum_inter.get(rt,0):.1f}, U={sum_union.get(rt,0):.1f})')

    src = args.jsonl if args.jsonl else args.npz
    result = {
        'micro_iou':    micro_iou,
        'macro_iou':    macro_iou,
        'per_type_iou': per_type_iou,
        'n_samples':    N,
        'skipped':      skipped,
        'source':       src,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
