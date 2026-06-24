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

        # 返回所有 roll 的样本列表，roll_k=-1 表示单 roll（ndim==3）
        roll_k = args.roll if hasattr(args, 'roll') else 0
        for i in range(len(gt_n_nodes)):
            n   = int(gt_n_nodes[i])
            adj = gt_adj[i, :n, :n]

            gt_c     = gt_coords[i, :n]
            gt_types = [id_to_combo.get(int(gt_combo_ids[i, j]), ['other'])
                        for j in range(n)]

            if pred_coords.ndim == 3:
                pc = pred_coords[i]
            else:
                pc = pred_coords[i, roll_k]
            pred_c = pc[:n].copy()
            pred_c -= pred_c.mean(axis=0)

            if pred_combo_ids.ndim == 2:
                pred_types = [id_to_combo.get(int(pred_combo_ids[i, j]), ['other'])
                              for j in range(n)]
            else:
                pred_types = [id_to_combo.get(int(pred_combo_ids[i, roll_k, j]), ['other'])
                              for j in range(n)]

            samples.append(dict(n=n, adj=adj,
                                gt_c=gt_c, gt_types=gt_types,
                                pred_c=pred_c, pred_types=pred_types))

    print(f'共 {len(samples)} 条样本')
    return samples


def compute_iou(samples) -> dict:
    """对给定样本列表计算 Micro/Macro IoU，返回结果 dict。"""
    N = len(samples)
    sum_inter: Dict[str, float] = defaultdict(float)
    sum_union: Dict[str, float] = defaultdict(float)
    skipped = 0

    for i, s in enumerate(samples):
        try:
            gt_polys   = extract_polygons(s['gt_c'],   s['adj'], s['gt_types'])
            pred_polys = extract_polygons(s['pred_c'], s['adj'], s['pred_types'])
        except Exception:
            skipped += 1
            continue

        for rt in set(gt_polys) | set(pred_polys):
            gt_m   = unary_union(gt_polys[rt])   if gt_polys.get(rt)   else None
            pred_m = unary_union(pred_polys[rt]) if pred_polys.get(rt) else None
            if gt_m is None and pred_m is None:
                continue
            elif gt_m is None:
                sum_union[rt] += pred_m.area
            elif pred_m is None:
                sum_union[rt] += gt_m.area
            else:
                sum_inter[rt] += gt_m.intersection(pred_m).area
                sum_union[rt] += gt_m.union(pred_m).area

        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{N}', flush=True)

    per_type_iou: Dict[str, float] = {}
    for rt in ROOM_TYPE_ORDER:
        u = sum_union.get(rt, 0.0)
        if u > 0:
            per_type_iou[rt] = sum_inter.get(rt, 0.0) / u

    micro = sum(sum_inter.values()) / max(sum(sum_union.values()), 1e-9)
    macro = sum(per_type_iou.values()) / max(len(per_type_iou), 1)
    return dict(micro_iou=micro, macro_iou=macro,
                per_type_iou=per_type_iou, skipped=skipped, n_samples=N)


def main():
    args = parse_args()
    if not args.jsonl and not args.npz:
        raise SystemExit('请指定 --jsonl 或 --npz')
    id_to_combo = load_vocab(Path(args.vocab))

    # ── JSONL 模式：单次评估 ──────────────────────────────────────────────────
    if args.jsonl:
        samples = load_samples(args, id_to_combo)
        res = compute_iou(samples)
        print(f'Skipped {res["skipped"]}/{res["n_samples"]} samples')
        print(f'\n=== IoU Results ===')
        print(f'Micro-IoU : {res["micro_iou"]*100:.2f}%')
        print(f'Macro-IoU : {res["macro_iou"]*100:.2f}%')
        result = dict(**res, source=args.jsonl)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'\nSaved → {args.out}')
        return

    # ── NPZ 模式：对每个 roll 独立评估，报 best / avg / worst ─────────────────
    print(f'读取 NPZ: {args.npz}')
    data = np.load(args.npz)
    K = int(data['rolls']) if 'rolls' in data else 1
    print(f'K={K} rolls，逐 roll 评估中...')

    roll_results = []
    for k in range(K):
        print(f'\n── Roll {k} ──────────────────────────────')
        args.roll = k
        samples = load_samples(args, id_to_combo)
        res = compute_iou(samples)
        res['roll'] = k
        roll_results.append(res)
        print(f'  Micro-IoU: {res["micro_iou"]*100:.2f}%  '
              f'Macro-IoU: {res["macro_iou"]*100:.2f}%  '
              f'skipped: {res["skipped"]}')

    micros = [r['micro_iou'] for r in roll_results]
    macros = [r['macro_iou'] for r in roll_results]

    best_k  = int(np.argmax(micros))
    worst_k = int(np.argmin(micros))

    print(f'\n{"="*45}')
    print(f'{"":12s}  {"Micro-IoU":>10}  {"Macro-IoU":>10}')
    print(f'{"─"*45}')
    print(f'{"Best (roll "+str(best_k)+")":12s}  '
          f'{micros[best_k]*100:>9.2f}%  {macros[best_k]*100:>9.2f}%')
    print(f'{"Average":12s}  '
          f'{np.mean(micros)*100:>9.2f}%  {np.mean(macros)*100:>9.2f}%')
    print(f'{"Worst (roll "+str(worst_k)+")":12s}  '
          f'{micros[worst_k]*100:>9.2f}%  {macros[worst_k]*100:>9.2f}%')
    print(f'{"="*45}')

    result = {
        'best':  {'roll': best_k,  'micro_iou': micros[best_k],  'macro_iou': macros[best_k]},
        'avg':   {'micro_iou': float(np.mean(micros)), 'macro_iou': float(np.mean(macros))},
        'worst': {'roll': worst_k, 'micro_iou': micros[worst_k], 'macro_iou': macros[worst_k]},
        'per_roll': roll_results,
        'source': args.npz,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
