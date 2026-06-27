"""
node_diffusion_room_bnd IoU 评估脚本

GT 和预测坐标均质心归零后，用 Shapely 多边形面积计算 micro/macro IoU。
推理时从 GT 坐标和邻接矩阵实时计算 is_boundary 作为轮廓条件。

用法：
  python -m node_diffusion_room_bnd.eval_iou \
      --ckpt  checkpoints/node_diffusion_room_bnd/run1/latest.pt \
      --jsonl data/jsonl/test_graph_dataset_10k.jsonl \
      --n_samples 200 \
      --timesteps 200
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer, _assign_room_membership_single, MAX_ROOMS
from .diffusion import GaussianDiffusion

MAX_NODES    = 40
MAX_TEXT_LEN = 192

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]


# ── 半边算法（与 build_graph_npz.py 保持一致）────────────────────────────────

def _build_sorted_neighbors(coords, adj, n):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
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
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _find_outer_face_nodes(coords_list, adj_list, n):
    """返回外轮廓节点的 bool 数组 [n]。"""
    if n < 3:
        return np.ones(n, dtype=bool)
    sorted_nbrs = _build_sorted_neighbors(coords_list, adj_list, n)
    visited = set(); all_faces = []
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
                all_faces.append(face)
    if not all_faces:
        return np.ones(n, dtype=bool)
    abs_areas = [abs(_signed_area(f, coords_list)) for f in all_faces]
    outer_idx = int(np.argmax(abs_areas))
    outer_set = set(all_faces[outer_idx])
    return np.array([i in outer_set for i in range(n)], dtype=bool)


# ── 平面图拓扑（find_faces / vote_room_type）─────────────────────────────────

def find_faces(coords, adj):
    """coords: list of (x,y), adj: list-of-lists 0/1  → 返回内部面列表（不含外轮廓面）"""
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


# ── 多边形 IoU ────────────────────────────────────────────────────────────────

def coords_to_polys_by_type(coords_np, adj_list, node_types, n):
    coords = [(float(coords_np[i, 0]), float(coords_np[i, 1])) for i in range(n)]
    faces = find_faces(coords, adj_list)
    if not faces:
        return {}
    all_nbrs = _build_sorted_neighbors(coords, adj_list, n)
    polys_by_type = {}
    for face in faces:
        rtype = vote_room_type(face, node_types, all_nbrs)
        pts = [coords[i] for i in face]
        try:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 1e-6:
                polys_by_type.setdefault(rtype, []).append(poly)
        except Exception:
            pass
    return polys_by_type


def compute_iou(gt_by_type, pred_by_type):
    all_types = set(gt_by_type) | set(pred_by_type)
    intersections, unions, ious = [], [], []
    for rtype in all_types:
        gt_u   = unary_union(gt_by_type.get(rtype,   []) or [Polygon()])
        pred_u = unary_union(pred_by_type.get(rtype, []) or [Polygon()])
        inter  = gt_u.intersection(pred_u).area
        union  = gt_u.union(pred_u).area
        if union > 1e-9:
            intersections.append(inter)
            unions.append(union)
            ious.append(inter / union)
    if not unions:
        return 0.0, 0.0
    return sum(intersections) / sum(unions), sum(ious) / len(ious)


# ── DDPM 反向采样 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, diffusion, cond_batched, gt_coords_np, device, timesteps):
    """
    DDPM 批次采样。

    cond_batched : dict，所有张量形状 [B, ...]
    gt_coords_np : [B, MAX_NODES, 2] 或 [MAX_NODES, 2] float32
    返回         : [B, 2, MAX_NODES] float32 tensor
    """
    diffusion._to(device)

    is_bnd = cond_batched.get('is_boundary', None)   # [B, MAX_NODES]
    B = next(iter(cond_batched.values())).shape[0]
    x = torch.randn(B, 2, MAX_NODES, device=device)

    gt_xy = bnd_mask = None
    if is_bnd is not None:
        gt_np = np.asarray(gt_coords_np, dtype=np.float32)
        if gt_np.ndim == 2:
            gt_np = gt_np[None]                                            # [1,N,2]
        gt_xy    = torch.from_numpy(gt_np.transpose(0, 2, 1)).to(device)  # [B,2,N]
        bnd_mask = is_bnd.unsqueeze(1).bool()                              # [B,1,N]
        x = torch.where(bnd_mask, gt_xy, x)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(x, t_tensor, **cond_batched)

        s1  = diffusion.sqrt_alphas_bar[t]
        s2  = diffusion.sqrt_one_minus_alphas_bar[t]
        x0  = (x - s2 * eps) / s1.clamp(min=1e-3)

        if t == 0:
            x = x0
        else:
            alpha          = diffusion.alphas[t]
            alpha_bar      = diffusion.alphas_bar[t]
            alpha_bar_prev = diffusion.alphas_bar_prev[t]
            beta           = diffusion.betas[t]
            coeff1 = beta * alpha_bar_prev.sqrt() / (1 - alpha_bar)
            coeff2 = (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar)
            mean   = coeff1 * x0 + coeff2 * x
            var    = diffusion.posterior_variance[t]
            x      = mean + var.sqrt() * torch.randn_like(x)

        if bnd_mask is not None:
            x = torch.where(bnd_mask, gt_xy, x)

    return x   # [B, 2, MAX_NODES]


# ── 质心归零 ──────────────────────────────────────────────────────────────────

def center_at_origin(coords_np, mask_np):
    valid = coords_np[mask_np.astype(bool)]
    centroid = valid.mean(axis=0)
    return coords_np - centroid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/node_diffusion_room_bnd/run1/latest.pt")
    p.add_argument("--jsonl",      default="data/jsonl/test_graph_dataset_10k.jsonl")
    p.add_argument("--n_samples",  type=int, default=200)
    p.add_argument("--timesteps",  type=int, default=200,
                   help="推理步数，可小于训练步数(1000)以加速")
    p.add_argument("--batch_size", type=int, default=16,
                   help="推理批次大小")
    p.add_argument("--bert",       default="models/bert-base-uncased")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    p.add_argument("--out",        default="", help="可选：结果保存路径(.json)")
    args = p.parse_args()

    device    = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    # ── 加载模型 ────────────────────────────────────────────────────────────
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    ).to(device)
    ckpt   = torch.load(args.ckpt, map_location=device)
    raw_sd = ckpt["model"]
    if any(k.startswith("module.") for k in raw_sd):
        raw_sd = {k[7:]: v for k, v in raw_sd.items()}
    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    model.eval()
    print(f"Loaded: {args.ckpt}  step={ckpt.get('step', '?')}")

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── 第一步：预处理所有样本（CPU）────────────────────────────────────────
    prepared = []
    skipped  = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if len(prepared) + skipped >= args.n_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n   = int(rec["n_nodes"])
            if n < 3:
                skipped += 1
                continue

            raw_coords = np.array(rec["node_coords"][:n], dtype=np.float32)
            adj_raw    = np.array(rec["adj_matrix"],       dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            node_types = [
                (t if isinstance(t, list) else [t])
                for t in rec["node_types"][:n]
            ]

            gt_centered = center_at_origin(raw_coords, np.ones(n))
            adj_list    = adj_raw.tolist()
            gt_polys    = coords_to_polys_by_type(gt_centered, adj_list, node_types, n)
            if not gt_polys:
                skipped += 1
                continue

            mask_np    = np.zeros(MAX_NODES, dtype=np.float32); mask_np[:n] = 1.0
            coords_pad = np.zeros((MAX_NODES, 2), dtype=np.float32); coords_pad[:n] = raw_coords
            adj_pad    = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
            adj_pad[:n, :n] = adj_raw.astype(np.float32)

            membership = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
            membership[:n] = _assign_room_membership_single(adj_raw.astype(bool), n)

            coords_ll = [(float(raw_coords[i, 0]), float(raw_coords[i, 1])) for i in range(n)]
            is_bnd_np = np.zeros(MAX_NODES, dtype=np.float32)
            is_bnd_np[:n] = _find_outer_face_nodes(coords_ll, adj_list, n).astype(np.float32)

            prompt = rec.get("prompt", "").replace("\n", " ").strip()
            enc  = tokenizer(prompt, add_special_tokens=True,
                             max_length=MAX_TEXT_LEN, padding="max_length", truncation=True)
            ptok = np.array(enc["input_ids"],      dtype=np.int64)
            pmsk = np.array(enc["attention_mask"], dtype=np.float32)

            prepared.append({
                "mask_np":    mask_np,
                "coords_pad": coords_pad,
                "adj_pad":    adj_pad,
                "membership": membership,
                "is_bnd_np":  is_bnd_np,
                "ptok":       ptok,
                "pmsk":       pmsk,
                "n":          n,
                "adj_list":   adj_list,
                "node_types": node_types,
                "gt_polys":   gt_polys,
            })

    print(f"预处理完成：{len(prepared)} 条有效（跳过 {skipped} 条）")

    # ── 第二步：批次 DDPM 推理（GPU）────────────────────────────────────────
    all_pred_np = []
    t0  = time.time()
    BS  = args.batch_size
    print(f"批次推理 batch={BS}，DDPM {args.timesteps} 步...", flush=True)

    for bi in range(0, len(prepared), BS):
        chunk = prepared[bi: bi + BS]
        B = len(chunk)

        cond_b = {
            "node_mask":       torch.from_numpy(np.stack([s["mask_np"]    for s in chunk])).to(device),
            "room_membership": torch.from_numpy(np.stack([s["membership"] for s in chunk])).to(device),
            "adj_matrix":      torch.from_numpy(np.stack([s["adj_pad"]    for s in chunk])).to(device),
            "prompt_tokens":   torch.from_numpy(np.stack([s["ptok"]       for s in chunk])).to(device),
            "prompt_mask":     torch.from_numpy(np.stack([s["pmsk"]       for s in chunk])).to(device),
            "is_boundary":     torch.from_numpy(np.stack([s["is_bnd_np"]  for s in chunk])).to(device),
        }
        gt_coords_b = np.stack([s["coords_pad"] for s in chunk])   # [B, MAX_NODES, 2]

        pred_xy = ddpm_sample(model, diffusion, cond_b, gt_coords_b, device, args.timesteps)
        for j in range(B):
            all_pred_np.append(pred_xy[j].cpu().numpy().T)   # [MAX_NODES, 2]

        done = min(bi + BS, len(prepared))
        elapsed = time.time() - t0
        print(f"  [{done}/{len(prepared)}]  elapsed={elapsed:.1f}s", flush=True)

    # ── 第三步：逐样本计算 IoU（CPU）────────────────────────────────────────
    micro_list, macro_list = [], []
    for s, pred_np in zip(prepared, all_pred_np):
        pred_cen   = center_at_origin(pred_np, s["mask_np"])
        pred_polys = coords_to_polys_by_type(pred_cen[:s["n"]], s["adj_list"], s["node_types"], s["n"])
        micro, macro = compute_iou(s["gt_polys"], pred_polys)
        micro_list.append(micro)
        macro_list.append(macro)

    print(f"\n=== 评估完成 ({len(micro_list)} 条, skipped={skipped}) ===")
    print(f"Micro-IoU : {np.mean(micro_list):.6f}")
    print(f"Macro-IoU : {np.mean(macro_list):.6f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n": len(micro_list),
            "micro_iou": float(np.mean(micro_list)),
            "macro_iou": float(np.mean(macro_list)),
            "micro_list": micro_list,
            "macro_list": macro_list,
        }, indent=2))
        print(f"结果保存 → {args.out}")


if __name__ == "__main__":
    main()
