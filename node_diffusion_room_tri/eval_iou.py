"""
node_diffusion_room_tri IoU 评估脚本

GT 和预测坐标均质心归零后，用 Shapely 多边形面积计算 micro/macro IoU。
背景区域不参与计算，与 ChatHouseDiffusion 的评估逻辑等价。

用法：
  python -m node_diffusion_room_tri.eval_iou \
      --ckpt  checkpoints/node_diffusion_room_tri/run1/latest.pt \
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


# ── 平面图拓扑（复用 find_faces / vote_room_type）─────────────────────────────

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
    pts = [coords[i] for i in face]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords, adj):
    """coords: list of (x,y), adj: list-of-lists 0/1"""
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
    """
    coords_np : np.ndarray [n, 2]（已质心归零）
    adj_list  : list of lists [n][n]
    返回 dict: room_type -> List[Polygon]
    """
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
def ddpm_sample(model, diffusion, cond_batched, device, timesteps):
    """
    cond_batched: dict, 所有张量已有 batch 维 (B=1)
    返回 pred_coords [2, MAX_NODES]
    """
    diffusion._to(device)
    x = torch.randn(1, 2, MAX_NODES, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], device=device)
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

    return x[0]  # [2, MAX_NODES]


# ── 质心归零 ──────────────────────────────────────────────────────────────────

def center_at_origin(coords_np, mask_np):
    """coords_np [MAX_NODES, 2], mask_np [MAX_NODES] → centered copy"""
    valid = coords_np[mask_np.astype(bool)]
    centroid = valid.mean(axis=0)
    return coords_np - centroid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/node_diffusion_room_tri/run1/latest.pt")
    p.add_argument("--jsonl",      default="data/jsonl/test_graph_dataset_10k.jsonl")
    p.add_argument("--n_samples",  type=int, default=200)
    p.add_argument("--timesteps",  type=int, default=200,
                   help="推理步数，可小于训练步数(1000)以加速，越小越快但精度略降")
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

    # ── 逐条评估 ────────────────────────────────────────────────────────────
    micro_list, macro_list = [], []
    skipped = 0
    t0 = time.time()

    with open(args.jsonl, encoding="utf-8") as f:
        for sample_idx, line in enumerate(f):
            if len(micro_list) + skipped >= args.n_samples:
                break
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            n   = int(rec["n_nodes"])
            if n < 3:
                skipped += 1
                continue

            # ── GT ──────────────────────────────────────────────────────────
            raw_coords = np.array(rec["node_coords"][:n], dtype=np.float32)  # [n,2]
            adj_raw    = np.array(rec["adj_matrix"],       dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            node_types = []
            for k in range(n):
                t = rec["node_types"][k]
                node_types.append(t if isinstance(t, list) else [t])

            gt_centered = center_at_origin(raw_coords, np.ones(n))
            adj_list    = adj_raw.tolist()
            gt_polys    = coords_to_polys_by_type(gt_centered, adj_list, node_types, n)
            if not gt_polys:
                skipped += 1
                continue

            # ── 构造模型输入 ─────────────────────────────────────────────────
            mask_np = np.zeros(MAX_NODES, dtype=np.float32); mask_np[:n] = 1.0

            coords_pad = np.zeros((MAX_NODES, 2), dtype=np.float32)
            coords_pad[:n] = raw_coords

            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
            adj_pad[:n, :n] = adj_raw.astype(np.float32)

            membership = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
            m = _assign_room_membership_single(adj_raw.astype(bool), n)
            membership[:n] = m

            prompt = rec.get("prompt", "").replace("\n", " ").strip()
            enc  = tokenizer(prompt, add_special_tokens=True,
                             max_length=MAX_TEXT_LEN, padding="max_length",
                             truncation=True)
            ptok = np.array(enc["input_ids"],      dtype=np.int64)
            pmsk = np.array(enc["attention_mask"], dtype=np.float32)

            cond = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in {
                "node_mask":       mask_np,
                "room_membership": membership,
                "adj_matrix":      adj_pad,
                "prompt_tokens":   ptok,
                "prompt_mask":     pmsk,
            }.items()}

            # ── 推理 ─────────────────────────────────────────────────────────
            pred_xy = ddpm_sample(model, diffusion, cond, device, args.timesteps)
            pred_np = pred_xy.cpu().numpy().T  # [MAX_NODES, 2]

            pred_centered = center_at_origin(pred_np, mask_np)
            pred_polys    = coords_to_polys_by_type(
                pred_centered[:n], adj_list, node_types, n)

            micro, macro = compute_iou(gt_polys, pred_polys)
            micro_list.append(micro)
            macro_list.append(macro)

            done = len(micro_list)
            if done % 20 == 0:
                elapsed = time.time() - t0
                print(f"[{done}/{args.n_samples}]  "
                      f"micro={np.mean(micro_list):.4f}  "
                      f"macro={np.mean(macro_list):.4f}  "
                      f"elapsed={elapsed:.1f}s")

    print(f"\n=== 评估完成 ({len(micro_list)} 条, skipped={skipped}) ===")
    print(f"Micro-IoU : {np.mean(micro_list):.6f}")
    print(f"Macro-IoU : {np.mean(macro_list):.6f}")

    if args.out:
        import json as _json
        Path(args.out).write_text(_json.dumps({
            "n": len(micro_list),
            "micro_iou": float(np.mean(micro_list)),
            "macro_iou": float(np.mean(macro_list)),
            "micro_list": micro_list,
            "macro_list": macro_list,
        }, indent=2))
        print(f"结果保存 → {args.out}")


if __name__ == "__main__":
    main()
