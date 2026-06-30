"""
node_diffusion_room_tri 推理脚本

从 jsonl 读取图结构条件，批次采样预测节点坐标，结果写回 jsonl。

输出每行字段：
  prompt          原始文本
  n_nodes         有效节点数
  adj_matrix      邻接矩阵（n×n）
  gt_node_coords  GT 坐标 [[x,y], ...]（n 条）
  gt_node_types   GT 节点类型
  pred_node_coords  预测坐标（质心归零后，n 条）

用法：
  python -m node_diffusion_room_tri.infer \\
      --ckpt      checkpoints/node_diffusion_room_tri/latest.pt \\
      --jsonl     data/jsonl/test_graph_dataset_18k5.jsonl \\
      --n_samples 1000 \\
      --sampler   ddim \\
      --ddim_steps 500 \\
      --out       outputs/tri_infer.jsonl
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .model import NodeDiffusionTransformer, _assign_room_membership_single, MAX_ROOMS
from .diffusion import GaussianDiffusion
from transformers import BertTokenizer

MAX_NODES    = 40
MAX_TEXT_LEN = 192


# ── DDIM ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, diffusion, cond_batched, device, ddim_steps=200):
    diffusion._to(device)
    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()
    B  = next(iter(cond_batched.values())).shape[0]
    x  = torch.randn(B, 2, MAX_NODES, device=device)

    # 预计算一次 BERT，避免每个 diffusion step 重复跑
    text_feat, text_mask = model.encode_text(
        cond_batched["prompt_tokens"], cond_batched.get("prompt_mask"))
    cond_no_text = {k: v for k, v in cond_batched.items()
                    if k not in ("prompt_tokens", "prompt_mask")}

    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps  = model(x, t_tensor, text_feat=text_feat, text_mask=text_mask, **cond_no_text)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)
        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0
    return x  # [B, 2, MAX_NODES]


# ── DDPM ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, diffusion, cond_batched, device, timesteps=1000):
    diffusion._to(device)
    B = next(iter(cond_batched.values())).shape[0]
    x = torch.randn(B, 2, MAX_NODES, device=device)

    text_feat, text_mask = model.encode_text(
        cond_batched["prompt_tokens"], cond_batched.get("prompt_mask"))
    cond_no_text = {k: v for k, v in cond_batched.items()
                    if k not in ("prompt_tokens", "prompt_mask")}

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(x, t_tensor, text_feat=text_feat, text_mask=text_mask, **cond_no_text)
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
            x = coeff1 * x0 + coeff2 * x + diffusion.posterior_variance[t].sqrt() * torch.randn_like(x)
    return x  # [B, 2, MAX_NODES]


# ── 质心归零 ─────────────────────────────────────────────────────────────────

def center_at_origin(coords_np, mask_np):
    valid = coords_np[mask_np.astype(bool)]
    return coords_np - valid.mean(axis=0)


# ── T 形接头吸附 ──────────────────────────────────────────────────────────────

def snap_nodes(coords, adj, n, threshold_ratio=0.02):
    """
    对 pred_node_coords 做 T 形吸附：
      - 阈值 = bounding box 斜对角线 × threshold_ratio
      - 若节点 i 到边 (u,v) 的距离 < 阈值，将 i 投影到 (u,v)，
        删除 u-v 边，新增 i-u、i-v 边
      - 每个节点只吸附到最近的一条边
    返回 (new_coords, new_adj)，均为 list
    """
    coords = [[c[0], c[1]] for c in coords]
    adj    = [list(row) for row in adj]

    xs = [coords[i][0] for i in range(n)]
    ys = [coords[i][1] for i in range(n)]
    diag   = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    thresh = threshold_ratio * diag

    for i in range(n):
        px, py   = coords[i]
        best_dist = thresh
        best_edge = None
        best_proj = None

        for u in range(n):
            for v in range(u + 1, n):
                if u == i or v == i:
                    continue
                if not adj[u][v]:
                    continue
                ax, ay   = coords[u]
                bx, by   = coords[v]
                abx, aby = bx - ax, by - ay
                ab2      = abx * abx + aby * aby
                if ab2 < 1e-12:
                    continue
                t   = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
                nx_ = ax + t * abx
                ny_ = ay + t * aby
                dist = math.sqrt((px - nx_) ** 2 + (py - ny_) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_edge = (u, v)
                    best_proj = [nx_, ny_]

        if best_edge is not None:
            u, v = best_edge
            coords[i] = best_proj
            adj[u][v] = adj[v][u] = 0
            adj[i][u] = adj[u][i] = 1
            adj[i][v] = adj[v][i] = 1

    return coords, adj


# ── 边交叉点插入节点 ──────────────────────────────────────────────────────────

def _seg_intersect(p1, p2, p3, p4):
    """返回线段 (p1,p2) 与 (p3,p4) 的严格内部交点，无则返回 None。"""
    x1, y1 = p1;  x2, y2 = p2
    x3, y3 = p3;  x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 1e-8 < t < 1 - 1e-8 and 1e-8 < u < 1 - 1e-8:
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    return None


def insert_crossing_nodes(coords, adj, n):
    """
    找所有边交叉点，在交叉处插入新节点，断开两条原边，新增四条边。
    迭代直到无新交叉为止。返回 (coords, adj, n)。
    """
    changed = True
    while changed:
        changed = False
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
        for ei in range(len(edges)):
            a, b = edges[ei]
            for ej in range(ei + 1, len(edges)):
                c, d = edges[ej]
                if a == c or a == d or b == c or b == d:
                    continue  # 共享端点，不算交叉
                pt = _seg_intersect(coords[a], coords[b], coords[c], coords[d])
                if pt is None:
                    continue
                # 插入新节点
                k = n
                coords = coords + [pt]
                new_adj = [[0] * (n + 1) for _ in range(n + 1)]
                for i in range(n):
                    for j in range(n):
                        new_adj[i][j] = adj[i][j]
                new_adj[a][b] = new_adj[b][a] = 0
                new_adj[c][d] = new_adj[d][c] = 0
                new_adj[a][k] = new_adj[k][a] = 1
                new_adj[b][k] = new_adj[k][b] = 1
                new_adj[c][k] = new_adj[k][c] = 1
                new_adj[d][k] = new_adj[k][d] = 1
                adj = new_adj
                n += 1
                changed = True
                break
            if changed:
                break
    return coords, adj, n


# ── 节点连接图渲染 ────────────────────────────────────────────────────────────

def render_graph(coords, adj, n, img_size=768, margin=48, node_r=18):
    """
    coords : [[x, y], ...] 长度 n，原始坐标（任意范围）
    adj    : n×n int 列表
    节点编号从 0 开始，与 jsonl 中 pred_node_coords 下标一一对应。
    返回 PIL.Image (RGB)
    """
    pts = np.array(coords, dtype=np.float32)   # [n, 2]

    # 归一化到 [margin, img_size-margin]
    lo, hi = pts.min(0), pts.max(0)
    span   = np.maximum(hi - lo, 1e-6)
    draw_range = img_size - 2 * margin
    pts_px = ((pts - lo) / span * draw_range + margin).astype(int)

    img  = Image.new("RGB", (img_size, img_size), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # 字体：先尝试系统 Bold，再尝试普通，最后 fallback
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # 边
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                draw.line([tuple(pts_px[i]), tuple(pts_px[j])],
                          fill=(150, 150, 150), width=3)

    # 节点 + 编号（0-based，与 jsonl 下标对应）
    for i in range(n):
        x, y = int(pts_px[i][0]), int(pts_px[i][1])
        draw.ellipse([x - node_r, y - node_r, x + node_r, y + node_r],
                     fill=(78, 140, 194), outline=(30, 30, 30), width=2)
        label = str(i)
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw // 2, y - th // 2), label,
                  fill=(255, 255, 255), font=font)

    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/node_diffusion_room_tri/latest.pt")
    p.add_argument("--jsonl",      default="data/jsonl/test_graph_dataset_18k5.jsonl")
    p.add_argument("--out",        default="outputs/tri_infer.jsonl")
    p.add_argument("--n_samples",  type=int, default=1000)
    p.add_argument("--sampler",    default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--ddim_steps", type=int, default=500)
    p.add_argument("--timesteps",  type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bert",       default="models/bert-base-uncased")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    p.add_argument("--img_dir",        default="outputs/tri_infer_imgs",
                   help="节点连接图保存目录")
    p.add_argument("--img_size",       type=int, default=768)
    args = p.parse_args()

    device    = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    # ── 加载模型 ──────────────────────────────────────────────────────────────
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
    model.load_state_dict(raw_sd, strict=False)
    model.eval()
    print(f"Loaded: {args.ckpt}  step={ckpt.get('step', '?')}")

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── 预处理 ────────────────────────────────────────────────────────────────
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

            adj_raw = np.array(rec["adj_matrix"], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            # GT 字段可选（θ₁ 直接输出的 JSONL 没有真实坐标和类型）
            if "node_coords" in rec:
                raw_coords = np.array(rec["node_coords"][:n], dtype=np.float32)
            else:
                raw_coords = np.zeros((n, 2), dtype=np.float32)
            if "node_types" in rec:
                gt_node_types = [
                    (t if isinstance(t, list) else [t])
                    for t in rec["node_types"][:n]
                ]
            else:
                gt_node_types = [["other"]] * n

            mask_np = np.zeros(MAX_NODES, dtype=np.float32); mask_np[:n] = 1.0
            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
            adj_pad[:n, :n] = adj_raw.astype(np.float32)
            membership = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
            membership[:n] = _assign_room_membership_single(adj_raw.astype(bool), n)

            prompt = rec.get("prompt", "").replace("\n", " ").strip()
            enc  = tokenizer(prompt, add_special_tokens=True,
                             max_length=MAX_TEXT_LEN, padding="max_length", truncation=True)
            ptok = np.array(enc["input_ids"],      dtype=np.int64)
            pmsk = np.array(enc["attention_mask"], dtype=np.float32)

            prepared.append({
                "mask_np":    mask_np,
                "adj_pad":    adj_pad,
                "membership": membership,
                "ptok":       ptok,
                "pmsk":       pmsk,
                "n":          n,
                "prompt":     prompt,
                "adj_list":   adj_raw.tolist(),
                "gt_coords":  raw_coords.tolist(),
                "gt_types":   gt_node_types,
            })

    print(f"预处理: {len(prepared)} 条有效，{skipped} 条跳过")
    sampler_info = (f"DDIM {args.ddim_steps} 步" if args.sampler == "ddim"
                    else f"DDPM {args.timesteps} 步")
    print(f"{sampler_info}，batch={args.batch_size}，开始推理...")

    # ── 批次采样 ──────────────────────────────────────────────────────────────
    all_pred = []
    VB = args.batch_size
    t0 = time.time()

    with torch.no_grad():
        for bi in range(0, len(prepared), VB):
            chunk = prepared[bi: bi + VB]
            B = len(chunk)

            cond_b = {
                "node_mask":       torch.from_numpy(np.stack([s["mask_np"]    for s in chunk])).to(device),
                "room_membership": torch.from_numpy(np.stack([s["membership"] for s in chunk])).to(device),
                "adj_matrix":      torch.from_numpy(np.stack([s["adj_pad"]    for s in chunk])).to(device),
                "prompt_tokens":   torch.from_numpy(np.stack([s["ptok"]       for s in chunk])).to(device),
                "prompt_mask":     torch.from_numpy(np.stack([s["pmsk"]       for s in chunk])).to(device),
            }

            if args.sampler == "ddim":
                pred_xy = ddim_sample(model, diffusion, cond_b, device, args.ddim_steps)
            else:
                pred_xy = ddpm_sample(model, diffusion, cond_b, device, args.timesteps)

            for j in range(B):
                pred_np = pred_xy[j].cpu().numpy().T   # [MAX_NODES, 2]
                n_j     = chunk[j]["n"]
                mask_j  = chunk[j]["mask_np"]
                pred_centered = center_at_origin(pred_np, mask_j)
                coords_j = pred_centered[:n_j].tolist()
                adj_j    = chunk[j]["adj_list"]
                coords_j, adj_j = snap_nodes(coords_j, adj_j, n_j)
                coords_j, adj_j, n_j = insert_crossing_nodes(coords_j, adj_j, n_j)
                all_pred.append((coords_j, adj_j, n_j))

            done = min(bi + VB, len(prepared))
            print(f"  [{done}/{len(prepared)}]  {time.time() - t0:.1f}s", flush=True)

    # ── 写出 jsonl + 渲染图片 ─────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.img_dir) if args.img_dir else None
    if img_dir:
        img_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (s, (pred_coords, pred_adj, pred_n)) in enumerate(zip(prepared, all_pred)):
            f.write(json.dumps({
                "prompt":           s["prompt"],
                "n_nodes":          pred_n,
                "adj_matrix":       pred_adj,
                "gt_node_coords":   s["gt_coords"],
                "gt_node_types":    s["gt_types"],
                "pred_node_coords": pred_coords,
            }, ensure_ascii=False) + "\n")

            if img_dir:
                img = render_graph(pred_coords, pred_adj, pred_n,
                                   img_size=args.img_size)
                img.save(img_dir / f"{idx:05d}.png")

    print(f"\n完成: {len(all_pred)} 条  耗时 {time.time() - t0:.1f}s")
    print(f"结果 -> {out_path}")
    if img_dir:
        print(f"图片 -> {img_dir}/")


if __name__ == "__main__":
    main()
