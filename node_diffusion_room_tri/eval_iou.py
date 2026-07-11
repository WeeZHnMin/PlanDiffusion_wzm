"""
node_diffusion_room_tri IoU 评估脚本

Pipeline:
  1. 扩散模型生成节点坐标
  2. TextCondGNN 预测节点类型（combo_id → 类型字符串列表）
  3. 半边算法 find_faces + vote_room_type 投票确定房间类型
  4. Shapely 多边形 micro/macro IoU

用法：
  python -m node_diffusion_room_tri.eval_iou \
      --ckpt       checkpoints/node_diffusion_room_tri/latest.pt \
      --type_ckpt  checkpoints/node_type/model_latest.pt \
      --jsonl      data/jsonl/test_graph_dataset_18k5.jsonl \
      --n_samples  200 \
      --ddim_steps 200 \
      --batch_size 16
"""

import argparse
import ast
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
from shapely.ops import unary_union
from transformers import BertModel, BertTokenizer

from .model import NodeDiffusionTransformer, _assign_room_membership_single, MAX_ROOMS
from .diffusion import GaussianDiffusion
from .graph_prune import prune_dangling_nodes


# ── 内联 TextCondGNN（来自 node_diffusion_cross_att，避免跨包依赖）──────────────

def _attn(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, -1e4)
    scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)


class _MHA(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        out = _attn(q, k, v, self.d_k, mask, self.dropout)
        return self.out(out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k))


class _FF(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class _EncoderLayer(nn.Module):
    """dual-stream: adj_attn + global_attn → cross_attn → ffn（与 cross_att 版本完全一致）"""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.adj_attn    = _MHA(heads, d_model, dropout)
        self.global_attn = _MHA(heads, d_model, dropout)
        self.cross_attn  = _MHA(heads, d_model, dropout)
        self.ff          = _FF(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, adj_mask, text_feat, text_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn(x2, x2, x2, adj_mask) +
            self.global_attn(x2, x2, x2, None)
        )
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class TextCondGNN(nn.Module):
    """节点类型分类器，权重与 checkpoints/node_type/ 兼容。"""
    def __init__(self, d_model=384, num_layers=4, num_heads=6, dropout=0.1,
                 bert_name='models/bert-base-uncased', n_types=33, **kwargs):
        super().__init__()
        if 'model_channels' in kwargs:
            d_model = kwargs['model_channels']
        self.d_model  = d_model
        self.coord_emb = nn.Linear(2, d_model)
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.bert.config.hidden_size, d_model)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_types),
        )

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def forward(self, x, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        B = x.shape[0]
        node_feat = self.coord_emb(x.permute(0, 2, 1).float())
        adj_mask  = self._build_adj_mask(adj_matrix.float(), node_mask.float())
        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens, attention_mask=bert_attn,
                ).last_hidden_state
            text_feat = self.text_proj(text_hidden)
            text_mask = (1 - bert_attn.float()).unsqueeze(1)
        else:
            text_feat = torch.zeros(B, 1, self.d_model,
                                    device=node_feat.device, dtype=node_feat.dtype)
            text_mask = None
        for layer in self.layers:
            node_feat = layer(node_feat, adj_mask, text_feat, text_mask)
        return self.type_head(node_feat)

MAX_NODES    = 40
MAX_TEXT_LEN = 192

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]


# ── Vocab ─────────────────────────────────────────────────────────────────────

def load_vocab(path: str):
    """返回 {combo_id(int): [type_str, ...]}"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload["id_to_combo"].items()}


# ── 半边面算法 ────────────────────────────────────────────────────────────────

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


# ── 类型投票 ──────────────────────────────────────────────────────────────────

def vote_room_type(face, node_types, all_nbrs):
    """
    specificity score = face_count(t) / (ext_count(t) + 1)
    来自 node_diffusion_cross_att/render.py 同名函数。
    """
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


# ── DDIM 批次推理 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, diffusion, cond_batched, device, ddim_steps=200):
    diffusion._to(device)
    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()
    B = next(iter(cond_batched.values())).shape[0]
    x = torch.randn(B, 2, MAX_NODES, device=device)
    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps  = model(x, t_tensor, **cond_batched)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)
        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0
    return x  # [B, 2, MAX_NODES]


@torch.no_grad()
def ddpm_sample(model, diffusion, cond_batched, device, timesteps=1000):
    diffusion._to(device)
    B = next(iter(cond_batched.values())).shape[0]
    x = torch.randn(B, 2, MAX_NODES, device=device)
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
            x      = mean + diffusion.posterior_variance[t].sqrt() * torch.randn_like(x)
    return x  # [B, 2, MAX_NODES]


# ── 质心归零 ──────────────────────────────────────────────────────────────────

def center_at_origin(coords_np, mask_np):
    valid = coords_np[mask_np.astype(bool)]
    centroid = valid.mean(axis=0)
    return coords_np - centroid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="checkpoints/node_diffusion_room_tri/latest.pt")
    p.add_argument("--type_ckpt",   default="checkpoints/node_type/model_latest.pt",
                   help="TextCondGNN 节点类型分类器权重")
    p.add_argument("--vocab",       default="node_diffusion_room_tri/type_combo_vocab_v3.json")
    p.add_argument("--jsonl",       default="data/jsonl/test_graph_dataset_18k5.jsonl")
    p.add_argument("--n_samples",   type=int, default=224, help="0=全量")
    p.add_argument("--sampler",     default="ddim", choices=["ddim", "ddpm", "both"])
    p.add_argument("--ddim_steps",  type=int, default=200,
                   help="DDIM 步数（--sampler ddim 时生效）")
    p.add_argument("--timesteps",   type=int, default=1000,
                   help="DDPM 步数（--sampler ddpm 时生效）")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--bert",        default="models/bert-base-uncased")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    p.add_argument("--out",         default="", help="可选：结果保存路径(.json)")
    p.add_argument("--no_type_model", action="store_true",
                   help="不加载 TextCondGNN，直接用 GT node_types 作为预测类型")
    p.add_argument("--ddim_jsonl",  default="", help="Save per-sample DDIM inference results as JSONL")
    p.add_argument("--ddpm_jsonl",  default="", help="Save per-sample DDPM inference results as JSONL")
    args = p.parse_args()

    device    = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    # ── 加载扩散模型 ───────────────────────────────────────────────────────────
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
    print(f"Loaded diffusion: {args.ckpt}  step={ckpt.get('step', '?')}")

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── 加载类型分类器 ─────────────────────────────────────────────────────────
    if not args.no_type_model:
        type_model = TextCondGNN(
            d_model=args.model_channels,
            num_layers=4,
            num_heads=args.num_heads,
            bert_name=args.bert,
        ).to(device)
        type_ckpt = torch.load(args.type_ckpt, map_location=device)
        type_sd   = type_ckpt.get("model", type_ckpt)
        if any(k.startswith("module.") for k in type_sd):
            type_sd = {k[7:]: v for k, v in type_sd.items()}
        type_model.load_state_dict(type_sd, strict=False)
        type_model.eval()
        print(f"Loaded type model: {args.type_ckpt}  step={type_ckpt.get('step', '?')}")
        id_to_combo = load_vocab(args.vocab)
    else:
        type_model  = None
        id_to_combo = None
        print("--no_type_model: 使用 GT node_types 作为预测类型")

    # ── 第一步：预处理所有样本 ────────────────────────────────────────────────
    prepared = []
    skipped  = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if args.n_samples > 0 and len(prepared) + skipped >= args.n_samples:
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

            # GT 类型（来自 jsonl）
            gt_node_types = [
                (t if isinstance(t, list) else [t])
                for t in rec["node_types"][:n]
            ]
            raw_coords, adj_raw, gt_node_types, _ = prune_dangling_nodes(
                raw_coords, adj_raw, gt_node_types
            )
            n = len(raw_coords)
            if n < 3:
                skipped += 1
                continue
            adj_raw = adj_raw.astype(np.int32)

            gt_centered = center_at_origin(raw_coords, np.ones(n))
            adj_list    = adj_raw.tolist()
            gt_polys    = coords_to_polys_by_type(gt_centered, adj_list, gt_node_types, n)
            if not gt_polys:
                skipped += 1
                continue

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
                "mask_np":       mask_np,
                "adj_pad":       adj_pad,
                "membership":    membership,
                "ptok":          ptok,
                "pmsk":          pmsk,
                "n":             n,
                "adj_list":      adj_list,
                "gt_polys":      gt_polys,
                "gt_node_types": gt_node_types,
                "prompt":        prompt,
                "gt_node_coords": gt_centered.tolist(),
            })

    print(f"预处理完成: {len(prepared)} 条有效，{skipped} 条跳过")
    if args.sampler == "ddim":
        print(f"DDIM {args.ddim_steps} 步，batch_size={args.batch_size}，开始推理...")
    else:
        print(f"DDPM {args.timesteps} 步，batch_size={args.batch_size}，开始推理...")

    # ── 第二步：批次 DDIM + 类型预测 ─────────────────────────────────────────
    def run_one_eval(sampler, ddim_steps=None, timesteps=None, jsonl_path=""):
        sampler_info = f"DDIM {ddim_steps} steps" if sampler == "ddim" else f"DDPM {timesteps} steps"
        print(f"{sampler_info}, batch_size={args.batch_size}, start inference...", flush=True)

        all_pred_np = []
        all_pred_types = []
        t0 = time.time()
        VB = args.batch_size

        with torch.no_grad():
            for bi in range(0, len(prepared), VB):
                chunk = prepared[bi: bi + VB]
                B = len(chunk)

                mask_t = torch.from_numpy(np.stack([s["mask_np"] for s in chunk])).to(device)
                memb_t = torch.from_numpy(np.stack([s["membership"] for s in chunk])).to(device)
                adj_t = torch.from_numpy(np.stack([s["adj_pad"] for s in chunk])).to(device)
                ptok_t = torch.from_numpy(np.stack([s["ptok"] for s in chunk])).to(device)
                pmsk_t = torch.from_numpy(np.stack([s["pmsk"] for s in chunk])).to(device)

                cond_b = {
                    "node_mask": mask_t,
                    "room_membership": memb_t,
                    "adj_matrix": adj_t,
                    "prompt_tokens": ptok_t,
                    "prompt_mask": pmsk_t,
                }

                if sampler == "ddim":
                    pred_xy = ddim_sample(model, diffusion, cond_b, device, ddim_steps)
                else:
                    pred_xy = ddpm_sample(model, diffusion, cond_b, device, timesteps)

                if type_model is not None:
                    type_logits = type_model(
                        pred_xy,
                        adj_matrix=adj_t,
                        node_mask=mask_t,
                        prompt_tokens=ptok_t,
                        prompt_mask=pmsk_t,
                    )
                    combo_ids = type_logits.argmax(dim=-1).cpu().numpy()

                for j in range(B):
                    all_pred_np.append(pred_xy[j].cpu().numpy().T)
                    n_j = chunk[j]["n"]
                    if type_model is not None:
                        node_types_j = [
                            id_to_combo.get(int(combo_ids[j, k]), ["other"])
                            for k in range(n_j)
                        ]
                    else:
                        node_types_j = chunk[j]["gt_node_types"]
                    all_pred_types.append(node_types_j)

                done = min(bi + VB, len(prepared))
                print(f"  [{done}/{len(prepared)}]  {time.time() - t0:.1f}s", flush=True)

        micro_list, macro_list, samples_out = [], [], []
        for s, pred_np, pred_types in zip(prepared, all_pred_np, all_pred_types):
            n = s["n"]
            pred_centered = center_at_origin(pred_np, s["mask_np"])
            pred_polys = coords_to_polys_by_type(
                pred_centered[:n], s["adj_list"], pred_types, n)
            micro, macro = compute_iou(s["gt_polys"], pred_polys)
            micro_list.append(micro)
            macro_list.append(macro)
            samples_out.append({
                "prompt": s["prompt"],
                "n_nodes": n,
                "adj_matrix": s["adj_list"],
                "gt_node_coords": s["gt_node_coords"],
                "gt_node_types": s["gt_node_types"],
                "pred_node_coords": pred_centered[:n].tolist(),
                "pred_node_types": pred_types,
                "micro_iou": round(micro, 6),
                "macro_iou": round(macro, 6),
            })

        result = {
            "sampler": sampler,
            "ddim_steps": ddim_steps if sampler == "ddim" else None,
            "timesteps": timesteps if sampler == "ddpm" else None,
            "ckpt": args.ckpt,
            "type_ckpt": None if args.no_type_model else args.type_ckpt,
            "n": len(micro_list),
            "skipped": skipped,
            "micro_iou": float(np.mean(micro_list)),
            "macro_iou": float(np.mean(macro_list)),
            "micro_list": micro_list,
            "macro_list": macro_list,
            "samples": samples_out,
        }

        print(f"\n=== Eval done ({sampler_info}, {len(micro_list)} samples, skipped={skipped}) ===")
        print(f"Micro-IoU : {result['micro_iou']:.6f}")
        print(f"Macro-IoU : {result['macro_iou']:.6f}")
        print(f"Elapsed   : {time.time() - t0:.1f}s")

        if jsonl_path:
            sample_path = Path(jsonl_path)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_path.open("w", encoding="utf-8") as f:
                for row in samples_out:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"per-sample JSONL saved -> {sample_path}")

        return result

    results = []
    if args.sampler == "ddim":
        results.append(run_one_eval("ddim", ddim_steps=args.ddim_steps, jsonl_path=args.ddim_jsonl))
    elif args.sampler == "ddpm":
        results.append(run_one_eval("ddpm", timesteps=args.timesteps, jsonl_path=args.ddpm_jsonl))
    elif args.sampler == "both":
        results.append(run_one_eval("ddim", ddim_steps=args.ddim_steps, jsonl_path=args.ddim_jsonl))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(run_one_eval("ddpm", timesteps=args.timesteps, jsonl_path=args.ddpm_jsonl))
    else:
        raise ValueError(f"unsupported sampler: {args.sampler}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = [{k: v for k, v in res.items() if k != "samples"} for res in results]
        payload = summary[0] if len(summary) == 1 else {"results": summary}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"summary saved -> {out_path}")
    return

    all_pred_np        = []   # [MAX_NODES, 2] per sample
    all_pred_types     = []   # List[List[str]] per sample (len=n)
    t0 = time.time()
    VB = args.batch_size

    with torch.no_grad():
        for bi in range(0, len(prepared), VB):
            chunk = prepared[bi: bi + VB]
            B = len(chunk)

            mask_t  = torch.from_numpy(np.stack([s["mask_np"]    for s in chunk])).to(device)
            memb_t  = torch.from_numpy(np.stack([s["membership"] for s in chunk])).to(device)
            adj_t   = torch.from_numpy(np.stack([s["adj_pad"]    for s in chunk])).to(device)
            ptok_t  = torch.from_numpy(np.stack([s["ptok"]       for s in chunk])).to(device)
            pmsk_t  = torch.from_numpy(np.stack([s["pmsk"]       for s in chunk])).to(device)

            cond_b = {
                "node_mask":       mask_t,
                "room_membership": memb_t,
                "adj_matrix":      adj_t,
                "prompt_tokens":   ptok_t,
                "prompt_mask":     pmsk_t,
            }

            # 采样
            if args.sampler == "ddim":
                pred_xy = ddim_sample(model, diffusion, cond_b, device, args.ddim_steps)
            else:
                pred_xy = ddpm_sample(model, diffusion, cond_b, device, args.timesteps)
            # pred_xy: [B, 2, MAX_NODES]

            if type_model is not None:
                # TextCondGNN 预测类型
                type_logits = type_model(
                    pred_xy,
                    adj_matrix=adj_t,
                    node_mask=mask_t,
                    prompt_tokens=ptok_t,
                    prompt_mask=pmsk_t,
                )  # [B, MAX_NODES, N_TYPES]
                combo_ids = type_logits.argmax(dim=-1).cpu().numpy()

            for j in range(B):
                all_pred_np.append(pred_xy[j].cpu().numpy().T)  # [MAX_NODES, 2]
                n_j = chunk[j]["n"]
                if type_model is not None:
                    node_types_j = [
                        id_to_combo.get(int(combo_ids[j, k]), ["other"])
                        for k in range(n_j)
                    ]
                else:
                    node_types_j = chunk[j]["gt_node_types"]
                all_pred_types.append(node_types_j)

            done = min(bi + VB, len(prepared))
            print(f"  [{done}/{len(prepared)}]  {time.time() - t0:.1f}s", flush=True)

    # ── 第三步：逐样本计算 IoU ────────────────────────────────────────────────
    micro_list, macro_list = [], []
    for s, pred_np, pred_types in zip(prepared, all_pred_np, all_pred_types):
        pred_centered = center_at_origin(pred_np, s["mask_np"])
        pred_polys    = coords_to_polys_by_type(
            pred_centered[:s["n"]], s["adj_list"], pred_types, s["n"])
        micro, macro  = compute_iou(s["gt_polys"], pred_polys)
        micro_list.append(micro)
        macro_list.append(macro)

    print(f"\n=== 评估完成 ({len(micro_list)} 条, skipped={skipped}) ===")
    print(f"Micro-IoU : {np.mean(micro_list):.6f}")
    print(f"Macro-IoU : {np.mean(macro_list):.6f}")
    print(f"耗时      : {time.time() - t0:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt":      args.ckpt,
            "type_ckpt": args.type_ckpt,
            "n":         len(micro_list),
            "micro_iou": float(np.mean(micro_list)),
            "macro_iou": float(np.mean(macro_list)),
            "micro_list": micro_list,
            "macro_list": macro_list,
        }, indent=2))
        print(f"结果保存 → {args.out}")


if __name__ == "__main__":
    main()
