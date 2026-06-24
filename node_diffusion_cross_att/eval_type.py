"""
NodeTypeClassifier 评估脚本（θ₃）

指标：
  1. Node Type Acc.  — 有效节点上的逐节点类型准确率
  2. Face Type Acc.  — 经投票算法后，每个面的房间类型准确率
                       (GT节点类型投票 vs 预测节点类型投票)

用法：
  python -m node_diffusion_cross_att.eval_type \
      --ckpt checkpoints/node_type/XXXXXX/model_latest.pt \
      --data data/jsonl/test_graph_dataset_10k.jsonl

可选：
  --n_eval 1000        只评估1000条
"""

import argparse
import ast
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer

from .type_model import NodeTypeClassifier


# ── Vocab ─────────────────────────────────────────────────────────────────────

def load_vocab(path: str) -> Dict[int, List[str]]:
    payload     = json.loads(Path(path).read_text(encoding="utf-8"))
    combo_to_id = payload["combo_to_id"]
    base_names  = payload["base_type_names"]
    id_to_combo: Dict[int, List[str]] = {}
    for combo_str, cid in combo_to_id.items():
        type_ids = ast.literal_eval(combo_str)
        id_to_combo[int(cid)] = [base_names[str(tid)] for tid in type_ids]
    return id_to_combo


# ── Half-edge face finder (from render.py) ────────────────────────────────────

def _build_sorted_neighbors(
    coords: List[Tuple[float, float]],
    adj: List[List[int]],
    n: int,
) -> Dict[int, List[int]]:
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
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


def _next_half_edge(u: int, v: int, sorted_nbrs: Dict[int, List[int]]) -> Optional[int]:
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face: List[int], coords: List[Tuple[float, float]]) -> float:
    pts = [coords[i] for i in face]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(
    coords: List[Tuple[float, float]],
    adj: List[List[int]],
) -> List[List[int]]:
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited: set = set()
    faces: List[List[int]] = []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face: List[int] = []
            cu, cv = u, v
            steps = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv))
                face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw
                steps += 1
            if len(face) >= 3:
                faces.append(face)
    if not faces:
        return []
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


# ── Room type voting (from render.py) ─────────────────────────────────────────

ROOM_TYPE_ORDER = [
    "bathroom", "bedroom", "living_room", "kitchen",
    "corridor", "dining_room", "other",
]


def vote_room_type(
    face: List[int],
    node_types: List[List[str]],
    all_nbrs: Dict[int, List[int]],
) -> str:
    face_set = set(face)
    face_counts: Counter = Counter()
    for node in face:
        for t in node_types[node]:
            face_counts[t] += 1
    if not face_counts:
        return "other"
    ext_counts: Counter = Counter()
    for node in face:
        for w in all_nbrs[node]:
            if w not in face_set:
                for t in node_types[w]:
                    ext_counts[t] += 1
    scores = {t: face_counts[t] / (ext_counts.get(t, 0) + 1) for t in face_counts}
    best_score = max(scores.values())
    winners = [t for t, s in scores.items() if s == best_score]
    order = {t: i for i, t in enumerate(ROOM_TYPE_ORDER)}
    return min(winners, key=lambda t: order.get(t, len(ROOM_TYPE_ORDER)))


# ── 面级准确率 ────────────────────────────────────────────────────────────────

def face_acc_for_sample(
    coords_np: np.ndarray,      # [N_valid, 2]
    adj_np: np.ndarray,         # [N_valid, N_valid]
    gt_combo_ids: np.ndarray,   # [N_valid]  int
    pred_combo_ids: np.ndarray, # [N_valid]  int
    id_to_combo: Dict[int, List[str]],
) -> Optional[Tuple[int, int]]:
    """
    返回 (correct_faces, total_faces)，无面则返回 None。
    """
    n = len(coords_np)
    coords = [(float(coords_np[i, 0]), float(coords_np[i, 1])) for i in range(n)]
    adj    = [[int(adj_np[i, j]) for j in range(n)] for i in range(n)]

    faces = find_faces(coords, adj)
    if not faces:
        return None

    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                all_nbrs[i].append(j)

    gt_types   = [id_to_combo.get(int(gt_combo_ids[i]),   ["other"]) for i in range(n)]
    pred_types = [id_to_combo.get(int(pred_combo_ids[i]), ["other"]) for i in range(n)]

    correct = 0
    for face in faces:
        gt_label   = vote_room_type(face, gt_types,   all_nbrs)
        pred_label = vote_room_type(face, pred_types, all_nbrs)
        if gt_label == pred_label:
            correct += 1

    return correct, len(faces)


# ── CLI ───────────────────────────────────────────────────────────────────────

MAX_BERT_LEN = 224


class JsonlTypeDataset(Dataset):
    """从 JSONL 读取测试集，格式与 TypeDataset 输出一致。"""

    def __init__(self, jsonl_path: str, bert_name: str):
        tok = BertTokenizer.from_pretrained(bert_name)
        self.coords     = []
        self.adj        = []
        self.mask       = []
        self.node_types = []
        self.ptok       = []
        self.pmsk       = []

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.coords.append(np.array(d['node_coords'],    dtype=np.float32))   # [40,2]
                self.adj.append(   np.array(d['adj_matrix'],     dtype=np.float32))   # [40,40]
                self.mask.append(  np.array(d['node_mask'],      dtype=np.float32))   # [40]
                self.node_types.append(np.array(d['node_combo_ids'], dtype=np.int64)) # [40]
                enc = tok(d['prompt'], max_length=MAX_BERT_LEN,
                          padding='max_length', truncation=True)
                self.ptok.append(np.array(enc['input_ids'],      dtype=np.int64))
                self.pmsk.append(np.array(enc['attention_mask'], dtype=np.float32))

        print(f'JsonlTypeDataset: {len(self.coords)} 条  ← {jsonl_path}')

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy()   # [2, 40]
        cond = {
            'adj_matrix':    self.adj[idx],
            'node_mask':     self.mask[idx],
            'node_types':    self.node_types[idx],
            'prompt_tokens': self.ptok[idx],
            'prompt_mask':   self.pmsk[idx],
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--data",       default="data/jsonl/test_graph_dataset_10k.jsonl")
    p.add_argument("--vocab",      default="node_diffusion_cross_att/type_combo_vocab_old.json")
    p.add_argument("--bert",       default="models/bert-base-uncased")
    p.add_argument("--n_eval",     type=int, default=0, help="0=全部")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="outputs/eval/type_eval_results.json")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = NodeTypeClassifier(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    raw_sd = ckpt["model"]
    if any(k.startswith("module.") for k in raw_sd):
        raw_sd = {k[7:]: v for k, v in raw_sd.items()}
    model.load_state_dict(raw_sd)
    model.eval()
    print(f"checkpoint: {args.ckpt}  step={ckpt.get('step', '?')}")

    # ── Vocab ─────────────────────────────────────────────────────────────────
    id_to_combo = load_vocab(args.vocab)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = JsonlTypeDataset(args.data, args.bert)
    total = len(dataset)
    print(f"数据集: {total} 条")

    all_coords    = np.stack(dataset.coords)     # [M, 40, 2]
    all_adj       = np.stack(dataset.adj)        # [M, 40, 40]
    all_node_mask = np.stack(dataset.mask)       # [M, 40]
    all_gt_types  = np.stack(dataset.node_types) # [M, 40]

    indices = list(range(total))
    if args.n_eval > 0 and args.n_eval < total:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(total, size=args.n_eval, replace=False).tolist()

    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    node_correct = node_valid = 0
    face_correct = face_total = 0

    # 收集所有预测（用于面级评估）
    all_pred_ids: List[np.ndarray] = []  # 每条样本 [40]

    with torch.no_grad():
        for i, (x, cond) in enumerate(loader):
            x    = x.to(device)
            cond = {k: v.to(device) for k, v in cond.items()}

            logits = model(
                x,
                adj_matrix    = cond["adj_matrix"],
                node_mask     = cond["node_mask"],
                prompt_tokens = cond["prompt_tokens"],
                prompt_mask   = cond["prompt_mask"],
            )                                        # [B, N, 33]

            targets   = cond["node_types"]           # [B, N]
            node_mask = cond["node_mask"]            # [B, N]
            pred      = logits.argmax(dim=-1)        # [B, N]
            valid     = node_mask > 0.5

            node_correct += ((pred == targets) & valid).sum().item()
            node_valid   += valid.sum().item()

            all_pred_ids.extend(pred.cpu().numpy())  # list of [N]

            if (i + 1) % 20 == 0:
                acc_so_far = node_correct / max(node_valid, 1)
                print(f"  [{min((i+1)*args.batch_size, len(subset))}/{len(subset)}]"
                      f" node_acc={acc_so_far:.4f}", flush=True)

    node_acc = node_correct / max(node_valid, 1)
    print(f"\nNode Type Acc. = {node_acc:.4f}  ({node_acc*100:.2f}%)")

    # ── 面级准确率 ────────────────────────────────────────────────────────────
    print("计算面级准确率 ...", flush=True)
    for rank, orig_idx in enumerate(indices):
        mask = all_node_mask[orig_idx]          # [40]
        n_valid = int(mask.sum())
        if n_valid < 3:
            continue

        coords_np  = all_coords[orig_idx, :n_valid]    # [n_valid, 2]
        adj_np     = all_adj[orig_idx, :n_valid, :n_valid]
        gt_ids     = all_gt_types[orig_idx, :n_valid]
        pred_ids   = all_pred_ids[rank][:n_valid]

        ret = face_acc_for_sample(coords_np, adj_np, gt_ids, pred_ids, id_to_combo)
        if ret is not None:
            face_correct += ret[0]
            face_total   += ret[1]

        if (rank + 1) % 500 == 0:
            f_acc = face_correct / max(face_total, 1)
            print(f"  [{rank+1}/{len(indices)}] face_acc={f_acc:.4f}", flush=True)

    face_acc = face_correct / max(face_total, 1)
    print(f"\n{'─'*40}")
    print(f"  Node Type Acc. : {node_acc:.4f}  ({node_acc*100:.2f}%)")
    print(f"  Face Type Acc. : {face_acc:.4f}  ({face_acc*100:.2f}%)")
    print(f"  valid nodes    : {node_valid}")
    print(f"  total faces    : {face_total}")
    print(f"{'─'*40}")

    result = {
        "node_type_acc":     round(node_acc,  6),
        "node_type_acc_pct": round(node_acc  * 100, 2),
        "face_type_acc":     round(face_acc,  6),
        "face_type_acc_pct": round(face_acc  * 100, 2),
        "valid_nodes":   node_valid,
        "total_faces":   face_total,
        "ckpt": args.ckpt,
        "data": args.data_path,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果保存至 {args.out}")


if __name__ == "__main__":
    main()
