"""
构建 room_type_clf 训练用 NPZ，包含预计算的 BERT 特征。

BERT 冻结，输出固定，在此预计算并存入 npz：
  训练时只需跑 text_proj（一个线性层），不再调用 BERT。

增强样本共用同一条文本特征，去重后存储：
  text_hidden   (n_unique, T, 768)  float16  BERT last_hidden_state
  text_attn_mask(n_unique, T)       bool     1=有效 token
  text_idx      (N,)                int32    每条样本对应的 unique 特征索引

其余字段：
  node_mask     (N, 40)             uint8
  adj_matrix    (N, 40, 40)         uint8
  room_membership(N, 40, MAX_ROOMS) float32
  type_labels   (N, 40)             int32    padding=-1

用法：
  python -m room_type_clf.build_npz \\
      --jsonl     data/jsonl/final_graph_dataset_v3.jsonl \\
      --val_jsonl data/jsonl/val_graph_dataset_18k5.jsonl \\
      --augment 4 --output data/processed/room_type_clf/train.npz
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from .dataset import load_combo_vocab, COMBO_VOCAB_PATH, MAX_NODES, MAX_TEXT_LEN
from .model import _assign_room_membership_single, MAX_ROOMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",       default="data/jsonl/final_graph_dataset_v3.jsonl")
    p.add_argument("--val_jsonl",   default="",
                   help="同时构建验证集，留空则跳过")
    p.add_argument("--bert",        default="models/bert-base-uncased")
    p.add_argument("--output",      default="data/processed/room_type_clf/train.npz")
    p.add_argument("--augment",     type=int, default=4,
                   help="节点顺序增强倍数，验证集固定=1")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--bert_batch",  type=int, default=64,
                   help="BERT 推理批次大小")
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


# ── 节点排列增强 ───────────────────────────────────────────────────────────────

def _permute(adj, labels, n, perm):
    full_perm  = perm + list(range(n, MAX_NODES))
    new_adj    = adj[np.ix_(full_perm, full_perm)]
    new_labels = labels[full_perm]
    return new_adj, new_labels


# ── room_membership 并行 worker ───────────────────────────────────────────────

def _compute_room_membership(args_tuple):
    idx, adj_row, n = args_tuple
    full = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    if n >= 3:
        m = _assign_room_membership_single(adj_row[:n, :n].astype(bool), n)
        full[:n, :] = m
    return idx, full


# ── BERT 批量推理 ─────────────────────────────────────────────────────────────

def run_bert(unique_input_ids, unique_attn_mask, bert_name, bert_batch=64):
    """对去重后的 unique prompts 跑 BERT，返回 fp16 last_hidden_state。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"BERT 推理设备: {device}  unique 条数: {len(unique_input_ids)}")

    bert = BertModel.from_pretrained(bert_name).to(device).eval()

    n = len(unique_input_ids)
    all_hidden = np.zeros((n, MAX_TEXT_LEN, 768), dtype=np.float16)

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, bert_batch):
            ids  = torch.from_numpy(unique_input_ids[i:i+bert_batch].astype(np.int64)).to(device)
            mask = torch.from_numpy(unique_attn_mask[i:i+bert_batch].astype(np.int64)).to(device)
            out  = bert(input_ids=ids, attention_mask=mask).last_hidden_state
            all_hidden[i:i+len(ids)] = out.cpu().to(torch.float16).numpy()
            if (i // bert_batch + 1) % 50 == 0:
                print(f"  BERT: {i+len(ids)}/{n}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    del bert
    torch.cuda.empty_cache()
    print(f"BERT 推理完成  ({time.perf_counter()-t0:.1f}s)")
    return all_hidden


# ── 处理单个 jsonl ─────────────────────────────────────────────────────────────

def process_jsonl(jsonl_path, tokenizer, max_samples=0, n_workers=1,
                  augment=1, rng=None):
    """
    返回:
      arrays        : dict，不含文本特征（文本由 BERT 单独处理）
      unique_ids    : [n_unique, T] int32  去重后的 input_ids
      unique_mask   : [n_unique, T] int32  去重后的 attention_mask
    """
    if rng is None:
        rng = random.Random(42)

    mask_list    = []
    adj_list     = []
    labels_list  = []
    n_nodes_list = []
    text_idx_list = []

    prompt_to_uid = {}   # prompt_str → unique_idx
    unique_ids_list  = []
    unique_mask_list = []

    n_graphs = n_skipped = 0
    t0 = time.perf_counter()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_samples > 0 and n_graphs >= max_samples:
                break

            rec    = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").strip()

            enc = tokenizer(prompt, add_special_tokens=True)
            if len(enc['input_ids']) > MAX_TEXT_LEN:
                n_skipped += 1
                continue

            n = min(int(rec["n_nodes"]), MAX_NODES)
            n_graphs += 1

            # 唯一文本去重
            if prompt not in prompt_to_uid:
                uid = len(unique_ids_list)
                prompt_to_uid[prompt] = uid
                padded   = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
                attn_msk = np.zeros(MAX_TEXT_LEN, dtype=np.int32)
                tlen = len(enc['input_ids'])
                padded[:tlen]   = enc['input_ids']
                attn_msk[:tlen] = enc['attention_mask']
                unique_ids_list.append(padded)
                unique_mask_list.append(attn_msk)
            uid = prompt_to_uid[prompt]

            # 邻接矩阵
            adj_raw = np.array(rec["adj_matrix"], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj_raw, 0)
            adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.uint8)
            adj_pad[:n, :n] = adj_raw.clip(0, 1)

            # node_mask
            mask = np.zeros(MAX_NODES, dtype=np.uint8)
            mask[:n] = 1

            # 节点类型标签
            combo_ids = rec.get("node_combo_ids", [])
            labels    = np.full(MAX_NODES, -1, dtype=np.int32)
            for i in range(n):
                labels[i] = int(combo_ids[i]) if i < len(combo_ids) else 0

            # 节点排列增强
            base_perm = list(range(n))
            perms = [base_perm]
            for _ in range(augment - 1):
                p = base_perm[:]
                rng.shuffle(p)
                perms.append(p)

            for perm in perms:
                new_adj, new_labels = _permute(adj_pad, labels, n, perm)
                mask_list.append(mask)
                adj_list.append(new_adj)
                labels_list.append(new_labels)
                n_nodes_list.append(n)
                text_idx_list.append(uid)

            if (line_no + 1) % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {line_no+1} 行 → {len(adj_list)} 条  unique_prompts={len(unique_ids_list)}"
                      f"  ({elapsed:.1f}s)")

    total = len(adj_list)
    print(f"共 {n_graphs} 张图（跳过 {n_skipped} 条文本过长），增强后 {total} 条"
          f"  unique_prompts={len(unique_ids_list)}")

    # 并行计算 room_membership
    print("计算 room_membership（并行）...")
    adj_arr  = np.stack(adj_list, axis=0)
    mask_arr = np.stack(mask_list, axis=0)
    tasks    = [(i, adj_arr[i], int(n_nodes_list[i])) for i in range(total)]
    membership_out = np.zeros((total, MAX_NODES, MAX_ROOMS), dtype=np.float32)
    with Pool(processes=n_workers) as pool:
        for done, (idx, m) in enumerate(
            pool.imap_unordered(_compute_room_membership, tasks, chunksize=256)
        ):
            membership_out[idx] = m
            if (done + 1) % 50000 == 0:
                print(f"  room_membership: {done+1}/{total}", flush=True)

    arrays = dict(
        node_mask       = mask_arr,
        adj_matrix      = adj_arr,
        room_membership = membership_out,
        type_labels     = np.stack(labels_list, axis=0),
        text_idx        = np.array(text_idx_list, dtype=np.int32),
    )
    unique_ids  = np.stack(unique_ids_list,  axis=0)  # [n_unique, T]
    unique_mask = np.stack(unique_mask_list, axis=0)  # [n_unique, T]
    return arrays, unique_ids, unique_mask


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    rng       = random.Random(args.seed)
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_types, _ = load_combo_vocab()
    print(f"组合类型数: {num_types}  (from {COMBO_VOCAB_PATH})")

    # ── 训练集 ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"\n处理训练集: {args.jsonl}  (augment={args.augment})")
    arrays, uid_ids, uid_mask = process_jsonl(
        args.jsonl, tokenizer, args.max_samples, n_workers, args.augment, rng)

    print("\n运行 BERT 推理（训练集）...")
    text_hidden = run_bert(uid_ids, uid_mask, args.bert, args.bert_batch)

    arrays['text_hidden']    = text_hidden          # [n_unique, T, 768] fp16
    arrays['text_attn_mask'] = uid_mask.astype(bool) # [n_unique, T] bool

    np.savez_compressed(out_path, **arrays)
    print(f"训练集 -> {out_path}  ({time.perf_counter()-t0:.1f}s)")
    _print_stats(arrays)

    # ── 验证集 ────────────────────────────────────────────────────────────────
    if args.val_jsonl:
        val_path = out_path.parent / (out_path.stem.replace('train', 'val') + '.npz')
        if val_path == out_path:
            val_path = out_path.parent / 'val.npz'
        t0 = time.perf_counter()
        print(f"\n处理验证集: {args.val_jsonl}  (augment=1)")
        val_arrays, val_uid_ids, val_uid_mask = process_jsonl(
            args.val_jsonl, tokenizer, 0, n_workers, augment=1)

        print("\n运行 BERT 推理（验证集）...")
        val_text = run_bert(val_uid_ids, val_uid_mask, args.bert, args.bert_batch)

        val_arrays['text_hidden']    = val_text
        val_arrays['text_attn_mask'] = val_uid_mask.astype(bool)

        np.savez_compressed(val_path, **val_arrays)
        print(f"验证集 -> {val_path}  ({time.perf_counter()-t0:.1f}s)")
        _print_stats(val_arrays)


def _print_stats(arrays):
    mask  = arrays['node_mask']
    valid = mask.astype(bool)
    mb    = arrays['room_membership']
    density = mb.sum(axis=2)[valid]
    labels  = arrays['type_labels']
    valid_labels = labels[valid]
    unique, counts = np.unique(valid_labels, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    n_unique = len(arrays['text_hidden'])
    print(f"  样本数: {len(mask)}  unique_prompts: {n_unique}")
    print(f"  平均节点数: {valid.sum(axis=1).mean():.1f}")
    print(f"  平均每节点属于 {density.mean():.2f} 个环")
    print(f"  top5 type_id: {[(int(uid), int(cnt)) for cnt, uid in top5]}")


if __name__ == "__main__":
    main()
