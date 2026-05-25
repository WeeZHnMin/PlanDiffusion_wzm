"""
Logistic regression experiment: text -> adj_half (binary).
One binary classifier per upper-triangle position (one-vs-rest style).
Goal: check if the model can overfit 1000 samples.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# ── config ──────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/jsonl/train_nodes_rel_half.jsonl")
BERT_PATH  = Path("models/bert-base-chinese")
N_SAMPLES  = 1000
N_MAX      = 40
MAX_LENGTH = 128
# ────────────────────────────────────────────────────────────────────────────────


def load_records(path, n):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
            if len(records) >= n:
                break
    return records


def encode_texts(records, bert_path, max_length):
    print("loading BERT ...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(bert_path))
    bert      = BertModel.from_pretrained(str(bert_path)).to(device).eval()

    prompts = [r["prompt"] for r in records]
    enc     = tokenizer(prompts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_length)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    print("encoding texts ...")
    all_vecs = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            out = bert(input_ids=input_ids[i:i+batch_size],
                       attention_mask=attention_mask[i:i+batch_size])
            cls = out.last_hidden_state[:, 0, :]   # CLS token [B, 768]
            all_vecs.append(cls.cpu().float().numpy())

    return np.concatenate(all_vecs, axis=0)   # [N, 768]


def build_adj_targets(records, n_max):
    """Flatten upper triangle (j > i) of adj into a fixed [n_max*(n_max-1)//2] vector.
    Falls back to deriving adj from rel_adj_half when adj_matrix is absent."""
    n_tri = n_max * (n_max - 1) // 2   # 780 for n_max=40
    targets = np.zeros((len(records), n_tri), dtype=np.float32)

    for idx, r in enumerate(records):
        adj_padded = np.zeros((n_max, n_max), dtype=np.float32)

        if "adj_matrix" in r:
            for i, row in enumerate(r["adj_matrix"]):
                if i >= n_max:
                    break
                for k, val in enumerate(row):
                    j = i + k
                    if j < n_max:
                        adj_padded[i, j] = float(val)
        else:
            for i, row in enumerate(r["rel_adj_half"]):
                if i >= n_max:
                    break
                for k, val in enumerate(row):
                    j = i + k
                    if j < n_max and (val[0] != 0 or val[1] != 0):
                        adj_padded[i, j] = 1.0

        # extract upper triangle (j > i)
        pos = 0
        for i in range(n_max):
            for j in range(i + 1, n_max):
                targets[idx, pos] = adj_padded[i, j]
                pos += 1

    return targets   # [N, 780]


def main():
    print(f"loading {N_SAMPLES} records ...")
    records = load_records(DATA_PATH, N_SAMPLES)
    print(f"loaded {len(records)} records")

    has_adj = sum(1 for r in records if "adj_matrix" in r)
    print(f"records with adj_matrix: {has_adj}/{len(records)}  (rest derived from rel_adj_half)")

    X = encode_texts(records, BERT_PATH, MAX_LENGTH)   # [N, 768]
    y = build_adj_targets(records, N_MAX)               # [N, 780]

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"edge ratio: {y.mean():.4f}  ({y.sum():.0f} / {y.size} positions)")

    # skip columns with only one class (always 0 in this subset)
    active_cols = np.where(y.sum(axis=0) > 0)[0]
    inactive_cols = np.where(y.sum(axis=0) == 0)[0]
    print(f"\nactive adj positions (have ≥1 edge in 1000 samples): {len(active_cols)} / {y.shape[1]}")
    print(f"always-zero positions (skipped):                      {len(inactive_cols)} / {y.shape[1]}")

    print("\ntraining logistic regression (one classifier per active adj position) ...")
    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        n_jobs=-1,
    )
    clf.fit(X, y[:, active_cols])

    pred = np.zeros_like(y)
    pred[:, active_cols] = clf.predict(X)

    # ── metrics ────────────────────────────────────────────────────────────────
    overall_acc = accuracy_score(y.reshape(-1), pred.reshape(-1))

    edge_mask   = y == 1
    noedge_mask = y == 0
    edge_acc    = (pred[edge_mask]   == 1).mean() if edge_mask.any()   else float("nan")
    noedge_acc  = (pred[noedge_mask] == 0).mean() if noedge_mask.any() else float("nan")

    # per-sample: all positions correct?
    per_sample_acc = (pred == y).all(axis=1).mean()

    print(f"\n── train results (overfit check) ──────────────────")
    print(f"  overall acc    : {overall_acc:.4f}")
    print(f"  edge acc       : {edge_acc:.4f}   (有边位置预测对了多少)")
    print(f"  no-edge acc    : {noedge_acc:.4f}  (无边位置预测对了多少)")
    print(f"  per-sample acc : {per_sample_acc:.4f}  (整条adj完全正确的比例)")
    print(f"───────────────────────────────────────────────────")

    if edge_acc > 0.5 and noedge_acc > 0.5:
        print("→ 文本包含足够信息，逻辑回归能从文本预测图结构")
    else:
        print("→ 文本信息不足以单独预测图结构，需要其他条件")


if __name__ == "__main__":
    main()
