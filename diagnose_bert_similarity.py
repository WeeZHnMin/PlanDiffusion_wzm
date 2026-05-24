"""
Diagnose BERT embedding diversity across prompts.
Samples N prompts, encodes them, reports cosine similarity distribution.
If mean cosine sim > 0.95, embeddings are nearly identical → text conditioning is blind.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="number of prompts to sample for pairwise similarity")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading records...")
    records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    print(f"total records: {len(records)}")

    samples = random.sample(records, min(args.n_samples, len(records)))
    prompts = [r["prompt"] for r in samples]

    # show a few prompts to sanity-check diversity
    print("\n--- sample prompts ---")
    for p in prompts[:8]:
        print(" ", p)
    print()

    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert = BertModel.from_pretrained(str(args.bert)).to(device)
    bert.eval()

    print("encoding...")
    all_cls = []
    all_mean = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch = prompts[i:i+32]
            enc = tokenizer(batch, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = bert(**enc).last_hidden_state   # (B, seq, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            cls_vec = out[:, 0, :]                # CLS token
            mean_vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pooling
            all_cls.append(F.normalize(cls_vec, dim=-1).cpu())
            all_mean.append(F.normalize(mean_vec, dim=-1).cpu())

    cls_mat  = torch.cat(all_cls,  dim=0)   # (N, 768)
    mean_mat = torch.cat(all_mean, dim=0)   # (N, 768)

    def sim_stats(mat, name):
        # pairwise cosine similarity (upper triangle only)
        sim = mat @ mat.T                         # (N, N)
        n = mat.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        vals = sim[idx[0], idx[1]]
        print(f"\n--- {name} cosine similarity (N={n}, pairs={vals.numel()}) ---")
        print(f"  mean:  {vals.mean():.4f}")
        print(f"  std:   {vals.std():.4f}")
        print(f"  min:   {vals.min():.4f}")
        print(f"  max:   {vals.max():.4f}")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{p:02d}:   {vals.float().quantile(p/100):.4f}")

    sim_stats(cls_mat,  "CLS token")
    sim_stats(mean_mat, "mean pool")

    print("\n--- interpretation ---")
    mean_sim = (cls_mat @ cls_mat.T).triu(diagonal=1)
    n = cls_mat.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    avg = mean_sim[idx[0], idx[1]].mean().item()
    if avg > 0.98:
        print("CRITICAL: embeddings nearly identical (mean cos_sim > 0.98).")
        print("  Text conditioning is effectively blind — model sees no difference between prompts.")
    elif avg > 0.95:
        print("WARNING: embeddings very similar (mean cos_sim > 0.95).")
        print("  Text signal is very weak. Consider unfreezing more BERT layers or using a stronger encoder.")
    elif avg > 0.85:
        print("MODERATE: some diversity but embeddings cluster tightly.")
        print("  Text conditioning has limited signal. Watch whether val_loss tracks prompt content.")
    else:
        print("OK: embeddings are reasonably diverse. Text conditioning has signal.")


if __name__ == "__main__":
    main()
