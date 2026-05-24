"""
Evaluate text encoder similarity before vs after fingerprint fine-tuning.
Compares raw BERT (original) vs fine-tuned BERT on the same prompts.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=Path, default=Path("data/jsonl/train_nodes_phase1.jsonl"))
    parser.add_argument("--bert-orig",   type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--bert-tuned",  type=Path, default=Path("models/bert-finetuned-fp"))
    parser.add_argument("--n-samples",   type=int,  default=500)
    parser.add_argument("--max-length",  type=int,  default=128)
    parser.add_argument("--seed",        type=int,  default=42)
    return parser.parse_args()


def encode_prompts(bert, tokenizer, prompts, max_length, device, batch_size=32):
    bert.eval()
    all_cls  = []
    all_mean = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            enc   = tokenizer(batch, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=max_length)
            enc   = {k: v.to(device) for k, v in enc.items()}
            hs    = bert(**enc).last_hidden_state           # (B, seq, hidden)
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            cls_v = F.normalize(hs[:, 0, :], dim=-1)
            mean_v = F.normalize((hs * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)
            all_cls.append(cls_v.cpu())
            all_mean.append(mean_v.cpu())
    return torch.cat(all_cls), torch.cat(all_mean)


def sim_stats(mat, name):
    n   = mat.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    sim = (mat @ mat.T)[idx[0], idx[1]]
    print(f"  {name}:")
    print(f"    mean={sim.mean():.4f}  std={sim.std():.4f}  "
          f"p50={sim.median():.4f}  p95={sim.float().quantile(0.95):.4f}  "
          f"min={sim.min():.4f}")


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading prompts...")
    records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    records = random.sample(records, min(args.n_samples, len(records)))
    prompts = [r["prompt"] for r in records]
    print(f"n={len(prompts)}")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_orig))

    print("\nencoding with original BERT...")
    bert_orig = BertModel.from_pretrained(str(args.bert_orig)).to(device)
    cls_orig, mean_orig = encode_prompts(bert_orig, tokenizer, prompts, args.max_length, device)
    del bert_orig; torch.cuda.empty_cache()

    print("encoding with fine-tuned BERT...")
    bert_tuned = BertModel.from_pretrained(str(args.bert_tuned)).to(device)
    cls_tuned, mean_tuned = encode_prompts(bert_tuned, tokenizer, prompts, args.max_length, device)
    del bert_tuned; torch.cuda.empty_cache()

    print("\n=== cosine similarity between prompt pairs ===")
    print("[ original BERT ]")
    sim_stats(cls_orig,  "CLS token")
    sim_stats(mean_orig, "mean pool")
    print("\n[ fine-tuned BERT ]")
    sim_stats(cls_tuned,  "CLS token")
    sim_stats(mean_tuned, "mean pool")

    print("\n=== delta (fine-tuned - original) ===")
    def mean_sim(mat):
        n = mat.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        return (mat @ mat.T)[idx[0], idx[1]].mean().item()

    for name, orig, tuned in [("CLS  ", cls_orig, cls_tuned),
                               ("mean ", mean_orig, mean_tuned)]:
        d = mean_sim(tuned) - mean_sim(orig)
        sign = "↓ more diverse" if d < -0.01 else ("↑ more similar" if d > 0.01 else "≈ no change")
        print(f"  {name}: {mean_sim(orig):.4f} → {mean_sim(tuned):.4f}  ({d:+.4f})  {sign}")


if __name__ == "__main__":
    main()
