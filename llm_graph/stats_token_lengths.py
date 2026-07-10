"""
Measure llm_graph tokenizer prompt length distribution on a JSONL dataset.

Examples:
  python -m llm_graph.stats_token_lengths
  python -m llm_graph.stats_token_lengths --jsonl data/jsonl/graph_160k_spatial.jsonl
  python -m llm_graph.stats_token_lengths --thresholds 128 160 192 224 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/jsonl/graphs_160k_spatial.jsonl")
    p.add_argument("--bpe", default="llm_graph/vocab/wp_tokenizer.json")
    p.add_argument("--text-key", default="prompt")
    p.add_argument("--progress-every", type=int, default=50000)
    p.add_argument("--max-text-len", type=int, default=224,
                   help="Report how many samples would be truncated by this limit.")
    p.add_argument("--thresholds", type=int, nargs="*", default=[128, 160, 192, 224, 256])
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    tokenizer = Tokenizer.from_file(args.bpe)

    lengths = []
    empty_text = 0
    t0 = time.perf_counter()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            text = rec.get(args.text_key, "")
            text = text.replace("\n", " ").strip()
            if not text:
                empty_text += 1

            token_ids = tokenizer.encode(text).ids
            lengths.append(len(token_ids))

            if args.progress_every > 0 and line_no % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"processed={line_no} elapsed={elapsed:.1f}s")

    if not lengths:
        raise RuntimeError(f"No valid samples found in {jsonl_path}")

    lengths = np.array(lengths, dtype=np.int32)
    total = len(lengths)
    elapsed = time.perf_counter() - t0

    print(f"jsonl={jsonl_path}")
    print(f"tokenizer={args.bpe}")
    print(f"text_key={args.text_key}")
    print(f"total={total} empty_text={empty_text} elapsed={elapsed:.1f}s")

    print("\nLength distribution")
    for pct in [50, 90, 95, 99, 100]:
        print(f"  p{pct:3d}: {int(np.percentile(lengths, pct))} tokens")
    print(f"  mean: {lengths.mean():.2f}")
    print(f"  std : {lengths.std():.2f}")

    print("\nThreshold ratios")
    for threshold in args.thresholds:
        n_over = int((lengths > threshold).sum())
        print(f"  > {threshold:3d}: {n_over:7d} / {total} ({100.0 * n_over / total:.2f}%)")

    n_trunc = int((lengths > args.max_text_len).sum())
    print(f"\nWould truncate at {args.max_text_len}: {n_trunc} / {total} ({100.0 * n_trunc / total:.2f}%)")


if __name__ == "__main__":
    main()
