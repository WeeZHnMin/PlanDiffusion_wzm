"""
Remove records from train_nodes.jsonl whose prompt exceeds max_length tokens.
Usage:
    python data/filter_long_prompts.py
    python data/filter_long_prompts.py --max-length 128 --input data/jsonl/train_nodes.jsonl
"""

import argparse
import json
from pathlib import Path

from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Loading tokenizer from {args.bert} ...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))

    tmp = args.input.with_suffix(".jsonl.tmp")
    kept = removed = bad = 0

    with args.input.open(encoding="utf-8", errors="replace") as fin, \
         tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            length = len(tokenizer.encode(r["prompt"], add_special_tokens=True))
            if length <= args.max_length:
                fout.write(s + "\n")
                kept += 1
            else:
                removed += 1

    tmp.replace(args.input)
    print(f"kept={kept}  removed={removed}  bad_json={bad}")
    print(f"Output: {args.input}")


if __name__ == "__main__":
    main()
