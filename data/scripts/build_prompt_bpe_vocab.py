"""
Build a reusable BPE tokenizer from the 5w prompt dataset.

Default source pair:
- data/processed/graph_tokens_combo_5w.npz
- data/processed/graph_prompts_5w.txt

The npz file is used only to determine the usable sample count so the prompt
side stays aligned with the token dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("data/processed/graph_tokens_combo_5w.npz"))
    parser.add_argument("--prompts", type=Path, default=Path("data/processed/graph_prompts_5w.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/graph_prompts_5w_bpe.json"))
    parser.add_argument("--stats_out", type=Path, default=Path("data/processed/graph_prompts_5w_bpe_stats.json"))
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all aligned prompts")
    return parser


def load_aligned_prompts(npz_path: Path, prompt_path: Path, limit: int) -> list[str]:
    raw = np.load(npz_path, allow_pickle=True)
    tokens = raw["tokens"]
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    usable_n = min(len(tokens), len(prompts))
    if len(tokens) != len(prompts):
        print(
            f"warning: prompt/token count mismatch, using first {usable_n} pairs "
            f"(npz={len(tokens)}, txt={len(prompts)})"
        )

    if limit > 0:
        usable_n = min(usable_n, limit)

    prompts = prompts[:usable_n]
    print(f"aligned prompts: {usable_n}")
    return prompts


def train_bpe(prompts: list[str], vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(prompts, trainer=trainer)
    return tokenizer


def compute_stats(tokenizer: Tokenizer, prompts: list[str]) -> dict:
    lengths = [len(tokenizer.encode(text).ids) for text in prompts]
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "num_prompts": int(len(prompts)),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "token_len_min": int(arr.min()) if len(arr) else 0,
        "token_len_mean": float(arr.mean()) if len(arr) else 0.0,
        "token_len_p95": int(np.percentile(arr, 95)) if len(arr) else 0,
        "token_len_max": int(arr.max()) if len(arr) else 0,
    }


def main():
    args = build_parser().parse_args()
    prompts = load_aligned_prompts(args.npz, args.prompts, args.limit)
    tokenizer = train_bpe(prompts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))

    stats = compute_stats(tokenizer, prompts)
    stats.update(
        {
            "npz": str(args.npz),
            "prompts": str(args.prompts),
            "out": str(args.out),
            "requested_vocab_size": int(args.vocab_size),
            "min_frequency": int(args.min_frequency),
            "limit": int(args.limit),
        }
    )
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"saved tokenizer to: {args.out}")
    print(f"saved stats to: {args.stats_out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
