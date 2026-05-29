"""
Freeze a reproducible subset from the old autograph prompt/token dataset.

Default source pair:
- data/processed/graph_tokens_combo_from_final_old.npz
- data/processed/graph_prompts_combo_from_final_old.txt

Outputs:
- subset index file (.npy)
- optional materialized subset npz/txt
- summary json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("data/processed/graph_tokens_combo_from_final_old.npz"))
    parser.add_argument("--prompts", type=Path, default=Path("data/processed/graph_prompts_combo_from_final_old.txt"))
    parser.add_argument("--subset_size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index_out",
        type=Path,
        default=Path("data/processed/subsets/autograph_old_subset_100k_seed42_indices.npy"),
    )
    parser.add_argument(
        "--subset_npz_out",
        type=Path,
        default=Path("data/processed/subsets/graph_tokens_combo_from_final_old_100k_seed42.npz"),
    )
    parser.add_argument(
        "--subset_prompts_out",
        type=Path,
        default=Path("data/processed/subsets/graph_prompts_combo_from_final_old_100k_seed42.txt"),
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=Path("data/processed/subsets/autograph_old_subset_100k_seed42_summary.json"),
    )
    parser.add_argument("--no_export_subset", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    raw = np.load(args.npz, allow_pickle=True)
    prompts = args.prompts.read_text(encoding="utf-8").splitlines()

    tokens = raw["tokens"]
    usable_n = min(len(tokens), len(prompts))
    if len(tokens) != len(prompts):
        print(
            f"warning: prompt/token count mismatch, using first {usable_n} pairs "
            f"(npz={len(tokens)}, txt={len(prompts)})"
        )

    subset_size = min(args.subset_size, usable_n)
    rng = np.random.default_rng(args.seed)
    subset_indices = np.sort(rng.choice(usable_n, size=subset_size, replace=False)).astype(np.int32)

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.index_out, subset_indices)

    if not args.no_export_subset:
        subset_npz = {}
        for key in raw.files:
            subset_npz[key] = raw[key][:usable_n][subset_indices]
        args.subset_npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.subset_npz_out, **subset_npz)

        subset_prompts = [prompts[i] for i in subset_indices]
        args.subset_prompts_out.parent.mkdir(parents=True, exist_ok=True)
        args.subset_prompts_out.write_text("\n".join(subset_prompts) + "\n", encoding="utf-8")

    summary = {
        "source_npz": str(args.npz),
        "source_prompts": str(args.prompts),
        "usable_pairs": int(usable_n),
        "subset_size": int(subset_size),
        "seed": int(args.seed),
        "index_out": str(args.index_out),
        "subset_npz_out": None if args.no_export_subset else str(args.subset_npz_out),
        "subset_prompts_out": None if args.no_export_subset else str(args.subset_prompts_out),
        "first_10_indices": subset_indices[:10].tolist(),
        "last_10_indices": subset_indices[-10:].tolist(),
    }

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved indices to: {args.index_out}")
    if not args.no_export_subset:
        print(f"saved subset npz to: {args.subset_npz_out}")
        print(f"saved subset prompts to: {args.subset_prompts_out}")
    print(f"saved summary to: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
