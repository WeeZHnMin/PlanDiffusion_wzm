"""
Preprocess train_nodes.jsonl for node diffusion training.

Two supported coordinate variants:
1. Raw-centered: subtract per-sample centroid only
2. Normalized: subtract centroid, then divide by a fixed scale

Examples:
    python data/scripts/preprocess_nodes.py
    python data/scripts/preprocess_nodes.py --normalize --output data/processed/nodes_train_norm.npz
"""

import argparse
import json
import os

import numpy as np


MAX_NODES = 40


def process_record(rec, normalize=False, scale=160.0):
    n = rec["n_nodes"]
    coords_raw = np.array(rec["node_coords"], dtype=np.float32)
    node_mask = np.array(rec["node_mask"], dtype=np.float32)
    adj = np.array(rec["adj_matrix"], dtype=np.float32)

    valid = coords_raw[:n]
    center = valid.mean(axis=0)
    valid = valid - center
    if normalize:
        valid = valid / scale

    coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
    coords[:n] = valid
    return coords, adj, node_mask, rec.get("prompt", "")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/jsonl/train_nodes.jsonl", help="path to train_nodes.jsonl")
    parser.add_argument("--output", default="data/processed/nodes_train.npz", help="output .npz path")
    parser.add_argument("--max_records", type=int, default=0, help="limit number of records (0 = all)")
    parser.add_argument("--normalize", action="store_true", help="apply fixed-scale coordinate normalization")
    parser.add_argument("--scale", type=float, default=160.0, help="normalization divisor when --normalize is set")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"reading {args.input} ...")
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if args.max_records > 0 and len(records) >= args.max_records:
                break
    print(f"  loaded {len(records)} records")
    print(f"  normalize: {args.normalize} | scale: {args.scale}")

    all_coords, all_adj, all_mask, all_prompts = [], [], [], []
    skipped = 0
    for rec in records:
        if rec["n_nodes"] < 2:
            skipped += 1
            continue
        coords, adj, mask, prompt = process_record(
            rec,
            normalize=args.normalize,
            scale=args.scale,
        )
        all_coords.append(coords)
        all_adj.append(adj)
        all_mask.append(mask)
        all_prompts.append(prompt)

    print(f"  kept {len(all_coords)}, skipped {skipped}")

    coords_arr = np.stack(all_coords, axis=0)
    adj_arr = np.stack(all_adj, axis=0)
    mask_arr = np.stack(all_mask, axis=0)
    prompts_obj = np.empty(len(all_prompts), dtype=object)
    for i, prompt in enumerate(all_prompts):
        prompts_obj[i] = prompt

    np.savez_compressed(
        args.output,
        coords=coords_arr,
        adj_matrix=adj_arr,
        node_mask=mask_arr,
        prompts=prompts_obj,
    )
    print(f"  saved -> {args.output}")
    print(f"  coords shape : {coords_arr.shape}, range [{coords_arr.min():.3f}, {coords_arr.max():.3f}]")
    print(f"  adj   shape  : {adj_arr.shape}")
    print(f"  mask  shape  : {mask_arr.shape}")


if __name__ == "__main__":
    main()
