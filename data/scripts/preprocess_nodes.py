"""
Preprocess train_nodes.jsonl for diffusion training.

Steps:
  1. Load each record from the jsonl file.
  2. Extract valid node coordinates (first n_nodes entries).
  3. Center coordinates around origin: subtract centroid of valid nodes.
  4. Normalize: divide by half the bounding-box diagonal (or fixed scale 128).
  5. Pad back to max_nodes=40 with zeros.
  6. Save as .npz cache with arrays:
       coords      [N, 40, 2]   float32, centered & normalized, padded
       adj_matrix  [N, 40, 40]  float32
       node_mask   [N, 40]      float32  (1=valid, 0=pad)
       prompts     [N]          object (str)
"""

import os
import json
import argparse
import numpy as np
from glob import glob

MAX_NODES = 40


def process_record(rec):
    n = rec["n_nodes"]
    coords_raw = np.array(rec["node_coords"], dtype=np.float32)  # [40, 2]
    node_mask  = np.array(rec["node_mask"],   dtype=np.float32)  # [40]
    adj        = np.array(rec["adj_matrix"],  dtype=np.float32)  # [40, 40]

    # center using only valid nodes, then normalize to [-1, 1]
    # 160 = empirical max absolute value after centering (actual max: 156.68)
    valid = coords_raw[:n]                    # [n, 2]
    center = valid.mean(axis=0)               # [2]
    valid  = (valid - center) / 160.0         # shift to origin and normalize

    coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
    coords[:n] = valid

    return coords, adj, node_mask, rec.get("prompt", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/jsonl/train_nodes.jsonl",
                        help="path to train_nodes.jsonl")
    parser.add_argument("--output", default="data/processed/nodes_train.npz",
                        help="output .npz path")
    parser.add_argument("--max_records", type=int, default=0,
                        help="limit number of records (0 = all)")
    args = parser.parse_args()

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

    all_coords, all_adj, all_mask, all_prompts = [], [], [], []
    skipped = 0
    for rec in records:
        if rec["n_nodes"] < 2:
            skipped += 1
            continue
        coords, adj, mask, prompt = process_record(rec)
        all_coords.append(coords)
        all_adj.append(adj)
        all_mask.append(mask)
        all_prompts.append(prompt)

    print(f"  kept {len(all_coords)}, skipped {skipped}")

    coords_arr = np.stack(all_coords, axis=0)   # [N, 40, 2]
    adj_arr    = np.stack(all_adj,    axis=0)   # [N, 40, 40]
    mask_arr   = np.stack(all_mask,   axis=0)   # [N, 40]
    prompts_obj = np.empty(len(all_prompts), dtype=object)
    for i, p in enumerate(all_prompts):
        prompts_obj[i] = p

    np.savez_compressed(
        args.output,
        coords=coords_arr,
        adj_matrix=adj_arr,
        node_mask=mask_arr,
        prompts=prompts_obj,
    )
    print(f"  saved → {args.output}")
    print(f"  coords shape : {coords_arr.shape},  range [{coords_arr.min():.3f}, {coords_arr.max():.3f}]")
    print(f"  adj   shape  : {adj_arr.shape}")
    print(f"  mask  shape  : {mask_arr.shape}")


if __name__ == "__main__":
    main()
