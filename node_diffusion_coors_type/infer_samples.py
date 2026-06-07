"""
Run conditional diffusion sampling on training-set examples and report final metrics.

Example:
    python -m node_diffusion.infer_samples ^
        --checkpoint checkpoints/node_diffusion/model_0140000.pt ^
        --data_path data/processed/nodes_train_6k_norm.npz ^
        --num_samples 8
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .model import NodeDiffusionTransformer


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    parser.add_argument("--data_path", default=defaults.get("data_path", "data/processed/nodes_train.npz"))
    parser.add_argument("--out_dir", default=defaults.get("out_dir", "outputs/node_diffusion_samples"))
    parser.add_argument("--num_samples", type=int, default=defaults.get("num_samples", 8))
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="optional explicit sample indices")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--device", default=defaults.get("device", "cpu"))
    parser.add_argument("--model_channels", type=int, default=defaults.get("model_channels", 256))
    parser.add_argument("--num_layers", type=int, default=defaults.get("num_layers", 6))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 4))
    parser.add_argument("--timesteps", type=int, default=defaults.get("timesteps", 1000))
    parser.add_argument("--clamp", type=float, default=defaults.get("clamp", 200.0))
    parser.add_argument(
        "--coord_scale",
        type=float,
        default=defaults.get("coord_scale"),
        help="multiply coords by this factor for metrics/plots; default auto-detects from data",
    )
    parser.add_argument("--save_npz", action="store_true", help="save sampled arrays to samples.npz")
    return parser


def choose_indices(total, num_samples, explicit, seed):
    if explicit:
        return explicit
    rng = random.Random(seed)
    count = min(num_samples, total)
    return sorted(rng.sample(range(total), count))


def load_examples(npz_path, indices):
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"][indices].astype(np.float32)
    adj = data["adj_matrix"][indices].astype(np.float32)
    mask = data["node_mask"][indices].astype(np.float32)
    prompts = data["prompts"][indices] if "prompts" in data else np.array([""] * len(indices), dtype=object)
    return coords, adj, mask, prompts


def build_model(args, device):
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def infer_coord_scale(coords_np, explicit_scale):
    if explicit_scale is not None:
        return explicit_scale
    max_abs = float(np.max(np.abs(coords_np)))
    return 160.0 if max_abs <= 2.0 else 1.0


def compute_metrics(gt_coords, pred_coords, mask, coord_scale):
    valid = np.repeat((mask[:, None] > 0.5), gt_coords.shape[-1], axis=1)
    diff = (pred_coords - gt_coords) * coord_scale
    sq = (diff ** 2)[valid]
    abs_err = np.abs(diff)[valid]
    rmse = float(np.sqrt(np.mean(sq))) if sq.size else 0.0
    mae = float(np.mean(abs_err)) if abs_err.size else 0.0
    return rmse, mae


def render_pair(gt_coords, pred_coords, adj, mask, title, out_path, coord_scale):
    n = int(mask.sum())
    gt_xy = gt_coords[:n] * coord_scale
    pred_xy = pred_coords[:n] * coord_scale

    both = np.concatenate([gt_xy, pred_xy], axis=0)
    x_pad = max(10.0, 0.1 * (both[:, 0].max() - both[:, 0].min() + 1e-6))
    y_pad = max(10.0, 0.1 * (both[:, 1].max() - both[:, 1].min() + 1e-6))
    xlim = (both[:, 0].min() - x_pad, both[:, 0].max() + x_pad)
    ylim = (both[:, 1].min() - y_pad, both[:, 1].max() + y_pad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, coords_xy, name in zip(axes, [gt_xy, pred_xy], ["GT", "Sampled"]):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0.5:
                    ax.plot(
                        [coords_xy[i, 0], coords_xy[j, 0]],
                        [coords_xy[i, 1], coords_xy[j, 1]],
                        color="steelblue",
                        lw=1.0,
                        alpha=0.7,
                        zorder=1,
                    )
        ax.scatter(coords_xy[:, 0], coords_xy[:, 1], c=np.arange(n), cmap="tab20", s=35, zorder=2)
        for k in range(n):
            ax.text(coords_xy[k, 0] + 1.5, coords_xy[k, 1] + 1.5, str(k), fontsize=7)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.invert_yaxis()
        ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None, defaults=None):
    args = build_parser(defaults).parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.data_path, allow_pickle=True)
    total = len(data["coords"])
    indices = choose_indices(total, args.num_samples, args.indices, args.seed)
    data.close()

    coords_np, adj_np, mask_np, prompts = load_examples(args.data_path, indices)
    coord_scale = infer_coord_scale(coords_np, args.coord_scale)

    device = torch.device(args.device)
    model, ckpt = build_model(args, device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    pred_list = []
    metrics = []

    with torch.no_grad():
        for local_i, sample_idx in enumerate(indices):
            x_gt = torch.from_numpy(coords_np[local_i:local_i + 1]).permute(0, 2, 1).to(device)
            cond = {
                "adj_matrix": torch.from_numpy(adj_np[local_i:local_i + 1]).to(device),
                "node_mask": torch.from_numpy(mask_np[local_i:local_i + 1]).to(device),
            }
            pred = diffusion.p_sample_loop(
                model=model,
                shape=x_gt.shape,
                model_kwargs=cond,
                device=device,
                clamp=args.clamp,
            )
            pred = pred * cond["node_mask"].unsqueeze(1)
            pred_xy = pred.permute(0, 2, 1).cpu().numpy()[0]
            pred_list.append(pred_xy)

            rmse_px, mae_px = compute_metrics(coords_np[local_i], pred_xy, mask_np[local_i], coord_scale)
            metrics.append({
                "index": int(sample_idx),
                "n_nodes": int(mask_np[local_i].sum()),
                "rmse_px": rmse_px,
                "mae_px": mae_px,
                "prompt": str(prompts[local_i]),
            })

    pred_np = np.stack(pred_list, axis=0)
    overall_rmse_px, overall_mae_px = compute_metrics(
        coords_np.reshape(-1, coords_np.shape[-1]),
        pred_np.reshape(-1, pred_np.shape[-1]),
        mask_np.reshape(-1),
        coord_scale,
    )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "data_path": args.data_path,
        "coord_scale": coord_scale,
        "indices": [int(i) for i in indices],
        "num_samples": len(indices),
        "overall_rmse_px": overall_rmse_px,
        "overall_mae_px": overall_mae_px,
        "samples": metrics,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_npz:
        np.savez_compressed(
            os.path.join(args.out_dir, "samples.npz"),
            indices=np.array(indices, dtype=np.int32),
            gt_coords=coords_np,
            pred_coords=pred_np,
            adj_matrix=adj_np,
            node_mask=mask_np,
            prompts=np.array(prompts, dtype=object),
        )

    for local_i, sample_idx in enumerate(indices):
        sample = metrics[local_i]
        title = (
            f"idx={sample_idx} | n={sample['n_nodes']} | "
            f"RMSE={sample['rmse_px']:.2f}px | MAE={sample['mae_px']:.2f}px"
        )
        prompt = sample["prompt"]
        if prompt:
            title += f" | {prompt[:80]}"
        out_path = os.path.join(args.out_dir, f"sample_{sample_idx:05d}.png")
        render_pair(
            gt_coords=coords_np[local_i],
            pred_coords=pred_np[local_i],
            adj=adj_np[local_i],
            mask=mask_np[local_i],
            title=title,
            out_path=out_path,
            coord_scale=coord_scale,
        )

    print(f"loaded checkpoint step: {ckpt.get('step', 'unknown')}")
    print(f"coord scale used: {coord_scale}")
    print(f"sample indices: {indices}")
    print(f"overall final RMSE: {overall_rmse_px:.2f} px")
    print(f"overall final MAE : {overall_mae_px:.2f} px")
    for sample in metrics:
        print(
            f"idx {sample['index']:5d} | n {sample['n_nodes']:2d} | "
            f"RMSE {sample['rmse_px']:.2f} px | MAE {sample['mae_px']:.2f} px"
        )
    print(f"saved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
