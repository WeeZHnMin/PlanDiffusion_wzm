import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from node_diffusion.dataset import COORD_SCALE, NodeDataset
from node_diffusion.diffusion import GaussianDiffusion
from node_diffusion.model import NodeDiffusionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate graph-only node diffusion checkpoints with full reverse diffusion."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/graph_tokens_combo_5w.npz",
        help="NPZ file containing coords/adj_matrix/node_mask.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path saved by node_diffusion/train.py or the notebook.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save aggregate metrics and per-sample RMSE.",
    )
    return parser.parse_args()


def masked_coord_metrics(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor):
    mask = node_mask.unsqueeze(1).float()
    diff = (pred - target) * mask
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)

    mse = (diff ** 2).sum(dim=(1, 2)) / denom
    mae = diff.abs().sum(dim=(1, 2)) / denom
    rmse_px = mse.sqrt() * COORD_SCALE
    mae_px = mae * COORD_SCALE
    return rmse_px, mae_px


def build_loader(data_path: str, num_samples: int, batch_size: int, seed: int) -> DataLoader:
    dataset = NodeDataset(data_path)
    total = min(num_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:total].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_loader(args.data_path, args.num_samples, args.batch_size, args.seed)

    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_rmse = []
    all_mae = []

    with torch.no_grad():
        for x, cond in loader:
            x = x.to(device)
            adj_matrix = cond["adj_matrix"].to(device)
            node_mask = cond["node_mask"].to(device)

            model_kwargs = {
                "adj_matrix": adj_matrix,
                "node_mask": node_mask,
            }
            pred = diffusion.p_sample_loop(
                model,
                shape=x.shape,
                model_kwargs=model_kwargs,
                device=device,
            )

            rmse_px, mae_px = masked_coord_metrics(pred, x, node_mask)
            all_rmse.extend(rmse_px.cpu().tolist())
            all_mae.extend(mae_px.cpu().tolist())

    metrics = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "num_samples": len(all_rmse),
        "rmse_px_mean": sum(all_rmse) / max(len(all_rmse), 1),
        "rmse_px_min": min(all_rmse) if all_rmse else None,
        "rmse_px_max": max(all_rmse) if all_rmse else None,
        "mae_px_mean": sum(all_mae) / max(len(all_mae), 1),
        "mae_px_min": min(all_mae) if all_mae else None,
        "mae_px_max": max(all_mae) if all_mae else None,
        "per_sample_rmse_px": all_rmse,
        "per_sample_mae_px": all_mae,
    }

    print(f"device: {device}")
    print(f"checkpoint: {args.ckpt}")
    print(f"evaluated samples: {metrics['num_samples']}")
    print(
        f"RMSE(px): mean={metrics['rmse_px_mean']:.2f} "
        f"min={metrics['rmse_px_min']:.2f} max={metrics['rmse_px_max']:.2f}"
    )
    print(
        f"MAE(px):  mean={metrics['mae_px_mean']:.2f} "
        f"min={metrics['mae_px_min']:.2f} max={metrics['mae_px_max']:.2f}"
    )

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved metrics -> {save_path}")


if __name__ == "__main__":
    main()
