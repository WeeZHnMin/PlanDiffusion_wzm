"""
Compatibility and inference checker for the `fbbb543` node_diffusion variant.

This script uses the fbbb543-style setup:
- dual attention transformer
- x0-prediction DDPM sampler
- unnormalized but centered coordinates (`nodes_train.npz`)

It tries to strict-load a list of checkpoints, then samples a few fixed
training-set examples to see whether the generated layouts look plausible.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        out = attention(q, k, v, self.d_k, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, pad_mask):
        x2 = self.norm1(x)
        x = (
            x
            + self.dropout(self.adj_attn(x2, x2, x2, adj_mask))
            + self.dropout(self.global_attn(x2, x2, x2, pad_mask))
        )
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    def __init__(self, model_channels=256, num_layers=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)
        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeDiffusionTransformer: {n_params:,} parameters")

    def _build_masks(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        adj_mask = torch.clamp(adj_mask + pad_keys, 0, 1)
        pad_mask = pad_keys.expand_as(adj_mask)
        return adj_mask, pad_mask

    def forward(self, x, timesteps, adj_matrix, node_mask, **kwargs):
        del kwargs
        x = x.permute(0, 2, 1).float()
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).unsqueeze(1)
        out = self.input_emb(x) + t_emb
        adj_mask, pad_mask = self._build_masks(adj_matrix.float(), node_mask.float())
        for layer in self.layers:
            out = layer(out, adj_mask, pad_mask)
        out = self.output_head(out)
        return out.permute(0, 2, 1)


class GaussianDiffusionX0:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - alphas_bar).sqrt()
        self.posterior_variance = (
            betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        ).clamp(min=1e-20)

    def _to(self, device):
        for attr in [
            "betas",
            "alphas",
            "alphas_bar",
            "alphas_bar_prev",
            "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar",
            "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    @torch.no_grad()
    def p_sample_loop(self, model, shape, model_kwargs, device, clamp=200.0):
        self._to(device)
        model.eval()
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            ts = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x0_pred = model(x, ts, **model_kwargs)
            x0_pred = x0_pred.clamp(-clamp, clamp)

            alpha_bar = self.alphas_bar[t]
            alpha_bar_prev = self.alphas_bar_prev[t]
            beta = self.betas[t]
            alpha = self.alphas[t]

            mean = (
                alpha_bar_prev.sqrt() * beta / (1 - alpha_bar) * x0_pred
                + alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar) * x
            )

            if t > 0:
                x = mean + self.posterior_variance[t].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[
            "checkpoints/node_diffusion/model_0080000.pt",
            "checkpoints/node_diffusion/model_0090000.pt",
            "checkpoints/node_diffusion/model_0100000.pt",
            "checkpoints/node_diffusion/model_0110000.pt",
            "checkpoints/node_diffusion/model_0120000.pt",
            "checkpoints/node_diffusion/model_0130000.pt",
        ],
    )
    parser.add_argument("--data_path", default="data/processed/nodes_train.npz")
    parser.add_argument("--out_dir", default="outputs/fbbb543_check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--indices", type=int, nargs="*", default=[1639, 41905, 48598])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--clamp", type=float, default=200.0)
    return parser


def load_examples(npz_path, indices):
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"][indices].astype(np.float32)
    adj = data["adj_matrix"][indices].astype(np.float32)
    mask = data["node_mask"][indices].astype(np.float32)
    prompts = data["prompts"][indices] if "prompts" in data else np.array([""] * len(indices), dtype=object)
    return coords, adj, mask, prompts


def compute_metrics(gt_coords, pred_coords, mask):
    valid = np.repeat((mask[:, None] > 0.5), gt_coords.shape[-1], axis=1)
    diff = pred_coords - gt_coords
    sq = (diff ** 2)[valid]
    abs_err = np.abs(diff)[valid]
    rmse = float(np.sqrt(np.mean(sq))) if sq.size else 0.0
    mae = float(np.mean(abs_err)) if abs_err.size else 0.0
    return rmse, mae


def render_pair(gt_coords, pred_coords, adj, mask, title, out_path):
    n = int(mask.sum())
    gt_xy = gt_coords[:n]
    pred_xy = pred_coords[:n]

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
            ax.text(coords_xy[k, 0] + 1.0, coords_xy[k, 1] + 1.0, str(k), fontsize=7)
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


def run_checkpoint(checkpoint_path, args, coords_np, adj_np, mask_np, prompts):
    device = torch.device(args.device)
    model = NodeDiffusionTransformer().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    diffusion = GaussianDiffusionX0(timesteps=args.timesteps)

    pred_list = []
    metrics = []
    with torch.no_grad():
        for local_i, sample_idx in enumerate(args.indices):
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
            rmse, mae = compute_metrics(coords_np[local_i], pred_xy, mask_np[local_i])
            metrics.append(
                {
                    "index": int(sample_idx),
                    "n_nodes": int(mask_np[local_i].sum()),
                    "rmse": rmse,
                    "mae": mae,
                    "prompt": str(prompts[local_i]),
                }
            )

    pred_np = np.stack(pred_list, axis=0)
    overall_rmse, overall_mae = compute_metrics(
        coords_np.reshape(-1, coords_np.shape[-1]),
        pred_np.reshape(-1, pred_np.shape[-1]),
        mask_np.reshape(-1),
    )
    return ckpt, pred_np, metrics, overall_rmse, overall_mae


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    coords_np, adj_np, mask_np, prompts = load_examples(args.data_path, args.indices)

    summary = {"data_path": args.data_path, "indices": args.indices, "results": []}

    for checkpoint_path in args.checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        ckpt_out_dir = os.path.join(args.out_dir, ckpt_name)
        os.makedirs(ckpt_out_dir, exist_ok=True)
        result = {"checkpoint": checkpoint_path}

        try:
            ckpt, pred_np, metrics, overall_rmse, overall_mae = run_checkpoint(
                checkpoint_path, args, coords_np, adj_np, mask_np, prompts
            )
        except Exception as exc:
            result["load_ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
            summary["results"].append(result)
            print(f"[FAIL] {checkpoint_path} -> {result['error']}")
            continue

        result["load_ok"] = True
        result["step"] = int(ckpt.get("step", -1))
        result["overall_rmse"] = overall_rmse
        result["overall_mae"] = overall_mae
        result["samples"] = metrics
        summary["results"].append(result)

        for local_i, metric in enumerate(metrics):
            title = (
                f"{ckpt_name} | idx={metric['index']} | "
                f"RMSE={metric['rmse']:.2f} | MAE={metric['mae']:.2f}"
            )
            render_pair(
                gt_coords=coords_np[local_i],
                pred_coords=pred_np[local_i],
                adj=adj_np[local_i],
                mask=mask_np[local_i],
                title=title,
                out_path=os.path.join(ckpt_out_dir, f"sample_{metric['index']:05d}.png"),
            )

        with open(os.path.join(ckpt_out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(
            f"[OK] {checkpoint_path} | step={result['step']} | "
            f"overall RMSE={overall_rmse:.2f} | overall MAE={overall_mae:.2f}"
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved summary to: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
