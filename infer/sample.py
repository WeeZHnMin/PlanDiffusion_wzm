"""
NodeDiffusion 采样推理：从数据集取样本，GT vs 生成结果对比图。

用法（本地）:
    python -m infer.sample

用法（Kaggle）:
    python -m infer.sample --ckpt /kaggle/input/... --data /kaggle/input/...
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_diffusion.model    import NodeDiffusionTransformer
from node_diffusion.diffusion import GaussianDiffusion
from node_diffusion.dataset  import NodeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default="checkpoints/node_diffusion/latest.pt")
    p.add_argument("--data",    default="data/processed/unified_dataset/layout_dataset.npz")
    p.add_argument("--vocab",   default="node_diffusion/unified_vocab/vocab_config.json")
    p.add_argument("--bpe",     default="node_diffusion/unified_vocab/bpe_tokenizer.json")
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out",     default="outputs/samples.png")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--model-channels", type=int, default=384)
    p.add_argument("--num-layers",     type=int, default=6)
    p.add_argument("--num-heads",      type=int, default=6)
    p.add_argument("--font",    default=None, help="中文字体路径（可选）")
    return p.parse_args()


def load_model(args, device):
    import json
    vocab_cfg    = json.loads(open(args.vocab, encoding="utf-8").read())
    bpe_vocab    = vocab_cfg["bpe_vocab_size"]
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bpe_vocab_size=bpe_vocab,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd   = ckpt["model"]
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"checkpoint step: {ckpt['step']}")
    return model, vocab_cfg


def p_sample_loop(model, shape, cond, diffusion, device):
    x = torch.randn(shape, device=device)
    model.eval()
    with torch.no_grad():
        for t in reversed(range(diffusion.T)):
            ts      = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps     = model(x, ts, **cond)
            ab      = diffusion.alphas_bar[t]
            ab_prev = diffusion.alphas_bar_prev[t]
            beta    = diffusion.betas[t]
            alpha   = diffusion.alphas[t]
            x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-300, 300)
            mean    = (ab_prev.sqrt() * beta / (1 - ab) * x0_pred
                       + alpha.sqrt() * (1 - ab_prev) / (1 - ab) * x)
            x = mean + (diffusion.posterior_variance[t].sqrt() * torch.randn_like(x)
                        if t > 0 else torch.zeros_like(x))
    return x


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.font and os.path.exists(args.font):
        fm.fontManager.addfont(args.font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=args.font).get_name()

    model, vocab_cfg = load_model(args, device)
    bpe_tok   = Tokenizer.from_file(args.bpe)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    diffusion._to(device)
    ds = NodeDataset(args.data)

    def decode_prompt(token_ids, prompt_len):
        ids = [int(i) for i in token_ids[:prompt_len] if int(i) < vocab_cfg["bpe_vocab_size"]]
        return bpe_tok.decode(ids)

    COORD_SCALE  = 160.0
    indices      = args.indices
    fig, axes = plt.subplots(len(indices), 2, figsize=(9, 4.5 * len(indices)))
    if len(indices) == 1:
        axes = axes[None]

    rmse_list = []
    for row, idx in enumerate(indices):
        x_gt, cond = ds[idx]
        x_gt   = x_gt.unsqueeze(0).to(device)
        cond   = {k: v.unsqueeze(0).to(device) for k, v in cond.items()}
        n      = int(cond["node_mask"].sum().item())
        prompt = decode_prompt(ds.prompt_tokens[idx], ds.prompt_lens[idx])
        adj_np = cond["adj_matrix"][0].cpu().numpy()

        pred    = p_sample_loop(model, x_gt.shape, cond, diffusion, device)
        pred    = pred * cond["node_mask"].unsqueeze(1)
        gt_xy   = x_gt[0].permute(1, 0).cpu().numpy()
        pred_xy = pred[0].permute(1, 0).cpu().numpy()

        valid = cond["node_mask"][0].cpu().numpy().astype(bool)
        rmse  = float(((pred_xy[valid] - gt_xy[valid]) ** 2).mean() ** 0.5) * COORD_SCALE
        rmse_list.append(rmse)

        for ax, xy, title in zip(axes[row], [gt_xy, pred_xy],
                                  ["GT", "Sampled  RMSE=" + str(round(rmse, 1))]):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_np[i, j] > 0.5:
                        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                                "steelblue", lw=1.2, alpha=0.7)
            ax.scatter(xy[:n, 0], xy[:n, 1], c=np.arange(n), cmap="tab20", s=45, zorder=3)
            for k in range(n):
                ax.text(xy[k, 0], xy[k, 1], str(k), fontsize=6, ha="center", va="bottom")
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.grid(alpha=0.2)
            if ax is axes[row][0]:
                ax.set_xlabel(prompt, fontsize=7)

    ckpt_step = torch.load(args.ckpt, map_location="cpu").get("step", "?")
    mean_rmse = sum(rmse_list) / len(rmse_list)
    fig.suptitle("Step " + str(ckpt_step) + "  |  mean RMSE = " + str(round(mean_rmse, 1)),
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("saved:", args.out)
    print("per-sample RMSE:", [round(r, 1) for r in rmse_list])


if __name__ == "__main__":
    main()
