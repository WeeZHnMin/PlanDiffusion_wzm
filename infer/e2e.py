"""
端到端推理：文本 → Stage2图结构 → NodeDiffusion坐标 → 可视化。

用法（本地）:
    python -m infer.e2e

用法（Kaggle）:
    python -m infer.e2e \
        --stage2-ckpt /kaggle/input/... \
        --diff-ckpt   /kaggle/input/... \
        --prompts "客厅在中央" "两间卧室在左侧"
"""

import argparse
import ast
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_diffusion.model     import NodeDiffusionTransformer
from node_diffusion.diffusion import GaussianDiffusion
from infer.stage2 import (
    build_combo_maps, load_stage2, constrained_generate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-ckpt",  default="checkpoints/stage2_text_graph/best.pt")
    p.add_argument("--diff-ckpt",    default="checkpoints/node_diffusion/latest.pt")
    p.add_argument("--vocab",        default="node_diffusion/unified_vocab/vocab_config.json")
    p.add_argument("--bpe",          default="node_diffusion/unified_vocab/bpe_tokenizer.json")
    p.add_argument("--combo-vocab",  default="data/processed/type_combo_vocab_old.json")
    p.add_argument("--prompts",      nargs="+", default=[
        "客厅位于中央，连接卧室、厨房和浴室。",
        "两间卧室在左侧，浴室居中，厨房在右侧与走廊相连。",
        "走廊居中，左侧连接三间卧室和一间浴室，右侧连接客厅和厨房。",
        "客厅在左上方，厨房在右侧，两间浴室分别位于左下和右下，卧室在中央。",
        "入口连接走廊，走廊通向客厅、两间卧室和浴室，厨房与客厅相邻。",
    ])
    p.add_argument("--out",          default="outputs/e2e_results.png")
    p.add_argument("--timesteps",    type=int, default=1000)
    p.add_argument("--model-channels", type=int, default=384)
    p.add_argument("--num-layers",     type=int, default=6)
    p.add_argument("--num-heads",      type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--font",         default=None)
    return p.parse_args()


def load_diff_model(args, vocab_cfg, device):
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bpe_vocab_size=vocab_cfg["bpe_vocab_size"],
    ).to(device)
    ckpt = torch.load(args.diff_ckpt, map_location=device)
    sd   = ckpt["model"]
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"NodeDiffusion loaded  step: {ckpt['step']}")
    return model


def graph_to_tensors(n_nodes, adj_dict, prompt_text, bpe_tok, vocab_cfg, device):
    MAX_NODES = vocab_cfg["MAX_NODES"]
    adj_t  = torch.zeros(1, MAX_NODES, MAX_NODES, device=device)
    mask_t = torch.zeros(1, MAX_NODES, device=device)
    for i, nbrs in adj_dict.items():
        for j in nbrs:
            adj_t[0, i, j] = 1.0
            adj_t[0, j, i] = 1.0
    mask_t[0, :n_nodes] = 1.0

    ids = bpe_tok.encode(prompt_text).ids[:128]
    pt  = torch.zeros(1, 128, dtype=torch.long, device=device)
    pm  = torch.zeros(1, 128, device=device)
    pt[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    pm[0, :len(ids)] = 1.0

    return {"adj_matrix": adj_t, "node_mask": mask_t,
            "prompt_tokens": pt, "prompt_mask": pm}


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


def draw_floor_plan(ax, xy, n, adj_np, node_types, combo_label, combo_color, title):
    for i in range(n):
        for j in range(i + 1, n):
            if adj_np[i, j] > 0.5:
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                        color="#cccccc", lw=1.5, zorder=1)
    for k in range(n):
        cid   = node_types.get(k, 7)
        color = combo_color.get(cid, "#cccccc")
        label = combo_label.get(cid, str(cid))
        ax.scatter(xy[k, 0], xy[k, 1], c=color, s=160, zorder=3,
                   edgecolors="white", linewidths=1.0)
        ax.text(xy[k, 0], xy[k, 1], label, fontsize=5,
                ha="center", va="center", color="white",
                fontweight="bold", zorder=4)
    seen    = set(node_types.get(i, 7) for i in range(n))
    handles = [mpatches.Patch(color=combo_color[c], label=combo_label[c])
               for c in sorted(seen)]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(alpha=0.15)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.font and os.path.exists(args.font):
        fm.fontManager.addfont(args.font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=args.font).get_name()

    vocab_cfg          = json.loads(open(args.vocab, encoding="utf-8").read())
    bpe_tok            = Tokenizer.from_file(args.bpe)
    combo_bases, combo_label = build_combo_maps(args.combo_vocab)

    BASE_COLOR = {1:"#4e9af1", 2:"#6dbf67", 3:"#f0a500",
                  4:"#e05c5c", 5:"#999999", 6:"#9b59b6", 7:"#a07850"}
    combo_color = {cid: BASE_COLOR.get(bases[0], "#cccccc")
                   for cid, bases in combo_bases.items()}

    s2_model   = load_stage2(args.stage2_ckpt, vocab_cfg, device)
    diff_model = load_diff_model(args, vocab_cfg, device)
    diffusion  = GaussianDiffusion(timesteps=args.timesteps)
    diffusion._to(device)

    MAX_NODES = vocab_cfg["MAX_NODES"]
    fig, axes = plt.subplots(len(args.prompts), 1,
                             figsize=(6, 5 * len(args.prompts)))
    if len(args.prompts) == 1:
        axes = [axes]

    for row, prompt in enumerate(args.prompts):
        print("\n" + "=" * 60)
        print("提示词:", prompt)

        n_nodes, adj, node_types, toks = constrained_generate(
            prompt, s2_model, bpe_tok, vocab_cfg, device, args.max_new_tokens)
        print(f"节点数: {n_nodes}  token数: {len(toks)}")

        if n_nodes == 0:
            axes[row].text(0.5, 0.5, "Stage2 未生成节点",
                           ha="center", va="center",
                           transform=axes[row].transAxes)
            continue

        cond   = graph_to_tensors(n_nodes, adj, prompt, bpe_tok, vocab_cfg, device)
        pred   = p_sample_loop(diff_model, (1, 2, MAX_NODES), cond, diffusion, device)
        pred   = pred * cond["node_mask"].unsqueeze(1)
        xy     = pred[0].permute(1, 0).cpu().numpy()
        adj_np = cond["adj_matrix"][0].cpu().numpy()

        title  = prompt[:35] + ("..." if len(prompt) > 35 else "")
        draw_floor_plan(axes[row], xy, n_nodes, adj_np,
                        node_types, combo_label, combo_color, title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
