"""
推理脚本：从 checkpoint 加载模型，对数据集样本执行 DDPM 逆向采样，
同时去噪坐标和类型嵌入，输出可视化图和指标。

用法（项目根目录执行）：
    python -m node_diffusion.infer \
        --checkpoint checkpoints/node_diffusion/latest.pt \
        --data_path  data/processed/graph_dataset.npz \
        --num_samples 8 \
        --out_dir outputs/infer
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ── 复用 kaggle notebook 中的模型 / 扩散类定义 ────────────────────────────────
from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .postprocess import snap_nodes_to_walls


# ── 参数 ──────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--data_path',     default='data/processed/node_diffusion/graph_dataset.npz')
    p.add_argument('--vocab',         default='node_diffusion/unified_vocab_wp/vocab_config.json')
    p.add_argument('--out_dir',       default='outputs/infer')
    p.add_argument('--num_samples',   type=int,   default=8)
    p.add_argument('--indices',       type=int,   nargs='*', default=None)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model_channels',type=int,   default=384)
    p.add_argument('--num_layers',    type=int,   default=6)
    p.add_argument('--num_heads',     type=int,   default=6)
    p.add_argument('--timesteps',     type=int,   default=1000)
    p.add_argument('--ddim_steps',    type=int,   default=0,
                   help='若 >0 则使用 DDIM 加速采样，否则使用完整 DDPM 1000步')
    p.add_argument('--snap_threshold', type=float, default=8.0,
                   help='将靠近墙的节点吸附到墙上的距离阈值（像素），0=不做吸附')
    return p


# ── DDPM 逆向采样循环 ─────────────────────────────────────────────────────────
@torch.no_grad()
def p_sample_loop(model, diffusion, shape_coord, model_kwargs, device, ddim_steps=0):
    """
    坐标逆向去噪。

    shape_coord : (B, 2, N)
    返回        : x0_coord (B, 2, N)
    """
    diff = diffusion
    diff._to(device)

    B = shape_coord[0]
    x_t = torch.randn(shape_coord, device=device)

    if ddim_steps > 0:
        step_indices = torch.linspace(0, diff.T - 1, ddim_steps + 1).long()
        timestep_seq = step_indices.flip(0)[:-1].tolist()
    else:
        timestep_seq = list(reversed(range(diff.T)))

    for t_val in timestep_seq:
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

        eps_coord = model(x_t, t_batch, **model_kwargs).float()

        alpha_t       = diff.alphas[t_val].to(device)
        alpha_bar_t   = diff.alphas_bar[t_val].to(device)
        alpha_bar_tm1 = diff.alphas_bar_prev[t_val].to(device)
        beta_t        = diff.betas[t_val].to(device)
        post_var      = diff.posterior_variance[t_val].to(device)

        x0_pred  = (x_t - (1 - alpha_bar_t).sqrt() * eps_coord) / alpha_bar_t.sqrt()
        coef1    = (alpha_bar_tm1.sqrt() * beta_t) / (1 - alpha_bar_t)
        coef2    = ((1 - alpha_bar_tm1) * alpha_t.sqrt()) / (1 - alpha_bar_t)
        mu_coord = coef1 * x0_pred + coef2 * x_t

        if t_val > 0:
            x_t = mu_coord + post_var.sqrt() * torch.randn_like(x_t)
        else:
            x_t = mu_coord

    return x_t


# ── 可视化 ────────────────────────────────────────────────────────────────────
COMBO_COLORS = [
    '#A9CDE8','#7BB9E0','#4D9FD5','#1A6FA6',
    '#A8D5A2','#6EBC68','#3A9E35','#1E7B19',
    '#F5D48B','#F0BE45','#E89E10','#B87A00',
    '#D4A8D4','#B87BB8','#8F4F8F','#6B2D6B',
    '#F5A8A8','#E86060','#D02020','#A00000',
    '#BBBBBB','#999999','#777777','#555555',
    '#FFDDB0','#FFB86C','#E88C30','#C06010',
    '#B0E0E0','#70C0C0','#30A0A0','#008080',
]

def type_color(type_id):
    if type_id <= 0 or type_id > 32:
        return '#DDDDDD'
    return COMBO_COLORS[(type_id - 1) % len(COMBO_COLORS)]


def render_result(gt_coords, pred_coords, gt_types, pred_types,
                  adj, mask, title, out_path):
    n = int(mask.sum())
    gt_xy   = gt_coords[:n]
    pred_xy = pred_coords[:n]

    both = np.concatenate([gt_xy, pred_xy], axis=0)
    pad  = max(10.0, 0.1 * (np.ptp(both, axis=0) + 1e-6).max())
    xlim = (both[:,0].min()-pad, both[:,0].max()+pad)
    ylim = (both[:,1].min()-pad, both[:,1].max()+pad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, xy, types, name in zip(
        axes,
        [gt_xy, pred_xy],
        [gt_types[:n], pred_types[:n]],
        ['Ground Truth', 'Generated']
    ):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] > 0.5:
                    ax.plot([xy[i,0], xy[j,0]], [xy[i,1], xy[j,1]],
                            color='#AAAAAA', lw=0.8, alpha=0.6, zorder=1)
        for k in range(n):
            c = type_color(int(types[k]))
            ax.scatter(xy[k,0], xy[k,1], color=c, s=60, zorder=3,
                       edgecolors='#555555', linewidths=0.5)
            ax.text(xy[k,0]+1.5, xy[k,1]+1.5, str(k), fontsize=6, zorder=4)
        ax.set_title(name, fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.invert_yaxis()
        ax.grid(alpha=0.15)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # 词表大小
    vocab_cfg  = json.loads(open(args.vocab, encoding='utf-8').read())
    bpe_vocab  = vocab_cfg['wp_vocab_size']

    # 加载模型
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bpe_vocab_size=bpe_vocab,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model']
    # 兼容 DataParallel 保存的权重
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'loaded checkpoint step={ckpt.get("step","?")}')

    # 扩散对象
    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    diffusion._to(device)

    # 加载数据
    data = np.load(args.data_path, allow_pickle=True)
    total = len(data['node_coords'])
    if args.indices:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(total, size=min(args.num_samples, total),
                                    replace=False).tolist())

    print(f'sample indices: {indices}')

    all_metrics = []

    for idx in indices:
        # 取单条样本
        coords_raw   = data['node_coords'][idx].astype(np.float32)   # [40, 2]
        adj_np       = data['adj_matrix'][idx].astype(np.float32)    # [40, 40]
        mask_np      = data['node_mask'][idx].astype(np.float32)     # [40]
        types_np     = data['node_combo_ids'][idx].astype(np.int64)  # [40]
        prompt_np    = data['prompt_tokens'][idx].astype(np.int64)   # [T]
        prompt_len   = int(data['prompt_lens'][idx])
        T_tok        = len(prompt_np)
        prompt_mask_np = np.zeros(T_tok, dtype=np.float32)
        prompt_mask_np[:prompt_len] = 1.0

        n_valid = int(mask_np.sum())

        # 转 tensor
        x_gt   = torch.from_numpy(coords_raw.T[None]).to(device)       # [1,2,40]
        cond   = {
            'adj_matrix':    torch.from_numpy(adj_np[None]).to(device),
            'node_mask':     torch.from_numpy(mask_np[None]).to(device),
            'node_types':    torch.from_numpy(types_np[None]).to(device),
            'prompt_tokens': torch.from_numpy(prompt_np[None]).to(device),
            'prompt_mask':   torch.from_numpy(prompt_mask_np[None]).to(device),
        }

        shape_coord = (1, 2, 40)

        # 逆向采样（仅坐标）
        x0_coord = p_sample_loop(
            model=model,
            diffusion=diffusion,
            shape_coord=shape_coord,
            model_kwargs=cond,
            device=device,
            ddim_steps=args.ddim_steps,
        )

        # 转 numpy
        pred_xy = x0_coord[0].permute(1, 0).cpu().numpy()   # [40, 2]
        gt_xy   = coords_raw
        gt_t    = types_np

        # 吸附后处理：将靠近墙的节点嵌入墙上
        if args.snap_threshold > 0:
            pred_xy, adj_np, _ = snap_nodes_to_walls(
                pred_xy, adj_np, mask_np, threshold=args.snap_threshold)

        # 指标（只算坐标 RMSE）
        valid_mask = mask_np > 0.5
        diff_xy    = (pred_xy - gt_xy)[valid_mask]
        rmse       = float(np.sqrt(np.mean(diff_xy**2)))

        all_metrics.append({'index': idx, 'n_nodes': n_valid, 'rmse': round(rmse, 3)})
        print(f'idx={idx:5d} | n={n_valid:2d} | RMSE={rmse:.2f}')

        # 可视化（节点颜色用 GT 类型）
        title = f'idx={idx}  n={n_valid}  RMSE={rmse:.2f}'
        render_result(
            gt_coords=gt_xy[:n_valid],
            pred_coords=pred_xy[:n_valid],
            gt_types=gt_t[:n_valid],
            pred_types=gt_t[:n_valid],
            adj=adj_np[:n_valid, :n_valid],
            mask=mask_np[:n_valid],
            title=title,
            out_path=os.path.join(args.out_dir, f'sample_{idx:05d}.png'),
        )

    # 汇总
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    summary  = {
        'checkpoint': args.checkpoint,
        'step': int(ckpt.get('step', -1)),
        'num_samples': len(indices),
        'avg_rmse': round(float(avg_rmse), 3),
        'samples': all_metrics,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\navg RMSE={avg_rmse:.2f}')
    print(f'结果保存至: {args.out_dir}')


if __name__ == '__main__':
    main()
