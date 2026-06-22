"""
评估 + 可视化脚本：从训练集取若干样本，跑完整 DDPM 逆向采样，绘制生成结果。

用法：
  python -m node_diffusion_x0.eval \
      --checkpoint checkpoints/node_diffusion/20260622_180612/latest.pt \
      --data_path  data/processed/node_diffusion_cross_att/graph_dataset_6k.npz \
      --bert       models/bert-base-uncased \
      --n_samples  6 \
      --out_dir    eval_output
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from .dataset import TypeDataset
from .diffusion import GaussianDiffusion
from .model import NodeDiffusionTransformer

# 32 种房间类型的名称（索引 1-32，0=padding）
TYPE_NAMES = [
    '', 'LivingRoom', 'MasterRoom', 'Kitchen', 'Bathroom', 'DiningRoom',
    'ChildRoom', 'StudyRoom', 'SecondRoom', 'GuestRoom', 'Balcony',
    'Entrance', 'Storage', 'Wall-in', 'External', 'ExteriorWall',
    'FrontDoor', 'InteriorWall', 'InteriorDoor', 'Corridor', 'Terrace',
    'Hallway', 'Bedroom', 'Livingroom', 'Garage', 'Laundry', 'Utility',
    'Basement', 'Gym', 'Office', 'Library', 'Media', 'Other',
]

CMAP = plt.get_cmap('tab20')


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_path',  default='data/processed/node_diffusion_cross_att/graph_dataset_6k.npz')
    p.add_argument('--bert',       default='models/bert-base-uncased')
    p.add_argument('--n_samples',  type=int, default=6)
    p.add_argument('--timesteps',  type=int, default=1000)
    p.add_argument('--gpu',        type=int, default=None)
    p.add_argument('--out_dir',    default='eval_output')
    p.add_argument('--model_channels', type=int, default=384)
    p.add_argument('--num_layers',     type=int, default=6)
    p.add_argument('--num_heads',      type=int, default=6)
    return p


@torch.no_grad()
def sample(model, diffusion, cond, timesteps, device):
    """DDPM 逆向采样，返回生成坐标 [2, N] 和类型 logits [N, 33]。"""
    adj   = cond['adj_matrix'].unsqueeze(0).to(device)   # [1, N, N]
    mask  = cond['node_mask'].unsqueeze(0).to(device)     # [1, N]
    ptok  = cond['prompt_tokens'].unsqueeze(0).to(device)
    pmask = cond['prompt_mask'].unsqueeze(0).to(device)

    N  = mask.shape[1]
    xt = torch.randn(1, 2, N, device=device)

    mk = {
        'adj_matrix':    adj,
        'node_mask':     mask,
        'prompt_tokens': ptok,
        'prompt_mask':   pmask,
    }

    for t_scalar in range(timesteps - 1, -1, -1):
        xt, _, type_logits = diffusion.p_sample(model, xt, int(t_scalar), mk)

    coords = xt[0].cpu()            # [2, N]
    logits = type_logits[0].cpu()   # [N, 33]
    return coords, logits


def draw_sample(ax, coords, logits, adj_matrix, node_mask, gt_coords, gt_types, title=''):
    """在 ax 上绘制生成结果（左）和 GT（右）对比。"""
    mask = node_mask.bool().numpy()
    adj  = adj_matrix.numpy()

    gen_xy  = coords.numpy().T    # [N, 2]
    gt_xy   = gt_coords.numpy().T # [N, 2]
    pred_t  = logits[:, 1:].argmax(dim=-1).numpy() + 1  # [N], 1-32
    gt_t    = gt_types.numpy()

    for data_xy, types, subplot_title, col_idx in [
        (gt_xy,  gt_t,   f'GT         {title}', 0),
        (gen_xy, pred_t, f'Generated  {title}', 1),
    ]:
        cur_ax = ax[col_idx]
        cur_ax.set_title(subplot_title, fontsize=7)
        cur_ax.set_aspect('auto')
        cur_ax.axis('on')

        # 画边
        for i in range(len(mask)):
            if not mask[i]:
                continue
            for j in range(i + 1, len(mask)):
                if mask[j] and adj[i, j] > 0:
                    cur_ax.plot(
                        [data_xy[i, 0], data_xy[j, 0]],
                        [data_xy[i, 1], data_xy[j, 1]],
                        'k-', lw=0.5, alpha=0.4,
                    )

        # 画节点
        for i in range(len(mask)):
            if not mask[i]:
                continue
            t   = int(types[i])
            col = CMAP((t - 1) % 20 / 20)
            cur_ax.scatter(data_xy[i, 0], data_xy[i, 1],
                           c=[col], s=60, zorder=3, edgecolors='white', linewidths=0.5)
            lbl = TYPE_NAMES[t] if 0 < t < len(TYPE_NAMES) else str(t)
            cur_ax.annotate(lbl, (data_xy[i, 0], data_xy[i, 1]),
                            fontsize=4, ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points')


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载数据 ──
    ds = TypeDataset(args.data_path)
    indices = np.random.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    # ── 加载模型 ──
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    ).to(device)

    ckpt    = torch.load(args.checkpoint, map_location=device)
    raw_sd  = ckpt['model']
    if any(k.startswith('module.') for k in raw_sd):
        raw_sd = {k[7:]: v for k, v in raw_sd.items()}
    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    if missing:    print(f'missing: {missing}')
    if unexpected: print(f'unexpected: {unexpected}')
    step = ckpt.get('step', '?')
    print(f'loaded checkpoint (step={step})')
    model.eval()

    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    # ── 逐样本推理 + 绘图 ──
    ncols = 2  # 生成 | GT
    nrows = len(indices)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 3 * nrows))
    if nrows == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        x, cond = ds[idx]
        gt_coords = x                          # [2, N]
        gt_types  = cond['node_types']         # [N]
        adj       = cond['adj_matrix']         # [N, N]
        mask      = cond['node_mask']          # [N]

        print(f'[{row+1}/{nrows}] sample #{idx} ...')
        gen_coords, gen_logits = sample(
            model, diffusion, cond,
            timesteps=args.timesteps,
            device=device,
        )

        draw_sample(
            axes[row],
            gen_coords, gen_logits,
            adj, mask,
            gt_coords, gt_types,
            title=f'#{idx}',
        )

    plt.tight_layout(pad=1.0)
    out_path = os.path.join(args.out_dir, f'eval_step{step}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out_path}')


if __name__ == '__main__':
    main()
