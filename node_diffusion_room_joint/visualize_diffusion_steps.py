"""
可视化单条样本的逆扩散过程：t=1000 → 0，每 200 步截一张快照。

输出：1行 × 7列图
  Col 1~6  t=1000,800,600,400,200,0 时刻的节点坐标图
  Col 7    GT 真实坐标图

用法：
    python -m node_diffusion_room_joint.visualize_diffusion_steps \\
        --ckpt  checkpoints/node_diffusion_room_joint/t700_centroid/best.pt \\
        --data  data/jsonl/final_graph_dataset_v3_4k5.jsonl \\
        --index 0 \\
        --out   outputs/diffusion_steps.png
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
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer, _assign_room_membership_single, MAX_ROOMS
from .diffusion import GaussianDiffusion

MAX_NODES    = 40
MAX_TEXT_LEN = 192
SNAP_T       = [1000, 800, 600, 400, 200, 0]   # 要截图的时间步


def _ax_nodes(ax, coords_np, adj_np, mask_np, title):
    """画节点 + 边，coords_np: [N,2]"""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#F8F8F8')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC'); sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=9, pad=3)

    valid = np.where(mask_np > 0.5)[0]
    if len(valid) == 0:
        return
    pts = coords_np[valid]
    c   = pts.mean(0)
    s   = max(np.abs(pts - c).max(), 1e-3)
    d   = (pts - c) / s * 2.5          # 归一化到 [-2.5, 2.5]

    # 画边
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj_np[ni, nj] > 0.5:
                ax.plot([d[ii, 0], d[jj, 0]], [d[ii, 1], d[jj, 1]],
                        color='#AAAAAA', lw=0.8, alpha=0.7, zorder=1)
    # 画节点
    ax.scatter(d[:, 0], d[:, 1], s=28, c='#4E8CC2',
               edgecolors='#334455', linewidths=0.5, zorder=3)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')


@torch.no_grad()
def sample_with_snapshots(model, diffusion, cond, device, ddim_steps=200):
    """
    DDIM 采样，每隔 ddim_steps//5 步截一次 x，返回对应 SNAP_T 的 coords 列表。
    返回 list of np.ndarray [N_NODES, 2]，顺序对应 SNAP_T=[1000,800,600,400,200,0]
    """
    diffusion._to(device)
    ts = torch.linspace(0, diffusion.T - 1, ddim_steps).long().flip(0).tolist()
    # ts[0] ≈ 999, ts[-1] = 0

    B = 1
    x = torch.randn(B, 2, MAX_NODES, device=device)

    # 预计算文本
    text_feat, text_mask = model.encode_text(
        cond['prompt_tokens'], cond.get('prompt_mask'))
    extra = {k: v for k, v in cond.items()
             if k not in ('prompt_tokens', 'prompt_mask')}
    extra['text_feat'] = text_feat
    extra['text_mask'] = text_mask

    # 计算每个 snap_t 对应最近的 ddim step index
    snap_indices = {}
    for snap in SNAP_T:
        if snap == 1000:
            snap_indices[snap] = -1          # 采样前（纯噪声）
        elif snap == 0:
            snap_indices[snap] = len(ts) - 1  # 最后一步
        else:
            # 找最近的 ts 值
            diffs = [abs(t - snap) for t in ts]
            snap_indices[snap] = int(np.argmin(diffs))

    snapshots = {}

    def _to_np(x_tensor):
        return x_tensor.squeeze(0).permute(1, 0).cpu().numpy()  # [N_NODES, 2]

    # t=1000：纯噪声
    snapshots[1000] = _to_np(x)

    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps  = model(x, t_tensor, **extra)
        ab_t = diffusion.alphas_bar[t]
        x0   = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-3)
        if i + 1 < len(ts):
            ab_prev = diffusion.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0

        # 检查是否是截图步
        for snap, sidx in snap_indices.items():
            if snap != 1000 and i == sidx:
                snapshots[snap] = _to_np(x)

    # 确保所有 snap 都有值
    for snap in SNAP_T:
        if snap not in snapshots:
            snapshots[snap] = _to_np(x)

    return [snapshots[s] for s in SNAP_T]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='checkpoints/node_diffusion_room_joint/t700_centroid/best.pt')
    p.add_argument('--data',  default='data/jsonl/final_graph_dataset_v3_4k5.jsonl')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--seed',  type=int, default=42)
    p.add_argument('--ddim',  type=int, default=200)
    p.add_argument('--gpu',   type=int, default=None)
    p.add_argument('--out',   default='outputs/diffusion_steps.png')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ── 读取一条记录 ─────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    with open(args.data, encoding='utf-8') as f:
        lines = f.readlines()
    rec = json.loads(lines[args.index])
    n   = int(rec['n_nodes'])
    print(f'index={args.index}  n_nodes={n}  text={rec["prompt"][:80]}')

    adj_raw = np.array(rec['adj_matrix'], dtype=np.float32)[:n, :n]
    np.fill_diagonal(adj_raw, 0)
    adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    adj_pad[:n, :n] = adj_raw

    mask_np = np.zeros(MAX_NODES, dtype=np.float32); mask_np[:n] = 1.0

    membership = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
    membership[:n] = _assign_room_membership_single(adj_raw.astype(bool), n)

    combo_ids = np.zeros(MAX_NODES, dtype=np.int32)
    raw_combos = rec.get('node_combo_ids', [])
    combo_ids[:min(n, len(raw_combos))] = [int(c) for c in raw_combos[:n]]

    gt_coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
    for k, xy in enumerate(rec.get('node_coords', [])[:n]):
        gt_coords[k] = xy

    enc  = tokenizer(rec['prompt'], add_special_tokens=True,
                     max_length=MAX_TEXT_LEN, padding='max_length', truncation=True)
    ptok = np.array(enc['input_ids'],      dtype=np.int64)
    pmsk = np.array(enc['attention_mask'], dtype=np.float32)

    cond = {
        'node_mask':       torch.from_numpy(mask_np[None]).to(device),
        'room_membership': torch.from_numpy(membership[None]).to(device),
        'adj_matrix':      torch.from_numpy(adj_pad[None]).to(device),
        'node_combo_ids':  torch.from_numpy(combo_ids[None]).to(device),
        'prompt_tokens':   torch.from_numpy(ptok[None]).to(device),
        'prompt_mask':     torch.from_numpy(pmsk[None]).to(device),
    }

    # ── 加载模型 ─────────────────────────────────────────────────────────────────
    print(f'加载模型: {args.ckpt}')
    model = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=False)
    model.eval()
    print(f'  step={ckpt.get("step","?")}')

    diffusion = GaussianDiffusion(timesteps=1000)

    # ── 采样 + 截图 ───────────────────────────────────────────────────────────────
    print(f'DDIM {args.ddim} 步采样，截取 t={SNAP_T}...')
    snapshots = sample_with_snapshots(model, diffusion, cond, device, args.ddim)

    # ── 绘图 ─────────────────────────────────────────────────────────────────────
    n_cols = len(SNAP_T) + 1   # 6个快照 + 1个GT
    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 3.2),
                             constrained_layout=True)

    for col, (snap_t, coords) in enumerate(zip(SNAP_T, snapshots)):
        _ax_nodes(axes[col], coords, adj_pad, mask_np, f't = {snap_t}')

    _ax_nodes(axes[-1], gt_coords, adj_pad, mask_np, 'GT')

    fig.suptitle(rec['prompt'][:100], fontsize=8, y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f'已保存: {args.out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
