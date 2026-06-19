"""
对 gen_adj_test.npz（θ₁ 生成的图结构）运行坐标扩散推理并可视化。

用法（项目根目录）：
    python -m node_diffusion_cross_att.infer_gen_adj \
        --ckpt  checkpoints/node_diffusion_cross_att/latest.pt \
        --bert  models/bert-base-uncased \
        --data  data/processed/node_diffusion_cross_att/gen_adj_test.npz \
        --n     8 \
        --out_dir outputs/infer_gen_adj
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .postprocess import snap_nodes_to_walls


NODE_COLOR = "#4E8CC2"


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='checkpoints/node_diffusion_cross_att/latest.pt',
                   help='NodeDiffusionTransformer checkpoint 路径')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    p.add_argument('--data',  default='data/processed/node_diffusion_cross_att/gen_adj_test.npz')
    p.add_argument('--n',     type=int, default=8,   help='随机抽取样本数')
    p.add_argument('--seed',  type=int, default=42)
    p.add_argument('--ddim_steps', type=int, default=0,
                   help='DDIM 步数（0 = 完整 DDPM 1000 步）')
    p.add_argument('--snap_threshold', type=float, default=8.0,
                   help='吸附阈值（像素），0=不做吸附')
    p.add_argument('--out_dir', default='outputs/infer_gen_adj')
    return p.parse_args()


# ── 采样 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_coords(model, diffusion, cond, device, ddim_steps):
    """返回 [1, 2, 40] 坐标 tensor。"""
    diff = diffusion
    diff._to(device)

    adj  = cond['adj_matrix'].to(device)
    mask = cond['node_mask'].to(device)
    ptok = cond['prompt_tokens'].to(device)
    pmsk = cond['prompt_mask'].to(device).long()

    # BERT 特征预计算（只跑一次）
    text_hidden = model.bert(input_ids=ptok, attention_mask=pmsk).last_hidden_state
    text_feat   = model.text_proj(text_hidden)
    text_mask   = (1 - pmsk.float()).unsqueeze(1)

    def fwd(x, t_val):
        from node_diffusion_cross_att.model import timestep_embedding
        tb    = torch.full((1,), t_val, device=device, dtype=torch.long)
        x_in  = x.permute(0, 2, 1).float()
        t_emb = model.time_embed(timestep_embedding(tb, model.model_channels)).unsqueeze(1)
        h     = model.input_emb(x_in) + t_emb
        am    = model._build_adj_mask(adj.float(), mask.float())
        for layer in model.layers:
            h = layer(h, am, text_feat, text_mask)
        return model.coord_head(h).permute(0, 2, 1).float()

    x = torch.randn(1, 2, 40, device=device)

    if ddim_steps > 0:
        step_seq = np.linspace(diff.T - 1, 0, ddim_steps, dtype=int).tolist()
        for i, t in enumerate(step_seq):
            eps  = fwd(x, t)
            ab   = diff.alphas_bar[t]
            x0   = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
            if i + 1 < len(step_seq):
                ab_prev = diff.alphas_bar[step_seq[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
    else:
        for t in reversed(range(diff.T)):
            eps = fwd(x, t)
            ab  = diff.alphas_bar[t]
            ap  = diff.alphas_bar_prev[t]
            a   = diff.alphas[t]
            b   = diff.betas[t]
            x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
            mu  = (ap.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
            x   = mu + diff.posterior_variance[t].sqrt() * torch.randn_like(x) if t > 0 else mu

    return x


# ── 可视化单张 ─────────────────────────────────────────────────────────────────

def draw_single(ax, coords, adj, valid_mask, title=''):
    """coords [40, 2], adj [40,40], valid_mask [40] bool"""
    valid = np.where(valid_mask)[0]
    pts   = coords[valid]

    # 归一化到显示空间
    if len(pts) > 0:
        c = pts.mean(0)
        s = max(np.abs(pts - c).max(), 1.0)
        pts_d  = (pts - c) / s * 2.5
        all_d  = (coords - c) / s * 2.5
    else:
        pts_d = pts
        all_d = coords

    # 边
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            ni, nj = valid[ii], valid[jj]
            if adj[ni, nj] > 0.5:
                ax.plot([pts_d[ii, 0], pts_d[jj, 0]],
                        [pts_d[ii, 1], pts_d[jj, 1]],
                        color='#AAAAAA', lw=0.8, alpha=0.7, zorder=1)

    # 节点
    for i in range(len(valid)):
        ax.add_patch(plt.Circle((pts_d[i, 0], pts_d[i, 1]), 0.10,
                                color=NODE_COLOR, ec='#333333', lw=0.5, zorder=3))

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, fontsize=7, pad=2)
    for sp in ax.spines.values():
        sp.set_edgecolor('#DDDDDD'); sp.set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 加载模型
    model = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'模型加载完成  step={ckpt.get("step", "?")}')

    diffusion = GaussianDiffusion(timesteps=1000)

    # 加载数据
    data      = np.load(args.data, allow_pickle=True)
    valid_idx = np.where(data['valid'])[0]
    print(f'gen_adj_test: 共 {len(data["valid"])} 条，有效 {len(valid_idx)} 条')

    rng     = np.random.default_rng(args.seed)
    chosen  = rng.choice(valid_idx, size=min(args.n, len(valid_idx)), replace=False)
    chosen  = sorted(chosen.tolist())
    print(f'抽取索引: {chosen}')

    # 逐条推理 + 可视化
    cols = min(4, args.n)
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.2, rows * 2.4),
                             squeeze=False)

    for plot_i, idx in enumerate(chosen):
        adj_np  = data['adj_matrix'][idx].astype('float32')   # [40, 40]
        mask_np = data['node_mask'][idx].astype('float32')    # [40]
        ptok_np = data['prompt_tokens'][idx].astype('int64')  # [T]
        pmsk_np = data['prompt_mask'][idx].astype('float32')  # [T]
        n_nodes = int(data['n_nodes'][idx])

        cond = {
            'adj_matrix':    torch.from_numpy(adj_np ).unsqueeze(0),
            'node_mask':     torch.from_numpy(mask_np).unsqueeze(0),
            'prompt_tokens': torch.from_numpy(ptok_np).unsqueeze(0),
            'prompt_mask':   torch.from_numpy(pmsk_np).unsqueeze(0),
        }

        mode = f'DDIM-{args.ddim_steps}' if args.ddim_steps > 0 else 'DDPM-1000'
        print(f'  [{plot_i+1}/{len(chosen)}] idx={idx}  n_nodes={n_nodes}  {mode}')

        x0 = sample_coords(model, diffusion, cond, device, args.ddim_steps)
        coords = x0[0].permute(1, 0).cpu().numpy()  # [40, 2]

        valid_mask = mask_np > 0.5
        if args.snap_threshold > 0:
            coords, adj_np, snapped = snap_nodes_to_walls(
                coords, adj_np, valid_mask, threshold=args.snap_threshold)
            if snapped:
                print(f'    snap: {len(snapped)} 节点吸附')

        r, c = divmod(plot_i, cols)
        draw_single(axes[r][c], coords, adj_np, valid_mask,
                    title=f'idx={idx}  n={n_nodes}')

    # 隐藏多余格子
    for k in range(len(chosen), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].set_visible(False)

    mode_str = f'DDIM-{args.ddim_steps}' if args.ddim_steps > 0 else 'DDPM-1000'
    fig.suptitle(f'Node Coord Diffusion on gen_adj_test  [{mode_str}]',
                 fontsize=9, y=1.01)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, 'results.png')
    out_pdf = os.path.join(args.out_dir, 'results.pdf')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'\n保存: {out_png}')
    print(f'保存: {out_pdf}')


if __name__ == '__main__':
    main()
