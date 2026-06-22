"""
节点坐标扩散模块（θ₂）Coord-RMSE 评估脚本。

条件：gen_adj_test.npz（θ₁ 生成的邻接矩阵 + BERT prompt）
GT坐标：text_graph_tree_test_10k.npz 中的 node_coords（与 gen_adj 按索引对齐）

用法（项目根目录）：
    python -m node_diffusion_cross_att.eval_coord_rmse \
        --ckpt checkpoints/node_diffusion_cross_att/latest.pt \
        --bert models/bert-base-uncased

    # 指定评估条数（0=全量）
    python -m node_diffusion_cross_att.eval_coord_rmse \
        --ckpt checkpoints/node_diffusion_cross_att/latest.pt \
        --n_eval 1000
"""

import argparse
import os
import numpy as np
import torch

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--bert',     default='models/bert-base-uncased')
    p.add_argument('--gen_adj',  default='data/processed/node_diffusion_cross_att/gen_adj_test.npz',
                   help='θ₁ 输出：生成的邻接矩阵 + BERT prompt')
    p.add_argument('--gt_npz',   default='data/processed/graph_tree/text_graph_tree_test_10k.npz',
                   help='含 node_coords 的测试集 npz（与 gen_adj 按索引对齐）')
    p.add_argument('--n_eval',   type=int, default=0,
                   help='评估样本数（0=全量 valid 样本）')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ddim_steps', type=int, default=0,
                   help='DDIM 步数（0=完整 DDPM 1000步）')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--out',      default='outputs/eval_coord_rmse/results.npz',
                   help='推理结果保存路径')
    return p.parse_args()


@torch.no_grad()
def sample_batch(model, diffusion, cond, device, ddim_steps):
    """返回预测坐标 numpy [B, 40, 2]。"""
    adj  = cond['adj_matrix'].to(device)
    mask = cond['node_mask'].to(device)
    ptok = cond['prompt_tokens'].to(device)
    pmsk = cond['prompt_mask'].to(device).long()
    B    = adj.shape[0]

    text_hidden = model.bert(input_ids=ptok, attention_mask=pmsk).last_hidden_state
    text_feat   = model.text_proj(text_hidden)
    text_mask   = (1 - pmsk.float()).unsqueeze(1)

    def fwd(x, t_val):
        from node_diffusion_cross_att.model import timestep_embedding
        tb    = torch.full((B,), t_val, device=device, dtype=torch.long)
        x_in  = x.permute(0, 2, 1).float()
        t_emb = model.time_embed(timestep_embedding(tb, model.model_channels)).unsqueeze(1)
        h     = model.input_emb(x_in) + t_emb
        am    = model._build_adj_mask(adj.float(), mask.float())
        for layer in model.layers:
            h = layer(h, am, text_feat, text_mask)
        return model.coord_head(h).permute(0, 2, 1).float()

    x = torch.randn(B, 2, 40, device=device)

    if ddim_steps > 0:
        step_seq = np.linspace(diffusion.T - 1, 0, ddim_steps, dtype=int).tolist()
        for i, t in enumerate(step_seq):
            eps = fwd(x, t)
            ab  = diffusion.alphas_bar[t]
            x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
            if i + 1 < len(step_seq):
                ab_prev = diffusion.alphas_bar[step_seq[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
    else:
        for t in reversed(range(diffusion.T)):
            eps = fwd(x, t)
            ab  = diffusion.alphas_bar[t]
            ap  = diffusion.alphas_bar_prev[t]
            a   = diffusion.alphas[t]
            b   = diffusion.betas[t]
            x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
            mu  = (ap.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
            x   = mu + diffusion.posterior_variance[t].sqrt() * torch.randn_like(x) if t > 0 else mu

    return x.permute(0, 2, 1).cpu().numpy()  # [B, 40, 2]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    # 加载 gen_adj_test.npz（θ₁ 输出，作为 θ₂ 条件）
    gen  = np.load(args.gen_adj, allow_pickle=True)
    # 加载 GT 坐标（与 gen_adj 按索引对齐）
    gt   = np.load(args.gt_npz,  allow_pickle=True)

    assert 'node_coords' in gt.files, \
        f"{args.gt_npz} 中没有 node_coords，请用修改后的 build_text_graph_tree.py 重新生成"

    total     = len(gen['valid'])
    valid_idx = np.where(gen['valid'])[0]
    n_eval    = len(valid_idx) if args.n_eval == 0 else min(args.n_eval, len(valid_idx))
    rng       = np.random.default_rng(args.seed)
    chosen    = sorted(rng.choice(valid_idx, size=n_eval, replace=False).tolist())
    print(f'gen_adj_test: 共 {total} 条，有效 {len(valid_idx)} 条，评估 {n_eval} 条')

    mode = f'DDIM-{args.ddim_steps}' if args.ddim_steps > 0 else 'DDPM-1000'
    print(f'采样模式: {mode}')

    adj_all  = gen['adj_matrix'  ][chosen].astype('float32')
    mask_all = gen['node_mask'   ][chosen].astype('float32')
    ptok_all = gen['prompt_tokens'][chosen].astype('int64')
    pmsk_all = gen['prompt_mask' ][chosen].astype('float32')
    gt_coords = gt['node_coords' ][chosen].astype('float32')  # [N, 40, 2]
    gt_mask   = gt['node_mask'   ][chosen].astype('float32')  # [N, 40]

    BS        = args.batch_size
    n_batches = (n_eval + BS - 1) // BS
    all_preds = []

    for bi in range(n_batches):
        s, e = bi * BS, min((bi + 1) * BS, n_eval)
        cond = {
            'adj_matrix':    torch.from_numpy(adj_all[s:e]),
            'node_mask':     torch.from_numpy(mask_all[s:e]),
            'prompt_tokens': torch.from_numpy(ptok_all[s:e]),
            'prompt_mask':   torch.from_numpy(pmsk_all[s:e]),
        }
        preds = sample_batch(model, diffusion, cond, device, args.ddim_steps)
        all_preds.append(preds)
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f'  [{min(e, n_eval)}/{n_eval}]', flush=True)

    all_preds = np.concatenate(all_preds, axis=0)  # [N, 40, 2]

    # 计算 Coord-RMSE（仅有效节点，以 GT mask 为准）
    rmse_list = []
    for i in range(n_eval):
        valid = gt_mask[i] > 0.5
        err   = all_preds[i][valid] - gt_coords[i][valid]
        rmse_list.append(float(np.sqrt(np.mean(err ** 2))))

    rmse_arr = np.array(rmse_list)
    print(f'\n{"─"*48}')
    print(f'采样模式  : {mode}')
    print(f'样本数    : {n_eval}')
    print(f'Coord-RMSE: {rmse_arr.mean():.4f}  '
          f'(std={rmse_arr.std():.4f}, median={np.median(rmse_arr):.4f})')
    print(f'{"─"*48}')
    print(f'\n论文主表填入: {rmse_arr.mean():.2f}')

    # 保存推理结果
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        pred_coords  = all_preds,          # [N, 40, 2] 预测坐标
        gt_coords    = gt_coords,           # [N, 40, 2] 真实坐标
        gt_mask      = gt_mask,             # [N, 40]    有效节点掩码
        adj_matrix   = adj_all,            # [N, 40, 40] 生成邻接矩阵
        sample_idx   = np.array(chosen),   # [N]         原始数据集索引
        rmse_per_sample = rmse_arr,        # [N]         每条样本 RMSE
    )
    print(f'结果已保存: {args.out}')


if __name__ == '__main__':
    main()
