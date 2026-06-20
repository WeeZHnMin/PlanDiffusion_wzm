"""
节点坐标扩散模块（θ₂）Coord-RMSE 评估脚本。

在测试集 GT 邻接矩阵条件下运行 DDPM/DDIM 逆采样，
计算预测坐标与真实坐标的均方根误差，用于填写论文主表。

用法（项目根目录）：
    python -m node_diffusion_cross_att.eval_coord_rmse \
        --ckpt  checkpoints/node_diffusion_cross_att/latest.pt \
        --bert  models/bert-base-uncased \
        --data  data/processed/node_diffusion_cross_att/type_dataset_test_10k.npz \
        --n_eval 1000

    # 全量测试集
    python -m node_diffusion_cross_att.eval_coord_rmse \
        --ckpt  checkpoints/node_diffusion_cross_att/latest.pt --n_eval 0
"""

import argparse
import os
import numpy as np
import torch

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--bert',       default='models/bert-base-uncased')
    p.add_argument('--data',       default='data/processed/node_diffusion_cross_att/type_dataset_test_10k.npz')
    p.add_argument('--n_eval',     type=int, default=0,
                   help='评估样本数（0=全部）')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ddim_steps', type=int, default=0,
                   help='DDIM 步数（0=完整 DDPM 1000步）')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def sample_batch(model, diffusion, cond, device, ddim_steps):
    """返回预测坐标 [B, 40, 2]（numpy）。"""
    diff = diffusion
    adj  = cond['adj_matrix'].to(device)
    mask = cond['node_mask'].to(device)
    ptok = cond['prompt_tokens'].to(device)
    pmsk = cond['prompt_mask'].to(device).long()

    B = adj.shape[0]

    # BERT 预计算（只跑一次）
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
        step_seq = np.linspace(diff.T - 1, 0, ddim_steps, dtype=int).tolist()
        for i, t in enumerate(step_seq):
            eps = fwd(x, t)
            ab  = diff.alphas_bar[t]
            x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
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

    # 加载数据
    data  = np.load(args.data, allow_pickle=True)
    total = len(data['node_coords'])
    n_eval = total if args.n_eval == 0 else min(args.n_eval, total)
    rng   = np.random.default_rng(args.seed)
    idxs  = sorted(rng.choice(total, size=n_eval, replace=False).tolist())
    print(f'数据集: {total} 条  评估: {n_eval} 条')

    mode = f'DDIM-{args.ddim_steps}' if args.ddim_steps > 0 else 'DDPM-1000'
    print(f'采样模式: {mode}')

    gt_coords  = data['node_coords'][idxs].astype('float32')   # [N, 40, 2]
    adj_all    = data['adj_matrix'][idxs].astype('float32')
    mask_all   = data['node_mask'][idxs].astype('float32')
    ptok_all   = data['prompt_tokens'][idxs].astype('int64')
    pmsk_all   = data['prompt_mask'][idxs].astype('float32')

    BS = args.batch_size
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

    # 计算 Coord-RMSE（仅有效节点）
    rmse_list = []
    for i in range(n_eval):
        valid = mask_all[i] > 0.5
        err   = all_preds[i][valid] - gt_coords[i][valid]
        rmse_list.append(float(np.sqrt(np.mean(err ** 2))))

    rmse_arr = np.array(rmse_list)
    print(f'\n{"─"*45}')
    print(f'评估模式  : {mode}')
    print(f'样本数    : {n_eval}')
    print(f'Coord-RMSE: {rmse_arr.mean():.4f}  (std={rmse_arr.std():.4f}, median={np.median(rmse_arr):.4f})')
    print(f'{"─"*45}')
    print(f'\n论文主表填入: {rmse_arr.mean():.2f}')


if __name__ == '__main__':
    main()
