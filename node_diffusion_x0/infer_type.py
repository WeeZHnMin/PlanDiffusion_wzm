"""
NodeTypeClassifier (θ₃) 批量推理脚本。

输入：
  outputs/eval_coord_rmse/results.npz  — θ₂ 预测坐标 + 生成邻接矩阵
  data/processed/node_diffusion_cross_att/gen_adj_test.npz — prompt tokens

输出：
  outputs/infer_type/results.npz
    pred_type_ids  [N, 40]  int32   预测节点类型 ID (1-32, 0=padding)
    sample_idx     [N]      int64   与 eval_coord_rmse/results.npz 对齐

用法（项目根目录）：
  python -m node_diffusion_cross_att.infer_type \
      --ckpt checkpoints/node_type/XXXXXX/model_latest.pt
"""

import argparse
import os

import numpy as np
import torch

from .type_model import NodeTypeClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/node_type/20260616_223156/model_latest.pt',
                   help='NodeTypeClassifier checkpoint 路径')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--results',     default='outputs/eval_coord_rmse/results.npz',
                   help='eval_coord_rmse 输出的 npz')
    p.add_argument('--gen_adj',     default='data/processed/node_diffusion_cross_att/gen_adj_test.npz',
                   help='含 prompt_tokens / prompt_mask 的 npz')
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--model_channels', type=int, default=384)
    p.add_argument('--num_layers',     type=int, default=4)
    p.add_argument('--num_heads',      type=int, default=6)
    p.add_argument('--out',         default='outputs/infer_type/results.npz')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = NodeTypeClassifier(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    sd   = ckpt['model']
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f'checkpoint step={ckpt.get("step", "?")}')

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    res = np.load(args.results, allow_pickle=True)
    gen = np.load(args.gen_adj,  allow_pickle=True)

    sample_idx   = res['sample_idx']               # [N]
    pred_coords  = res['pred_coords'].astype('float32')   # [N, 40, 2]
    adj_matrix   = res['adj_matrix'].astype('float32')    # [N, 40, 40]
    node_mask    = res['gt_mask'].astype('float32')       # [N, 40]

    prompt_tokens = gen['prompt_tokens'][sample_idx].astype('int64')   # [N, T]
    prompt_mask   = gen['prompt_mask'  ][sample_idx].astype('float32') # [N, T]

    N  = len(sample_idx)
    BS = args.batch_size
    all_preds = np.zeros((N, 40), dtype=np.int32)

    print(f'共 {N} 条样本，batch_size={BS}')
    for bi in range(0, N, BS):
        s, e = bi, min(bi + BS, N)

        x    = torch.from_numpy(pred_coords[s:e]).permute(0, 2, 1).to(device)  # [B, 2, 40]
        adj  = torch.from_numpy(adj_matrix[s:e]).to(device)
        mask = torch.from_numpy(node_mask[s:e]).to(device)
        ptok = torch.from_numpy(prompt_tokens[s:e]).to(device)
        pmsk = torch.from_numpy(prompt_mask[s:e]).long().to(device)

        logits = model(x, adj_matrix=adj, node_mask=mask,
                       prompt_tokens=ptok, prompt_mask=pmsk)   # [B, 40, 33]
        pred   = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)  # [B, 40]
        all_preds[s:e] = pred

        if (bi // BS + 1) % 20 == 0 or e == N:
            print(f'  [{e}/{N}]', flush=True)

    # ── 保存 ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        pred_type_ids = all_preds,     # [N, 40]
        sample_idx    = sample_idx,    # [N]
    )
    print(f'结果已保存: {args.out}')


if __name__ == '__main__':
    main()
