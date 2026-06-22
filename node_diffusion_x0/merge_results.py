"""
将四个 npz 合并为一个完整的推理结果 npz。

输入：
  outputs/eval_coord_rmse/results.npz      — θ₂ 预测坐标
  outputs/infer_type/results.npz           — θ₃ 预测节点类型
  data/processed/node_diffusion_cross_att/gen_adj_test.npz  — θ₁ 邻接矩阵 + prompt
  data/processed/graph_tree/text_graph_tree_test_10k.npz    — GT 坐标 + mask

输出：
  outputs/merged_results.npz
    sample_idx      [N]         原始测试集索引
    pred_coords     [N, 40, 2]  θ₂ 预测坐标
    gt_coords       [N, 40, 2]  GT 坐标
    node_mask       [N, 40]     有效节点掩码
    adj_matrix      [N, 40, 40] θ₁ 生成邻接矩阵
    pred_type_ids   [N, 40]     θ₃ 预测节点类型 ID (1-32)
    prompt_tokens   [N, T]      BERT prompt token IDs
    prompt_mask     [N, T]      BERT prompt mask
    rmse_per_sample [N]         θ₂ 每条样本 RMSE

用法：
  python -m node_diffusion_cross_att.merge_results
"""

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coord_results', default='outputs/eval_coord_rmse/results.npz')
    p.add_argument('--type_results',  default='outputs/infer_type/results.npz')
    p.add_argument('--gen_adj',       default='data/processed/node_diffusion_cross_att/gen_adj_test.npz')
    p.add_argument('--gt_npz',        default='data/processed/graph_tree/text_graph_tree_test_10k.npz')
    p.add_argument('--out',           default='outputs/merged_results.npz')
    return p.parse_args()


def main():
    args = parse_args()

    print('加载 eval_coord_rmse/results.npz ...')
    coord = np.load(args.coord_results, allow_pickle=True)

    print('加载 infer_type/results.npz ...')
    typ   = np.load(args.type_results,  allow_pickle=True)

    print('加载 gen_adj_test.npz ...')
    gen   = np.load(args.gen_adj,       allow_pickle=True)

    print('加载 text_graph_tree_test_10k.npz ...')
    gt    = np.load(args.gt_npz,        allow_pickle=True)

    # ── 对齐检查 ──────────────────────────────────────────────────────────────
    idx_coord = coord['sample_idx']
    idx_type  = typ['sample_idx']
    assert np.array_equal(idx_coord, idx_type), \
        'sample_idx 不一致：coord_results 与 type_results 不对齐'

    sample_idx = idx_coord   # [N]
    N = len(sample_idx)
    print(f'共 {N} 条样本')

    # ── 合并 ──────────────────────────────────────────────────────────────────
    merged = dict(
        sample_idx      = sample_idx,
        pred_coords     = coord['pred_coords'].astype(np.float32),    # [N, 40, 2]
        gt_coords       = coord['gt_coords'  ].astype(np.float32),    # [N, 40, 2]
        node_mask       = coord['gt_mask'    ].astype(np.float32),    # [N, 40]
        adj_matrix      = coord['adj_matrix' ].astype(np.float32),    # [N, 40, 40]
        rmse_per_sample = coord['rmse_per_sample'].astype(np.float64),# [N]
        pred_type_ids   = typ['pred_type_ids'].astype(np.int32),      # [N, 40]
        prompt_tokens   = gen['prompt_tokens'][sample_idx].astype(np.int64),   # [N, T]
        prompt_mask     = gen['prompt_mask'  ][sample_idx].astype(np.float32), # [N, T]
        gt_coords_ref   = gt['node_coords'   ][sample_idx].astype(np.float32), # [N, 40, 2]
    )

    # 快速验证对齐
    assert np.allclose(merged['pred_coords'][:3], coord['pred_coords'][:3])
    assert np.allclose(merged['gt_coords'][:3],   coord['gt_coords'][:3])
    print('对齐验证通过')

    # ── 保存 ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **merged)
    print(f'已保存: {args.out}')

    print('\n字段汇总:')
    for k, v in merged.items():
        print(f'  {k:20s}  shape={v.shape}  dtype={v.dtype}')


if __name__ == '__main__':
    main()
