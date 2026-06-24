"""
对已有推理 NPZ 做节点-墙体吸附后处理，再重跑 θ₃ 预测节点类型，输出新 NPZ。

流程：
    1. 读取 infer NPZ（含 pred_coords [N,K,40,2]、gt_adj [N,40,40] 等）
    2. 对每条样本的每个 roll：
       a. 将靠近某条边的节点投影到该边（吸附）
       b. 更新邻接图（插入节点：加 i-j、i-k，删 j-k）
    3. 批量重跑 θ₃，得到新的 pred_combo_ids [N,K,40]
    4. 保存新 NPZ（在原字段基础上覆盖 pred_coords / pred_adj / pred_combo_ids）

用法：
    python -m node_diffusion_cross_att.snap_and_retype \\
        --npz    outputs/eval/infer_all_5roll.npz \\
        --ckpt3  checkpoints/node_type/model_latest.pt \\
        --out    outputs/eval/infer_all_5roll_snapped.npz \\
        --snap_thresh 0.05 \\
        --batch  64 \\
        --gpu    0
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer

from .type_model import NodeTypeClassifier
from .visualize_e2e import snap_nodes_to_walls

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',         required=True,  help='输入 NPZ 路径')
    p.add_argument('--ckpt3',       required=True,  help='θ₃ checkpoint 路径')
    p.add_argument('--out',         required=True,  help='输出 NPZ 路径')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_10k.jsonl',
                   help='原始 JSONL，用于读取 prompt（θ₃ 需要文本条件）')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--snap_thresh', type=float, default=0.05,
                   help='吸附阈值，相对包围盒对角线比例（默认 0.05）')
    p.add_argument('--batch',       type=int,   default=64,
                   help='θ₃ 推理批次大小（按样本×roll 计）')
    p.add_argument('--workers',     type=int,   default=0,
                   help='吸附多进程数（0=CPU核心数）')
    p.add_argument('--gpu',         type=int,   default=None)
    return p.parse_args()


def _snap_one(args_tuple):
    """多进程工作函数：对单条样本的所有 roll 做吸附。"""
    i, coords_ik, adj_ik, n, thresh = args_tuple   # coords_ik: [K,40,2]
    K = coords_ik.shape[0]
    out_c = coords_ik.copy()
    out_a = adj_ik.copy()
    for k in range(K):
        c, a = snap_nodes_to_walls(out_c[k], out_a[k], n, thresh)
        out_c[k] = c
        out_a[k] = a
    return i, out_c, out_a


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ── 读取 NPZ ──────────────────────────────────────────────────────────────
    print(f'读取 NPZ: {args.npz}')
    data        = np.load(args.npz)
    N           = int(data['gt_n_nodes'].shape[0])
    K           = int(data['rolls'])
    gt_adj      = data['gt_adj']        # [N, 40, 40]
    gt_n_nodes  = data['gt_n_nodes']    # [N]
    gt_mask     = data['gt_mask']       # [N, 40]
    pred_coords = data['pred_coords']   # [N, K, 40, 2]
    si          = data['sample_indices'] if 'sample_indices' in data else np.arange(N, dtype=np.int32)
    print(f'N={N}  K={K}')

    # ── 读取 JSONL prompt，用 BERT 编码 ──────────────────────────────────────
    print(f'读取 JSONL: {args.data}')
    bert_tok = BertTokenizer.from_pretrained(args.bert)
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()

    ptok_all = np.zeros((N, MAX_BERT_LEN), dtype=np.int64)
    pmsk_all = np.zeros((N, MAX_BERT_LEN), dtype=np.float32)
    for i in range(N):
        line = all_lines[int(si[i])]
        import json
        d    = json.loads(line)
        enc  = bert_tok(d['prompt'], max_length=MAX_BERT_LEN,
                        padding='max_length', truncation=True)
        ptok_all[i] = enc['input_ids']
        pmsk_all[i] = enc['attention_mask']
    print('BERT 编码完成')

    # ── 吸附后处理：多进程并行 ────────────────────────────────────────────────
    n_workers = args.workers if args.workers > 0 else cpu_count()
    print(f'节点吸附（thresh_ratio={args.snap_thresh}，workers={n_workers}）...')

    snapped_coords = pred_coords.copy()          # [N, K, 40, 2]
    snapped_adj    = np.stack(                   # [N, K, 40, 40]
        [np.stack([gt_adj[i].copy() for _ in range(K)]) for i in range(N)]
    )

    tasks = [
        (i, snapped_coords[i], snapped_adj[i], int(gt_n_nodes[i]), args.snap_thresh)
        for i in range(N)
    ]

    with Pool(processes=n_workers) as pool:
        for done, (i, c, a) in enumerate(pool.imap_unordered(_snap_one, tasks, chunksize=32)):
            snapped_coords[i] = c
            snapped_adj[i]    = a
            if (done + 1) % 500 == 0:
                print(f'  {done+1}/{N}', flush=True)

    print('吸附完成')

    # ── 重跑 θ₃ ──────────────────────────────────────────────────────────────
    print(f'\n[θ₃] 加载: {args.ckpt3}')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step", "?")}')
    del ckpt3

    # 把 [N, K, 40, ...] 展开成 [N*K, 40, ...] 分批推理
    total    = N * K
    bs       = args.batch
    new_combo = np.zeros((N, K, 40), dtype=np.int32)

    for start in range(0, total, bs):
        end    = min(start + bs, total)
        idxs   = [(start + m) // K for m in range(end - start)]  # 样本索引
        rolls  = [(start + m) %  K for m in range(end - start)]  # roll 索引

        coords_b = np.stack([snapped_coords[i, k] for i, k in zip(idxs, rolls)])  # [B,40,2]
        adj_b    = np.stack([snapped_adj[i, k]    for i, k in zip(idxs, rolls)])  # [B,40,40]
        mask_b   = np.stack([gt_mask[i]           for i    in idxs])              # [B,40]
        ptok_b   = np.stack([ptok_all[i]          for i    in idxs])              # [B,T]
        pmsk_b   = np.stack([pmsk_all[i]          for i    in idxs])              # [B,T]

        with torch.no_grad():
            logits = model3(
                torch.from_numpy(coords_b.transpose(0, 2, 1)).to(device),
                adj_matrix    = torch.from_numpy(adj_b).to(device),
                node_mask     = torch.from_numpy(mask_b).to(device),
                prompt_tokens = torch.from_numpy(ptok_b).to(device),
                prompt_mask   = torch.from_numpy(pmsk_b).long().to(device),
            )  # [B, 40, n_combos]
        preds = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)  # [B, 40]

        for m, (i, k) in enumerate(zip(idxs, rolls)):
            new_combo[i, k] = preds[m]

        if (end // K) % 500 == 0 or end == total:
            print(f'  θ₃: {end}/{total}', flush=True)

    print('θ₃ 推理完成')

    # ── 保存新 NPZ ────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(data)          # 保留原有所有字段
    save_dict['pred_coords']    = snapped_coords   # 吸附后坐标
    save_dict['pred_adj']       = snapped_adj      # 吸附后邻接图（新增字段）
    save_dict['pred_combo_ids'] = new_combo        # 重新预测的类型
    np.savez(args.out, **save_dict)
    print(f'\n已保存 → {args.out}')


if __name__ == '__main__':
    main()
