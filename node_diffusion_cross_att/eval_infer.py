"""
θ₂ + θ₃ 推理脚本：对测试集运行推理，每条样本 K 种噪声初始化，结果保存为 NPZ。

输出 NPZ 包含：
    sample_indices : int32   [N]            对应 JSONL 文件的行号（用于下游对齐）
    seeds          : int64   [N, K]         每个 roll 的噪声初始化种子（保证可复现）
    gt_coords      : float32 [N, 40, 2]     GT 节点坐标
    gt_combo_ids   : int32   [N, 40]        GT 房间类型组合 ID
    gt_adj         : float32 [N, 40, 40]    邻接矩阵
    gt_mask        : float32 [N, 40]        有效节点掩码
    gt_n_nodes     : int32   [N]            有效节点数
    pred_coords    : float32 [N, K, 40, 2]  θ₂ 预测坐标（K 次 roll）
    pred_combo_ids : int32   [N, K, 40]     θ₃ 预测类型（K 次 roll）

噪声种子策略（与 visualize_gacha.py 一致）：
    seed = sample_idx + roll_k * 1_000_000

Usage (from project root):
    python -m node_diffusion_cross_att.eval_infer \\
        --ckpt2      checkpoints/node_diffusion/latest.pt \\
        --ckpt3      checkpoints/node_type/20260623_010747/model_latest.pt \\
        --data       data/jsonl/test_graph_dataset_10k.jsonl \\
        --n          0 \\
        --rolls      5 \\
        --batch      16 \\
        --gpu        0 \\
        --out        outputs/eval/infer_all_5roll.npz \\
        --render-out outputs/visualize_gacha \\
        --workers    8
    # 只保存 NPZ，不渲染：
        --no-render
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .type_model import NodeTypeClassifier
from .visualize_e2e import sample_coords_batch

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt2',  default='checkpoints/node_diffusion/latest.pt')
    p.add_argument('--ckpt3',  default='checkpoints/node_type/20260623_010747/model_latest.pt')
    p.add_argument('--data',   default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--bert',   default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--n',      type=int, default=0,
                   help='样本数（0=全部，与 --start/--end 互斥）')
    p.add_argument('--start',  type=int, default=0,
                   help='起始样本索引（含，配合 --end 做顺序切片，用于多卡并行）')
    p.add_argument('--end',    type=int, default=0,
                   help='结束样本索引（不含，0=到末尾）')
    p.add_argument('--rolls',  type=int, default=5,
                   help='每条样本的噪声初始化次数')
    p.add_argument('--batch',  type=int, default=16,
                   help='推理批次大小（按样本数计，实际 GPU batch = batch × rolls）')
    p.add_argument('--seed',   type=int, default=42,
                   help='随机选样本用的 RNG 种子（rolls 的种子由 sample_idx 决定）')
    p.add_argument('--gpu',        type=int, default=None)
    p.add_argument('--out',        default='outputs/eval/infer_all_5roll.npz')
    p.add_argument('--render-out', default='outputs/visualize_gacha',
                   help='渲染图输出目录（每条样本一张 1×K PNG）')
    p.add_argument('--workers',    type=int, default=0,
                   help='渲染进程数（0=CPU核心数）')
    p.add_argument('--no-render',  action='store_true',
                   help='跳过渲染步骤，只保存 NPZ')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = args.rolls
    print(f'device: {device}  rolls: {K}')

    bert_tok = BertTokenizer.from_pretrained(args.bert)

    # ── 读取测试集 ────────────────────────────────────────────────────────────
    print(f'Loading: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()

    total = len(all_lines)
    if args.start > 0 or args.end > 0:
        # 顺序切片模式（多卡并行用）
        end = args.end if args.end > 0 else total
        indices = list(range(args.start, min(end, total)))
    else:
        n = args.n if args.n > 0 else total
        if n < total:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(total, size=n, replace=False).tolist()
        else:
            indices = list(range(total))
    N = len(indices)
    print(f'Samples: {N} / {total}  [{indices[0]}~{indices[-1]}]')

    records = []
    for idx in indices:
        d = json.loads(all_lines[idx])
        enc = bert_tok(d['prompt'], max_length=MAX_BERT_LEN,
                       padding='max_length', truncation=True)
        records.append(dict(
            idx       = idx,
            n_nodes   = int(d['n_nodes']),
            adj_np    = np.array(d['adj_matrix'],     dtype=np.float32),  # [40,40]
            mask_np   = np.array(d['node_mask'],      dtype=np.float32),  # [40]
            coords_gt = np.array(d['node_coords'],    dtype=np.float32),  # [40,2]
            combo_gt  = np.array(d['node_combo_ids'], dtype=np.int32),    # [40]
            ptok_np   = np.array(enc['input_ids'],    dtype=np.int64),
            pmsk_np   = np.array(enc['attention_mask'], dtype=np.float32),
        ))

    # ── θ₂：DDPM 批量推理 ─────────────────────────────────────────────────────
    print(f'\n[θ₂] Loading: {args.ckpt2}')
    model2    = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2     = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    diffusion = GaussianDiffusion(timesteps=1000)
    print(f'  step={ckpt2.get("step", "?")}')
    del ckpt2

    # 每次取 bs 条样本，K 次 roll 合并为一个 GPU batch（大小 bs*K）
    bs = args.batch
    for start in range(0, N, bs):
        chunk = records[start:start + bs]
        C = len(chunk)

        # 每条样本重复 K 次，对应 K 种 seed
        adj_b  = np.stack([r['adj_np']  for r in chunk for _ in range(K)])  # [C*K,40,40]
        mask_b = np.stack([r['mask_np'] for r in chunk for _ in range(K)])  # [C*K,40]
        ptok_b = np.stack([r['ptok_np'] for r in chunk for _ in range(K)])  # [C*K,T]
        pmsk_b = np.stack([r['pmsk_np'] for r in chunk for _ in range(K)])  # [C*K,T]
        seeds  = [r['idx'] + k * 1_000_000 for r in chunk for k in range(K)]

        with torch.no_grad():
            coords_all = sample_coords_batch(
                model2, diffusion,
                adj_b, mask_b, ptok_b, pmsk_b,
                device, sample_indices=seeds,
            )  # [C*K, 40, 2]

        # 分发：coords_all[i*K : i*K+K] → 样本 i 的 K 次 roll
        for i, rec in enumerate(chunk):
            rec['coords_pred'] = coords_all[i * K: i * K + K]  # [K, 40, 2]

        print(f'  θ₂: {min(start + bs, N)}/{N}', flush=True)

    del model2, diffusion
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── θ₃：类型预测批量推理 ──────────────────────────────────────────────────
    print(f'\n[θ₃] Loading: {args.ckpt3}')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step", "?")}')
    del ckpt3

    for start in range(0, N, bs):
        chunk = records[start:start + bs]
        C = len(chunk)

        # 同样把 K 次 roll 展开成大 batch
        coords_b = np.stack([
            chunk[i]['coords_pred'][k]          # [40, 2]
            for i in range(C) for k in range(K)
        ])  # [C*K, 40, 2]
        adj_b  = np.stack([r['adj_np']  for r in chunk for _ in range(K)])
        mask_b = np.stack([r['mask_np'] for r in chunk for _ in range(K)])
        ptok_b = np.stack([r['ptok_np'] for r in chunk for _ in range(K)])
        pmsk_b = np.stack([r['pmsk_np'] for r in chunk for _ in range(K)])

        with torch.no_grad():
            logits = model3(
                torch.from_numpy(coords_b.transpose(0, 2, 1)).to(device),  # [C*K, 2, 40]
                adj_matrix    = torch.from_numpy(adj_b).to(device),
                node_mask     = torch.from_numpy(mask_b).to(device),
                prompt_tokens = torch.from_numpy(ptok_b).to(device),
                prompt_mask   = torch.from_numpy(pmsk_b).long().to(device),
            )  # [C*K, 40, n_combos]
            preds = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)  # [C*K, 40]

        for i, rec in enumerate(chunk):
            rec['combo_pred'] = preds[i * K: i * K + K]  # [K, 40]

        print(f'  θ₃: {min(start + bs, N)}/{N}', flush=True)

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 保存 NPZ ──────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample_idx_arr = np.array([r['idx'] for r in records], dtype=np.int32)  # [N]
    seeds_arr = np.array(
        [[r['idx'] + k * 1_000_000 for k in range(K)] for r in records],
        dtype=np.int64,
    )  # [N, K]

    np.savez(
        args.out,
        sample_indices = sample_idx_arr,   # [N]    → JSONL 行号
        seeds          = seeds_arr,        # [N, K] → 每个 roll 的噪声种子
        gt_coords      = np.stack([r['coords_gt']   for r in records]),  # [N,40,2]
        gt_combo_ids   = np.stack([r['combo_gt']    for r in records]),  # [N,40]
        gt_adj         = np.stack([r['adj_np']      for r in records]),  # [N,40,40]
        gt_mask        = np.stack([r['mask_np']     for r in records]),  # [N,40]
        gt_n_nodes     = np.array([r['n_nodes']     for r in records], dtype=np.int32),
        pred_coords    = np.stack([r['coords_pred'] for r in records]),  # [N,K,40,2]
        pred_combo_ids = np.stack([r['combo_pred']  for r in records]),  # [N,K,40]
        rolls          = np.array(K, dtype=np.int32),
    )
    print(f'\nSaved {N} samples × {K} rolls → {args.out}')

    # ── 渲染 PNG（可选）────────────────────────────────────────────────────────
    if not args.no_render:
        from .render_infer import _worker_init, _render_one
        render_out = Path(args.render_out)
        render_out.mkdir(parents=True, exist_ok=True)
        n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
        n_workers = min(n_workers, N)
        indices = list(range(N))
        print(f'\n[Render] {N} 条，K={K}，进程数={n_workers} → {render_out}')
        total_ok = total_err = 0
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(args.combo_vocab, args.out, str(render_out), K),
        ) as pool:
            for j, (ok, err) in enumerate(
                pool.imap_unordered(_render_one, indices, chunksize=4)
            ):
                total_ok  += ok
                total_err += err
                if (j + 1) % 200 == 0 or (j + 1) == N:
                    print(f'  {j+1}/{N}  ok={total_ok} err={total_err}', flush=True)
        print(f'[Render] 完成  ok={total_ok} err={total_err}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
