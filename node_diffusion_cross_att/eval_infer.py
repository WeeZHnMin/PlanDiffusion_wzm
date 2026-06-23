"""
θ₂ + θ₃ 推理脚本：对测试集运行推理，将 GT 和预测结果保存为 NPZ。

输出 NPZ 包含：
    gt_coords    : float32 [N, 40, 2]   GT 节点坐标（原像素空间，已以原点为中心）
    gt_combo_ids : int32   [N, 40]      GT 房间类型组合 ID
    gt_adj       : float32 [N, 40, 40]  邻接矩阵
    gt_mask      : float32 [N, 40]      有效节点掩码
    gt_n_nodes   : int32   [N]          有效节点数
    pred_coords  : float32 [N, 40, 2]   θ₂ 预测坐标
    pred_combo_ids: int32  [N, 40]      θ₃ 预测类型组合 ID

Usage (from project root):
    python -m node_diffusion_cross_att.eval_infer \\
        --ckpt2  checkpoints/node_diffusion_cross_att/latest.pt \\
        --ckpt3  checkpoints/node_type/model_latest.pt \\
        --data   data/jsonl/test_graph_dataset_10k.jsonl \\
        --n      500 \\
        --gpu    0 \\
        --out    outputs/eval/infer_500.npz
"""

import argparse
import json
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
    p.add_argument('--ckpt2',       default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--ckpt3',       default='checkpoints/node_type/model_latest.pt')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--n',           type=int, default=500,
                   help='Number of test samples (0 = all)')
    p.add_argument('--batch_size',  type=int, default=8,
                   help='Batch size for θ₂ DDPM inference')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--gpu',         type=int, default=None)
    p.add_argument('--out',         default='outputs/eval/infer_500.npz')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    bert_tok = BertTokenizer.from_pretrained(args.bert)

    # ── Load test data ────────────────────────────────────────────────────────
    print(f'Loading: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()

    total = len(all_lines)
    n     = args.n if args.n > 0 else total
    rng   = np.random.default_rng(args.seed)
    indices = rng.choice(total, size=min(n, total), replace=False).tolist()
    print(f'Samples: {len(indices)} / {total}')

    records = []
    for idx in indices:
        d       = json.loads(all_lines[idx])
        n_nodes = int(d['n_nodes'])
        enc     = bert_tok(d['prompt'], max_length=MAX_BERT_LEN,
                           padding='max_length', truncation=True)
        records.append(dict(
            idx        = idx,
            n_nodes    = n_nodes,
            adj_np     = np.array(d['adj_matrix'],    dtype=np.float32),  # [40,40]
            mask_np    = np.array(d['node_mask'],     dtype=np.float32),  # [40]
            coords_gt  = np.array(d['node_coords'],   dtype=np.float32),  # [40,2]
            combo_gt   = np.array(d['node_combo_ids'],dtype=np.int32),    # [40]
            ptok_np    = np.array(enc['input_ids'],   dtype=np.int64),
            pmsk_np    = np.array(enc['attention_mask'], dtype=np.float32),
        ))

    # ── θ₂: DDPM inference ───────────────────────────────────────────────────
    print(f'\n[θ₂] Loading: {args.ckpt2}')
    model2    = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2     = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    diffusion = GaussianDiffusion(timesteps=1000)
    print(f'  step={ckpt2.get("step", "?")}')
    del ckpt2

    bs = args.batch_size
    for start in range(0, len(records), bs):
        batch  = records[start:start + bs]
        pred   = sample_coords_batch(
            model2, diffusion,
            np.stack([r['adj_np']  for r in batch]),
            np.stack([r['mask_np'] for r in batch]),
            np.stack([r['ptok_np'] for r in batch]),
            np.stack([r['pmsk_np'] for r in batch]),
            device,
            sample_indices=[r['idx'] for r in batch],
        )  # [B, 40, 2]
        for i, rec in enumerate(batch):
            rec['coords_pred'] = pred[i]
        print(f'  θ₂: {min(start+bs, len(records))}/{len(records)}', flush=True)

    del model2, diffusion
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── θ₃: type prediction ──────────────────────────────────────────────────
    print(f'\n[θ₃] Loading: {args.ckpt3}')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step", "?")}')
    del ckpt3

    for rec in records:
        with torch.no_grad():
            logits = model3(
                torch.from_numpy(rec['coords_pred'].T[None]).to(device),
                adj_matrix    = torch.from_numpy(rec['adj_np'][None]).to(device),
                node_mask     = torch.from_numpy(rec['mask_np'][None]).to(device),
                prompt_tokens = torch.from_numpy(rec['ptok_np'][None]).to(device),
                prompt_mask   = torch.from_numpy(rec['pmsk_np'][None]).long().to(device),
            )
            rec['combo_pred'] = logits[0].argmax(dim=-1).cpu().numpy().astype(np.int32)

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        gt_coords     = np.stack([r['coords_gt']   for r in records]),   # [N,40,2]
        gt_combo_ids  = np.stack([r['combo_gt']    for r in records]),   # [N,40]
        gt_adj        = np.stack([r['adj_np']      for r in records]),   # [N,40,40]
        gt_mask       = np.stack([r['mask_np']     for r in records]),   # [N,40]
        gt_n_nodes    = np.array([r['n_nodes']     for r in records], dtype=np.int32),
        pred_coords   = np.stack([r['coords_pred'] for r in records]),   # [N,40,2]
        pred_combo_ids= np.stack([r['combo_pred']  for r in records]),   # [N,40]
    )
    print(f'\nSaved {len(records)} samples → {args.out}')


if __name__ == '__main__':
    main()
