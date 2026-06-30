"""
文本消融可视化：固定同一条文本编码，测试文本编码器的贡献度。

对 5 个不同样本，分别用：
  A. 该样本自身的文本编码  → 正常生成
  B. 固定的同一条文本编码  → 消融生成

若两列结果相同 → 文本编码器没有贡献（模型只靠邻接图生成）
若两列结果不同 → 文本编码器有贡献

输出 5 行 × 6 列图：
  Col1  样本文本
  Col2  GT 邻接图
  Col3  GT 平面图
  Col4  预测平面图（自身文本）
  Col5  预测平面图（固定文本）
  Col6  固定文本内容

用法：
    python -m node_diffusion_room_tri.visualize_text_ablation \\
        --ckpt  checkpoints/node_diffusion_room_tri/latest.pt \\
        --data  data/jsonl/final_graph_dataset_v3.jsonl \\
        --out   outputs/text_ablation/result.png \\
        --fixed_idx 0
"""

import argparse
import json
import math
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize':   7,
    'font.size':        7,
})

import numpy as np
import torch
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer, _assign_room_membership_single
from .diffusion import GaussianDiffusion
from .visualize_gt_adj import (
    spring_layout, _ax_style, draw_col1_text, draw_col2_adj,
    draw_col3_coords, draw_col4_render,
    find_faces, vote_room_type, _build_sorted_neighbors,
    ROOM_COLORS, ROOM_LABELS, ROOM_TYPE_ORDER,
)

MAX_BERT_LEN = 224
N_NODES      = 40


# ── DDIM 采样（比 DDPM 快 5 倍）────────────────────────────────────────────────

@torch.no_grad()
def ddpm_sample(model, diffusion, room_mb, adj, mask, text_feat, text_mask,
                device, seed=0):
    diffusion._to(device)
    g = torch.Generator(device=device); g.manual_seed(seed)
    x = torch.randn(1, 2, N_NODES, device=device, generator=g)
    for t in reversed(range(diffusion.T)):
        tb  = torch.full((1,), t, device=device, dtype=torch.long)
        eps = model(x, tb, mask,
                    text_feat=text_feat, text_mask=text_mask,
                    room_membership=room_mb, adj_matrix=adj)
        ab  = diffusion.alphas_bar[t]
        ap  = diffusion.alphas_bar_prev[t]
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-3)).clamp(-300, 300)
        a   = diffusion.alphas[t]
        b_  = diffusion.betas[t]
        mu  = (ap.sqrt() * b_ / (1 - ab)) * x0 + (a.sqrt() * (1 - ap) / (1 - ab)) * x
        if t > 0:
            x = mu + diffusion.posterior_variance[t].sqrt() * \
                torch.randn(x.shape, device=device, generator=g)
        else:
            x = mu
    return x.squeeze(0).permute(1, 0).cpu().numpy()   # [N_NODES, 2]


def encode_text(model, ptok_np, pmsk_np, device):
    ptok = torch.from_numpy(ptok_np[None]).to(device)
    pmsk = torch.from_numpy(pmsk_np[None]).long().to(device)
    text_feat, text_mask = model.encode_text(ptok, pmsk)
    return text_feat, text_mask   # keep on device


def prepare_record(d, bert_tok):
    n      = int(d['n_nodes'])
    adj_np = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    raw_a  = np.array(d['adj_matrix'], dtype=np.float32)[:N_NODES, :N_NODES]
    adj_np[:raw_a.shape[0], :raw_a.shape[1]] = raw_a

    mask_np = np.zeros(N_NODES, dtype=np.float32); mask_np[:min(n, N_NODES)] = 1.0

    raw_types  = d.get('node_types', [])
    node_types = []
    for k in range(N_NODES):
        t = raw_types[k] if k < len(raw_types) else ['other']
        node_types.append(t if isinstance(t, list) else [t])

    n_eff      = int(mask_np.sum())
    adj_bool   = adj_np[:n_eff, :n_eff].astype(bool)
    room_mb_np = np.zeros((N_NODES, _assign_room_membership_single(adj_bool, n_eff).shape[1]),
                           dtype=np.float32)
    room_mb_np[:n_eff] = _assign_room_membership_single(adj_bool, n_eff)

    text = d['prompt']
    enc  = bert_tok(text, max_length=MAX_BERT_LEN, padding='max_length', truncation=True)
    ptok_np = np.array(enc['input_ids'],      dtype=np.int64)
    pmsk_np = np.array(enc['attention_mask'], dtype=np.float32)

    gt_coords_np = np.zeros((N_NODES, 2), dtype=np.float32)
    for k in range(min(n_eff, len(d.get('node_coords', [])))):
        gt_coords_np[k] = d['node_coords'][k]

    return dict(text=text, n_nodes=n_eff, adj_np=adj_np, mask_np=mask_np,
                node_types=node_types, room_mb_np=room_mb_np,
                ptok_np=ptok_np, pmsk_np=pmsk_np, gt_coords_np=gt_coords_np)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       default='checkpoints/node_diffusion_room_tri/latest.pt')
    p.add_argument('--data',       default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--bert',       default='models/bert-base-uncased')
    p.add_argument('--n',          type=int, default=5)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--gpu',        type=int, default=None)
    p.add_argument('--fixed_idx',  type=int, default=0,
                   help='用第几条样本的文本作为固定文本（默认第0条）')
    p.add_argument('--fixed_prompt', default='',
                   help='直接指定固定文本（优先于 --fixed_idx）')
    p.add_argument('--out',        default='outputs/text_ablation/result.png')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    np.random.seed(args.seed)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tok = BertTokenizer.from_pretrained(args.bert)

    print(f'读取: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_lines = [l for l in f if l.strip()]

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_lines), size=args.n, replace=False).tolist()
    print(f'选取样本索引: {indices}')
    records = [prepare_record(json.loads(all_lines[i]), bert_tok) for i in indices]

    # ── 固定文本编码 ───────────────────────────────────────────────────────────
    if args.fixed_prompt:
        fixed_text = args.fixed_prompt
    else:
        fixed_text = json.loads(all_lines[args.fixed_idx])['prompt']
    print(f'固定文本 (idx={args.fixed_idx}): {fixed_text[:80]}...')

    enc_fix       = bert_tok(fixed_text, max_length=MAX_BERT_LEN,
                             padding='max_length', truncation=True)
    fixed_ptok_np = np.array(enc_fix['input_ids'],      dtype=np.int64)
    fixed_pmsk_np = np.array(enc_fix['attention_mask'], dtype=np.float32)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f'加载模型: {args.ckpt}')
    model = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=False)
    model.eval()
    print(f'  step={ckpt.get("step","?")}')
    del ckpt

    diffusion = GaussianDiffusion(timesteps=1000)

    # 预计算固定文本特征（只算一次）
    fixed_feat, fixed_mask = encode_text(model, fixed_ptok_np, fixed_pmsk_np, device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, rec in enumerate(records):
            print(f'  DDIM {args.ddim_steps}步  样本{i}...', flush=True)
            room_mb = torch.from_numpy(rec['room_mb_np'][None]).float().to(device)
            adj     = torch.from_numpy(rec['adj_np'][None]).float().to(device)
            mask    = torch.from_numpy(rec['mask_np'][None]).float().to(device)

            # 自身文本
            own_feat, own_mask = encode_text(model, rec['ptok_np'], rec['pmsk_np'], device)
            rec['pred_own']   = ddpm_sample(model, diffusion, room_mb, adj, mask,
                                            own_feat, own_mask, device, seed=i)
            # 固定文本
            rec['pred_fixed'] = ddpm_sample(model, diffusion, room_mb, adj, mask,
                                            fixed_feat, fixed_mask, device, seed=i)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    B = len(records)
    COL_W = [3.8, 2.4, 2.6, 2.6, 2.6, 3.8]
    ROW_H = 2.7
    fig, axes = plt.subplots(
        B, 6,
        figsize=(sum(COL_W) + 0.2, B * ROW_H + 0.8),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        draw_col1_text  (axs[0], rec['text'])
        draw_col2_adj   (axs[1], rec['adj_np'], rec['n_nodes'], seed=args.seed)
        draw_col4_render(axs[2], rec['gt_coords_np'], rec['adj_np'],
                         rec['mask_np'], rec['node_types'])
        draw_col4_render(axs[3], rec['pred_own'],   rec['adj_np'],
                         rec['mask_np'], rec['node_types'])
        draw_col4_render(axs[4], rec['pred_fixed'], rec['adj_np'],
                         rec['mask_np'], rec['node_types'])
        draw_col1_text  (axs[5], fixed_text)

    col_titles = [
        'Sample Text',
        'GT Adjacency',
        'GT Floor Plan',
        'Pred (own text)',
        'Pred (fixed text)',
        'Fixed Text',
    ]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=11, fontweight='bold', pad=5)
        if j in (3, 4):
            axes[0][j].set_title(title, fontsize=11, fontweight='bold', pad=5,
                                 color='#C0392B' if j == 4 else '#1A5276')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches='tight')
    pdf_out = os.path.splitext(args.out)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {args.out}')
    print(f'已保存: {pdf_out}')


if __name__ == '__main__':
    main()
