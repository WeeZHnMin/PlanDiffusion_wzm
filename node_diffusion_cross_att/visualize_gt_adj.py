"""
用测试集 GT 邻接矩阵 + 文本条件，跳过 θ₁ 直接运行 θ₂→θ₃→render 并可视化。

用于隔离评估 θ₂/θ₃ 的生成质量，排除 θ₁ 误差的影响。

输出 N 行 × 5 列图：
  Col1  输入文本描述
  Col2  GT 邻接图（spring layout）
  Col3  θ₂ 预测坐标图（节点统一蓝色）
  Col4  θ₃ 类型预测（节点按类型着色）
  Col5  渲染平面图

用法（项目根目录）：
    python -m node_diffusion_cross_att.visualize_gt_adj \\
        --ckpt2  checkpoints/node_diffusion_cross_att/latest.pt \\
        --ckpt3  checkpoints/node_type/20260616_223156/model_latest.pt \\
        --data   data/jsonl/test_graph_dataset_10k.jsonl \\
        --indices 0 42 100 200 500 \\
        --out    outputs/visualize_gt_adj/result.png
"""

import argparse
import math
import os
import textwrap
from pathlib import Path
from typing import Dict, List

from shapely.geometry import Polygon as ShapelyPolygon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize':   7,
    'font.size':        7,
})

import json
import numpy as np
import torch
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .type_model import NodeTypeClassifier
from .render import (
    load_vocab,
    find_faces,
    vote_room_type,
    _build_sorted_neighbors,
    ROOM_COLORS,
    ROOM_LABELS,
)
from .visualize_e2e import (
    COMBO_COLORS,
    type_color,
    spring_layout,
    sample_coords_batch,
    _ax_style,
    draw_col1_text,
    draw_col2_adj,
    draw_col3_coords,
    draw_col4_types,
    draw_col5_render,
)

MAX_BERT_LEN = 224


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt2', default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--ckpt3', default='checkpoints/node_type/20260616_223156/model_latest.pt')
    p.add_argument('--data',  default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--n',       type=int, default=5)
    p.add_argument('--indices', type=int, nargs='+', default=None,
                   help='手动指定测试集行号，例如 --indices 0 42 100 200 500')
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--out',   default='outputs/visualize_gt_adj/result.png')
    return p.parse_args()


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    id_to_combo = load_vocab(Path(args.combo_vocab))
    bert_tok    = BertTokenizer.from_pretrained(args.bert)

    # ── 读取测试集 ────────────────────────────────────────────────────────────
    print(f'读取测试集: {args.data}')
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()

    total = len(all_lines)
    if args.indices is not None:
        indices = args.indices
    else:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(total, size=args.n, replace=False).tolist()
    print(f'选取索引: {indices}')

    records = []
    for idx in indices:
        d = json.loads(all_lines[idx])
        n = int(d['n_nodes'])

        adj_np  = np.array(d['adj_matrix'],  dtype=np.float32)   # [40, 40]
        mask_np = np.array(d['node_mask'],   dtype=np.float32)   # [40]

        text = d['prompt']
        enc  = bert_tok(text, max_length=MAX_BERT_LEN,
                        padding='max_length', truncation=True)
        ptok_np = np.array(enc['input_ids'],      dtype=np.int64)
        pmsk_np = np.array(enc['attention_mask'], dtype=np.float32)

        print(f'  [idx={idx}] n_nodes={n}  text={text[:60]}...')
        records.append(dict(idx=idx, text=text, n_nodes=n,
                            adj_np=adj_np, mask_np=mask_np,
                            ptok_np=ptok_np, pmsk_np=pmsk_np))

    # ════════════════════════════════════════════════════════════════════════
    # θ₂：加载 → 逐条 DDPM 采样 → 卸载
    # ════════════════════════════════════════════════════════════════════════
    print('\n[θ₂] 加载模型...')
    model2 = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2  = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    diffusion = GaussianDiffusion(timesteps=1000)
    print(f'  step={ckpt2.get("step","?")}')
    del ckpt2

    for rec in records:
        print(f'  [θ₂] idx={rec["idx"]} DDPM 1000步...', flush=True)
        with torch.no_grad():
            rec['pred_coords'] = sample_coords_batch(
                model2, diffusion,
                rec['adj_np' ][None], rec['mask_np'][None],
                rec['ptok_np'][None], rec['pmsk_np'][None],
                device, sample_indices=[rec['idx']],
            )[0]   # [40, 2]

    del model2, diffusion
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₂] 模型已卸载')

    # ════════════════════════════════════════════════════════════════════════
    # θ₃：加载 → 逐条类型预测 → 卸载
    # ════════════════════════════════════════════════════════════════════════
    print('\n[θ₃] 加载模型...')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step","?")}')
    del ckpt3

    for rec in records:
        with torch.no_grad():
            x_in  = torch.from_numpy(rec['pred_coords'].T[None]).to(device)
            adj_t = torch.from_numpy(rec['adj_np' ][None]).to(device)
            msk_t = torch.from_numpy(rec['mask_np'][None]).to(device)
            ptk_t = torch.from_numpy(rec['ptok_np'][None]).to(device)
            pmk_t = torch.from_numpy(rec['pmsk_np'][None]).long().to(device)
            logits = model3(x_in, adj_matrix=adj_t, node_mask=msk_t,
                            prompt_tokens=ptk_t, prompt_mask=pmk_t)
            rec['type_ids'] = logits[0].argmax(dim=-1).cpu().numpy()
        print(f'  [θ₃] idx={rec["idx"]} 完成')

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('[θ₃] 模型已卸载')

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    B = len(records)
    print(f'\n绘制 {B} × 5 图...')
    COL_W = [4.2, 2.6, 2.6, 2.6, 2.8]
    ROW_H = 2.7
    fig, axes = plt.subplots(
        B, 5,
        figsize=(sum(COL_W) + 0.2, B * ROW_H + 0.55),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        draw_col1_text  (axs[0], rec['text'])
        draw_col2_adj   (axs[1], rec['adj_np'], rec['n_nodes'], seed=args.seed)
        draw_col3_coords(axs[2], rec['pred_coords'], rec['adj_np'], rec['mask_np'], rec['type_ids'])
        draw_col4_types (axs[3], rec['pred_coords'], rec['adj_np'], rec['mask_np'], rec['type_ids'])
        draw_col5_render(axs[4], rec['pred_coords'], rec['adj_np'], rec['mask_np'],
                         rec['type_ids'], id_to_combo)

    col_titles = ['Text Description',
                  r'GT Adjacency Graph',
                  r'$\theta_2$: Coordinate Graph',
                  r'$\theta_3$: Type Prediction',
                  'Rendered Floor Plan']
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=13, fontweight='bold', pad=5)

    for row_i, rec in enumerate(records):
        axes[row_i][0].set_ylabel(f"#{rec['idx']}", fontsize=6, labelpad=2)

    fig.get_layout_engine().set(hspace=0.03, wspace=0.03,
                                h_pad=0.02, w_pad=0.02)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    pdf_out = os.path.splitext(args.out)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {args.out}')
    print(f'已保存: {pdf_out}')


if __name__ == '__main__':
    main()
