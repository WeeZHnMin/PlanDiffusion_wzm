"""
Gacha 可视化：相同输入（GT adj + text），不同初始噪声种子，观察 θ₂ 生成的多样性。

输出 N_samples 行 × (1 + rolls) 列图：
  Col 0   文本描述
  Col 1~K 每次不同噪声种子的渲染平面图

不同 roll 的噪声种子：seed_k = idx + 123456 + k * 1_000_000

用法（项目根目录）：
    python -m node_diffusion_cross_att.visualize_gacha \\
        --ckpt2  checkpoints/node_diffusion_cross_att/latest.pt \\
        --ckpt3  checkpoints/node_type/model_latest.pt \\
        --data   data/jsonl/test_graph_dataset_10k.jsonl \\
        --indices 0 42 100 \\
        --rolls  5 \\
        --out    outputs/visualize_gacha/result.png
"""

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize':   7,
    'font.size':        7,
})

import numpy as np
import torch
from transformers import BertTokenizer

from .model import NodeDiffusionTransformer
from .diffusion import GaussianDiffusion
from .type_model import NodeTypeClassifier
from .render import load_vocab, find_faces, vote_room_type, ROOM_COLORS, ROOM_LABELS
from .visualize_e2e import sample_coords_batch, draw_col3_coords

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt2',       default='checkpoints/node_diffusion/latest.pt')
    p.add_argument('--ckpt3',       default='checkpoints/node_type/20260623_010747/model_latest.pt')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--n',           type=int, default=3)
    p.add_argument('--indices',     type=int, nargs='+', default=None)
    p.add_argument('--rolls',       type=int, default=5)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--gpu',         type=int, default=None)
    p.add_argument('--out',         default='outputs/visualize_gacha/result.png')
    return p.parse_args()


def render_to_ax(ax, coords: np.ndarray, adj: np.ndarray, n: int,
                 node_types: List[List[str]]) -> None:
    """coords: [n,2]  adj: [n,n]  — 直接在坐标空间渲染，自动归一化到 [0,1]。"""
    coords_list = [(float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
    adj_list    = adj.tolist()

    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj_list[i][j] == 1:
                all_nbrs[i].append(j)

    faces      = find_faces(coords_list, adj_list)
    face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]

    xs = [c[0] for c in coords_list]
    ys = [c[1] for c in coords_list]
    mn_x, mx_x = min(xs), max(xs)
    mn_y, mx_y = min(ys), max(ys)
    span = max(mx_x - mn_x, mx_y - mn_y, 1.0)
    margin = span * 0.12

    def norm(x, y):
        return (
            (x - mn_x + margin) / (span + 2 * margin),
            (y - mn_y + margin) / (span + 2 * margin),
        )

    ax.set_aspect('equal')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#F8F8F8')

    for face, rt in zip(faces, face_types):
        pts = [norm(*coords_list[i]) for i in face]
        poly = MplPolygon(pts, closed=True,
                          facecolor=ROOM_COLORS.get(rt, '#EAEDED'),
                          edgecolor='#555555', linewidth=0.8, alpha=0.88, zorder=1)
        ax.add_patch(poly)
        try:
            rp = ShapelyPolygon(pts).representative_point()
            cx, cy = rp.x, rp.y
        except Exception:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, ROOM_LABELS.get(rt, rt),
                ha='center', va='center', fontsize=5.5, color='#222222', zorder=3)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    id_to_combo = load_vocab(Path(args.combo_vocab))
    bert_tok    = BertTokenizer.from_pretrained(args.bert)

    # ── 读取测试集 ────────────────────────────────────────────────────────────
    with open(args.data, encoding='utf-8') as f:
        all_lines = f.readlines()
    total = len(all_lines)

    if args.indices is not None:
        indices = args.indices
    else:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(total, size=args.n, replace=False).tolist()
    print(f'选取索引: {indices}，每条跑 {args.rolls} 次不同噪声')

    records = []
    for idx in indices:
        d = json.loads(all_lines[idx])
        n = int(d['n_nodes'])
        enc = bert_tok(d['prompt'], max_length=MAX_BERT_LEN,
                       padding='max_length', truncation=True)
        records.append(dict(
            idx     = idx,
            n_nodes = n,
            text    = d['prompt'],
            adj_np  = np.array(d['adj_matrix'],    dtype=np.float32),
            mask_np = np.array(d['node_mask'],     dtype=np.float32),
            combo_gt= d['node_combo_ids'],
            ptok_np = np.array(enc['input_ids'],   dtype=np.int64),
            pmsk_np = np.array(enc['attention_mask'], dtype=np.float32),
        ))

    # ── θ₂ 加载 ───────────────────────────────────────────────────────────────
    print(f'\n[θ₂] 加载: {args.ckpt2}')
    model2    = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2     = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    diffusion = GaussianDiffusion(timesteps=1000)
    print(f'  step={ckpt2.get("step", "?")}')
    del ckpt2

    # 每个 (样本, roll) 独立推理，种子 = idx + 123456 + roll * 1_000_000
    for rec in records:
        rec['rolls'] = []
        for k in range(args.rolls):
            fake_idx = rec['idx'] + k * 1_000_000   # 不同 roll 用不同种子
            print(f'  [θ₂] idx={rec["idx"]} roll={k} ...', flush=True)
            with torch.no_grad():
                coords = sample_coords_batch(
                    model2, diffusion,
                    rec['adj_np'][None], rec['mask_np'][None],
                    rec['ptok_np'][None], rec['pmsk_np'][None],
                    device, sample_indices=[fake_idx],
                )[0]   # [40, 2]
            rec['rolls'].append({'coords': coords})

    del model2, diffusion
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── θ₃ 加载 ───────────────────────────────────────────────────────────────
    print(f'\n[θ₃] 加载: {args.ckpt3}')
    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    print(f'  step={ckpt3.get("step", "?")}')
    del ckpt3

    for rec in records:
        for roll in rec['rolls']:
            with torch.no_grad():
                logits = model3(
                    torch.from_numpy(roll['coords'].T[None]).to(device),
                    adj_matrix    = torch.from_numpy(rec['adj_np'][None]).to(device),
                    node_mask     = torch.from_numpy(rec['mask_np'][None]).to(device),
                    prompt_tokens = torch.from_numpy(rec['ptok_np'][None]).to(device),
                    prompt_mask   = torch.from_numpy(rec['pmsk_np'][None]).long().to(device),
                )
                roll['combo_ids'] = logits[0].argmax(dim=-1).cpu().numpy()

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    # 布局：Col0=文本, 每个roll占两列=[坐标图, 房间渲染图]
    B = len(records)
    K = args.rolls
    COL_W = [3.5] + [1.8, 2.2] * K
    ROW_H  = 2.5
    fig, axes = plt.subplots(
        B, 1 + 2 * K,
        figsize=(sum(COL_W) + 0.2, B * ROW_H + 0.6),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if B == 1:
        axes = [axes]

    for row_i, rec in enumerate(records):
        axs = axes[row_i]
        n   = rec['n_nodes']
        adj = rec['adj_np']

        # Col 0: 文字描述
        axs[0].axis('off')
        axs[0].text(0.5, 0.5, textwrap.fill(rec['text'], width=32),
                    ha='center', va='center', fontsize=6,
                    transform=axs[0].transAxes, multialignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                              edgecolor='#CCCCCC', linewidth=0.7))

        # Col 1,2 | 3,4 | ... : 每次 roll 的坐标图 + 房间渲染图
        for k, roll in enumerate(rec['rolls']):
            ax_coord  = axs[1 + 2 * k]
            ax_render = axs[2 + 2 * k]
            node_types = [id_to_combo.get(int(roll['combo_ids'][i]), ['other'])
                          for i in range(n)]
            # 坐标图：节点位置 + 边连接
            try:
                draw_col3_coords(ax_coord, roll['coords'], adj, rec['mask_np'],
                                 roll['combo_ids'])
            except Exception as e:
                ax_coord.axis('off')
                ax_coord.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                              fontsize=5, transform=ax_coord.transAxes)
            # 房间渲染图
            try:
                render_to_ax(ax_render, roll['coords'][:n], adj[:n, :n], n, node_types)
            except Exception as e:
                ax_render.axis('off')
                ax_render.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                               fontsize=5, transform=ax_render.transAxes)

        axs[0].set_ylabel(f"#{rec['idx']}", fontsize=6, labelpad=2)

    # 列标题
    axes[0][0].set_title('Text Description', fontsize=9, fontweight='bold', pad=4)
    for k in range(K):
        axes[0][1 + 2 * k].set_title(f'Roll {k+1} Coords', fontsize=8, fontweight='bold', pad=4)
        axes[0][2 + 2 * k].set_title(f'Roll {k+1} Render', fontsize=8, fontweight='bold', pad=4)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    pdf_out = os.path.splitext(args.out)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    plt.close(fig)
    print(f'\n已保存: {args.out}')
    print(f'已保存: {pdf_out}')

    # 保存输入数据到 JSONL，方便检查邻接图是否有断开
    jsonl_out = os.path.splitext(args.out)[0] + '_input.jsonl'
    with open(jsonl_out, 'w', encoding='utf-8') as jf:
        for rec in records:
            jf.write(json.dumps({
                'idx':        rec['idx'],
                'n_nodes':    rec['n_nodes'],
                'adj_matrix': rec['adj_np'].tolist(),
                'node_mask':  rec['mask_np'].tolist(),
                'combo_ids':  rec['combo_gt'],
                'text':       rec['text'],
            }, ensure_ascii=False) + '\n')
    print(f'已保存: {jsonl_out}')


if __name__ == '__main__':
    main()
