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
from typing import Dict, List

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
from .visualize_e2e import (
    sample_coords_batch,
    CUSTOM_PROMPTS,
)

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt1',       default='checkpoints/llm_graph/stage2/20260614_155601/latest.pt',
                   help='θ₁ checkpoint（--custom 模式必须）')
    p.add_argument('--ckpt2',       default='checkpoints/node_diffusion/latest.pt')
    p.add_argument('--ckpt3',       default='checkpoints/node_type/20260623_010747/model_latest.pt')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json',
                   help='BPE tokenizer（--custom 模式必须）')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--custom',      action='store_true',
                   help='使用内置 CUSTOM_PROMPTS，经 θ₁ 生成邻接图后抽卡')
    p.add_argument('--n',           type=int, default=3)
    p.add_argument('--indices',     type=int, nargs='+', default=None)
    p.add_argument('--rolls',       type=int, default=5)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--gpu',         type=int, default=None)
    p.add_argument('--out',         default='outputs/visualize_gacha')
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

    records = []

    if args.custom:
        # ── --custom：θ₁ 生成邻接图 ─────────────────────────────────────────
        from tokenizers import Tokenizer
        from llm_graph.infer_stage1 import (
            load_model as load_llm, encode_text, parse_sequence, generate, BOS_ID,
        )
        print(f'\n[θ₁] 加载: {args.ckpt1}')
        model1 = load_llm(args.ckpt1, device)
        for i, txt in enumerate(CUSTOM_PROMPTS):
            print(f'  [θ₁] custom#{i} 推理中...', end='', flush=True)
            cpu_state  = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            torch.manual_seed(i + 777777)
            if device.type == 'cuda':
                torch.cuda.manual_seed(i + 777777)
            prefix  = encode_text(txt, args.vocab) + [BOS_ID]
            gen_seq = generate(model1, prefix, device, max_new_tokens=200)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            parsed = parse_sequence(gen_seq)
            print(f'  n_nodes={parsed["n_nodes"]}  valid={parsed["valid"]}')
            if not parsed['valid']:
                continue
            N      = parsed['n_nodes']
            adj_np = np.zeros((40, 40), dtype=np.float32)
            adj_np[:N, :N] = np.array(parsed['adj'], dtype=np.float32)
            mask_np = np.zeros(40, dtype=np.float32)
            mask_np[:N] = 1.0
            enc = bert_tok(txt, max_length=MAX_BERT_LEN,
                           padding='max_length', truncation=True)
            records.append(dict(
                idx     = i,
                n_nodes = N,
                text    = txt,
                adj_np  = adj_np,
                mask_np = mask_np,
                combo_gt= [],
                ptok_np = np.array(enc['input_ids'],      dtype=np.int64),
                pmsk_np = np.array(enc['attention_mask'], dtype=np.float32),
            ))
        del model1
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'[θ₁] 完成，有效 {len(records)}/{len(CUSTOM_PROMPTS)} 条')
    else:
        # ── 读取测试集 ────────────────────────────────────────────────────────
        with open(args.data, encoding='utf-8') as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if args.indices is not None:
            indices = args.indices
        else:
            rng     = np.random.default_rng(args.seed)
            indices = rng.choice(total, size=args.n, replace=False).tolist()
        print(f'选取索引: {indices}，每条跑 {args.rolls} 次不同噪声')
        for idx in indices:
            d = json.loads(all_lines[idx])
            n = int(d['n_nodes'])
            enc = bert_tok(d['prompt'], max_length=MAX_BERT_LEN,
                           padding='max_length', truncation=True)
            records.append(dict(
                idx     = idx,
                n_nodes = n,
                text    = d['prompt'],
                adj_np  = np.array(d['adj_matrix'],       dtype=np.float32),
                mask_np = np.array(d['node_mask'],        dtype=np.float32),
                combo_gt= d['node_combo_ids'],
                ptok_np = np.array(enc['input_ids'],      dtype=np.int64),
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

    # ── θ₂ 批量推理：所有 (样本 × roll) 合并为一个大 batch ──────────────────
    K = args.rolls
    B = len(records)
    all_adj  = np.stack([rec['adj_np']  for rec in records for _ in range(K)])  # [B*K,40,40]
    all_mask = np.stack([rec['mask_np'] for rec in records for _ in range(K)])  # [B*K,40]
    all_ptok = np.stack([rec['ptok_np'] for rec in records for _ in range(K)])  # [B*K,T]
    all_pmsk = np.stack([rec['pmsk_np'] for rec in records for _ in range(K)])  # [B*K,T]
    all_seeds = [rec['idx'] + k * 1_000_000 for rec in records for k in range(K)]

    print(f'  [θ₂] 批量推理 {B}样本 × {K}rolls = {B*K} 条 ...', flush=True)
    with torch.no_grad():
        all_coords = sample_coords_batch(
            model2, diffusion,
            all_adj, all_mask, all_ptok, all_pmsk,
            device, sample_indices=all_seeds,
        )  # [B*K, 40, 2]

    for i, rec in enumerate(records):
        rec['rolls'] = [{'coords': all_coords[i * K + k]} for k in range(K)]

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

    # θ₃ 同样批量推理
    coords_b = np.stack([rec['rolls'][k]['coords'] for rec in records for k in range(K)])  # [B*K,40,2]
    with torch.no_grad():
        logits_all = model3(
            torch.from_numpy(coords_b.transpose(0, 2, 1)).to(device),       # [B*K,2,40]
            adj_matrix    = torch.from_numpy(all_adj).to(device),
            node_mask     = torch.from_numpy(all_mask).to(device),
            prompt_tokens = torch.from_numpy(all_ptok).to(device),
            prompt_mask   = torch.from_numpy(all_pmsk).long().to(device),
        )  # [B*K, 40, n_combos]
    combo_all = logits_all.argmax(dim=-1).cpu().numpy()  # [B*K, 40]

    for i, rec in enumerate(records):
        for k in range(K):
            rec['rolls'][k]['combo_ids'] = combo_all[i * K + k]

    del model3
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 绘图：每条样本单独保存一张图（1行 × K列，每列一次 roll 的渲染图）─────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    COL_W = [2.5] * K
    FIG_W = sum(COL_W) + 0.3
    FIG_H = 3.2   # 单行高度（留足标题+文字空间）

    for rec in records:
        n   = rec['n_nodes']
        adj = rec['adj_np']

        fig, axes = plt.subplots(
            1, K,
            figsize=(FIG_W, FIG_H),
            gridspec_kw={'width_ratios': COL_W},
            constrained_layout=True,
        )
        if K == 1:
            axes = [axes]

        for k, roll in enumerate(rec['rolls']):
            ax = axes[k]
            node_types = [id_to_combo.get(int(roll['combo_ids'][i]), ['other'])
                          for i in range(n)]
            try:
                render_to_ax(ax, roll['coords'][:n], adj[:n, :n], n, node_types)
            except Exception as e:
                ax.axis('off')
                ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                        fontsize=5, transform=ax.transAxes)
            ax.set_title(f'Roll {k + 1}', fontsize=7, pad=3)

        # 文本描述作为整张图的大标题
        wrapped = textwrap.fill(rec['text'], width=100)
        fig.suptitle(f"#{rec['idx']}  {wrapped}", fontsize=6,
                     ha='left', x=0.01, y=1.01, va='bottom')

        png_out = out_dir / f"{rec['idx']:06d}.png"
        fig.savefig(png_out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved: {png_out}')

    print(f'\n已保存 {len(records)} 张图 → {out_dir}')

    # 保存输入+推理结果到 JSONL，便于复现和分析
    jsonl_out = out_dir / 'result_input.jsonl'
    with open(jsonl_out, 'w', encoding='utf-8') as jf:
        for rec in records:
            for k, roll in enumerate(rec['rolls']):
                jf.write(json.dumps({
                    'idx':            rec['idx'],
                    'roll_k':         k,
                    'seed':           rec['idx'] + k * 1_000_000,
                    'noise_strategy': 'g = torch.Generator(); g.manual_seed(seed)',
                    'n_nodes':        rec['n_nodes'],
                    'text':           rec['text'],
                    'adj_matrix':     rec['adj_np'].tolist(),
                    'node_mask':      rec['mask_np'].tolist(),
                    'gt_combo_ids':   rec['combo_gt'],
                    'pred_coords':    roll['coords'][:rec['n_nodes']].tolist(),
                    'pred_combo_ids': roll['combo_ids'][:rec['n_nodes']].tolist(),
                }, ensure_ascii=False) + '\n')
    print(f'已保存: {jsonl_out}')





if __name__ == '__main__':
    main()
