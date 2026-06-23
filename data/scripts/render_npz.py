"""
从 NPZ 数据集抽样渲染房间平面图，左侧布局图，右侧文本描述。

Usage:
    python -m data.scripts.render_npz \
        --input  data/processed/node_diffusion_cross_att/graph_dataset_6k.npz \
        --vocab  node_diffusion_cross_att/type_combo_vocab_old.json \
        --out    outputs/render_npz \
        --n      100 \
        --seed   42
"""

import argparse
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np
from transformers import BertTokenizer

from node_diffusion_cross_att.render import (
    load_vocab, find_faces, vote_room_type, ROOM_COLORS, ROOM_LABELS,
)


def decode_prompt(tokens: np.ndarray, mask: np.ndarray, tokenizer) -> str:
    valid = tokens[mask.astype(bool)].tolist()
    # 去掉 [CLS] / [SEP] / [PAD]
    skip = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    valid = [t for t in valid if t not in skip]
    return tokenizer.decode(valid, skip_special_tokens=True)


def render_to_ax(ax, coords_list, adj_list, node_types, n):
    all_nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj_list[i][j] == 1:
                all_nbrs[i].append(j)

    faces = find_faces(coords_list, adj_list)
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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
                ha='center', va='center', fontsize=6, color='#222222', zorder=3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/processed/node_diffusion_cross_att/graph_dataset_6k.npz')
    p.add_argument('--vocab', default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    p.add_argument('--out',   default='outputs/render_npz')
    p.add_argument('--n',     type=int, default=100)
    p.add_argument('--seed',  type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_combo = load_vocab(Path(args.vocab))
    tokenizer   = BertTokenizer.from_pretrained(args.bert)

    print(f'读取: {args.input}')
    d = np.load(args.input)
    total = int(d['n_nodes'].shape[0])
    print(f'总条数: {total}')

    random.seed(args.seed)
    indices = random.sample(range(total), min(args.n, total))
    print(f'抽样: {len(indices)} 条 → {out_dir}')

    ok = err = 0
    for i, idx in enumerate(indices):
        n        = int(d['n_nodes'][idx])
        coords   = [(float(d['node_coords'][idx, j, 0]), float(d['node_coords'][idx, j, 1]))
                    for j in range(n)]
        adj      = d['adj_matrix'][idx].tolist()
        combo_ids = d['node_combo_ids'][idx].tolist()
        node_types = [id_to_combo.get(int(combo_ids[j]), ['other']) for j in range(n)]
        prompt   = decode_prompt(d['prompt_tokens'][idx], d['prompt_mask'][idx], tokenizer)

        out_path = out_dir / f'{i:05d}_npz{idx}.png'
        try:
            fig, (ax_plan, ax_text) = plt.subplots(
                1, 2, figsize=(10, 5),
                gridspec_kw={'width_ratios': [1, 1]},
            )
            fig.patch.set_facecolor('#FFFFFF')

            render_to_ax(ax_plan, coords, adj, node_types, n)
            ax_plan.set_title(f'npz idx={idx}  n_nodes={n}', fontsize=8, pad=4)

            ax_text.axis('off')
            ax_text.set_facecolor('#F5F5F5')
            ax_text.text(
                0.5, 0.5,
                textwrap.fill(prompt, width=48),
                ha='center', va='center',
                fontsize=9, wrap=True,
                transform=ax_text.transAxes,
                multialignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                          edgecolor='#CCCCCC', linewidth=0.8),
            )

            plt.tight_layout(pad=1.0)
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            ok += 1
        except Exception as e:
            err += 1
            print(f'  ERR [{i}] npz_idx={idx}: {e}')
            try:
                plt.close(fig)
            except Exception:
                pass

        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(indices)}  ok={ok} err={err}', flush=True)

    print(f'\n完成  ok={ok}  err={err}  → {out_dir}')


if __name__ == '__main__':
    main()
