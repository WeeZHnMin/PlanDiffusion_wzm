"""
从 eval_infer.py 输出的 NPZ 渲染平面图。
每条样本保存一张图（1行 × K列，每列一次 roll）。
支持多进程并行加速（--workers）。

Usage (from project root):
    python -m node_diffusion_cross_att.render_infer \\
        --npz     outputs/eval/infer_all_5roll.npz \\
        --vocab   node_diffusion_cross_att/type_combo_vocab_old.json \\
        --out     outputs/visualize_gacha \\
        --n       0 \\
        --workers 8
"""

import argparse
import multiprocessing as mp
import os
import textwrap
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── 全局状态（每个 worker 进程独立持有）─────────────────────────────────────
_g: dict = {}


def _worker_init(vocab_path: str, npz_path: str, out_dir_str: str, K: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.titlesize':   7,
        'font.size':        7,
    })
    from node_diffusion_cross_att.render import load_vocab

    data = np.load(npz_path, mmap_mode='r')
    _g['id_to_combo']    = load_vocab(Path(vocab_path))
    _g['pred_coords']    = data['pred_coords']     # [N, K, 40, 2]
    _g['pred_combo_ids'] = data['pred_combo_ids']  # [N, K, 40]
    _g['gt_adj']         = data['gt_adj']           # [N, 40, 40]
    _g['gt_n_nodes']     = data['gt_n_nodes']       # [N]
    _g['sample_indices'] = data['sample_indices'] if 'sample_indices' in data.files else None
    _g['out_dir']        = Path(out_dir_str)
    _g['K']              = K
    _g['COL_W']          = [2.5] * K
    _g['FIG_W']          = 2.5 * K + 0.3
    _g['FIG_H']          = 3.2


def _render_one(i: int):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import Polygon as ShapelyPolygon
    from node_diffusion_cross_att.render import find_faces, vote_room_type, ROOM_COLORS, ROOM_LABELS

    id_to_combo    = _g['id_to_combo']
    pred_coords    = _g['pred_coords']
    pred_combo_ids = _g['pred_combo_ids']
    gt_adj         = _g['gt_adj']
    gt_n_nodes     = _g['gt_n_nodes']
    out_dir        = _g['out_dir']
    K              = _g['K']
    COL_W          = _g['COL_W']
    FIG_W          = _g['FIG_W']
    FIG_H          = _g['FIG_H']

    n   = int(gt_n_nodes[i])
    adj = np.array(gt_adj[i, :n, :n])   # mmap → contiguous copy for tolist()

    fig, axes = plt.subplots(
        1, K,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={'width_ratios': COL_W},
        constrained_layout=True,
    )
    if K == 1:
        axes = [axes]

    ok = err = 0
    for k in range(K):
        ax        = axes[k]
        coords    = np.array(pred_coords[i, k, :n])   # mmap → copy
        combo_ids = np.array(pred_combo_ids[i, k, :n])
        node_types = [id_to_combo.get(int(combo_ids[j]), ['other']) for j in range(n)]

        try:
            # ── inline render_to_ax ─────────────────────────────────────────
            coords_list = [(float(coords[r, 0]), float(coords[r, 1])) for r in range(n)]
            adj_list    = adj.tolist()

            all_nbrs: Dict[int, List[int]] = {r: [] for r in range(n)}
            for r in range(n):
                for c2 in range(n):
                    if r != c2 and adj_list[r][c2] == 1:
                        all_nbrs[r].append(c2)

            faces      = find_faces(coords_list, adj_list)
            face_types = [vote_room_type(f, node_types, all_nbrs) for f in faces]

            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            mn_x, mx_x = min(xs), max(xs)
            mn_y, mx_y = min(ys), max(ys)
            span   = max(mx_x - mn_x, mx_y - mn_y, 1.0)
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
                pts  = [norm(*coords_list[r]) for r in face]
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
            ok += 1
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                    fontsize=5, transform=ax.transAxes)
            err += 1

        ax.set_title(f'Roll {k + 1}', fontsize=7, pad=3)

    si = _g.get('sample_indices')
    file_idx = int(si[i]) if si is not None else i
    fig.suptitle(f'#{file_idx}  n_nodes={n}', fontsize=7,
                 ha='left', x=0.01, y=1.01, va='bottom')
    png_out = out_dir / f'{file_idx:06d}.png'
    fig.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return ok, err


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',     default='outputs/eval/infer_all_5roll.npz')
    p.add_argument('--vocab',   default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--out',     default='outputs/visualize_gacha')
    p.add_argument('--n',       type=int, default=0,  help='渲染前N条，0=全部')
    p.add_argument('--start',   type=int, default=0,  help='从第几条开始')
    p.add_argument('--workers', type=int, default=0,
                   help='进程数（0=自动，即 CPU 核心数）')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'读取: {args.npz}')
    data = np.load(args.npz, mmap_mode='r')
    gt_n_nodes = data['gt_n_nodes']
    K = int(data['rolls']) if 'rolls' in data else data['pred_coords'].shape[1]
    N = len(gt_n_nodes)

    end = (args.start + args.n) if args.n > 0 else N
    end = min(end, N)
    indices = list(range(args.start, end))

    n_workers = args.workers if args.workers > 0 else os.cpu_count() or 4
    n_workers = min(n_workers, len(indices))
    print(f'渲染 {len(indices)} 条（共 {N} 条），K={K}，进程数={n_workers}')

    total_ok = total_err = 0

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(args.vocab, args.npz, str(out_dir), K),
    ) as pool:
        for j, (ok, err) in enumerate(
            pool.imap_unordered(_render_one, indices, chunksize=4)
        ):
            total_ok  += ok
            total_err += err
            if (j + 1) % 200 == 0 or (j + 1) == len(indices):
                print(f'  {j+1}/{len(indices)}  ok={total_ok} err={total_err}', flush=True)

    print(f'\n完成  ok={total_ok} err={total_err} → {out_dir}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
