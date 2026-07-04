"""
噪声灵敏度可视化实验
====================
验证 DDPM 中 "减去预测噪声 → 重建 x0" 这一步的实际可靠性。

实验设计：
  1. 取一条真实样本 x0（归一化节点坐标）
  2. 在几个 t 时间步做前向扩散：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
  3. 用真实噪声ε 还原 x0（理论上完全一致）
  4. 给ε 加一丢丢额外扰动 δ，再还原 x0，看偏差多大

理论推导：
  重建误差 = sqrt((1-ᾱ_t)/ᾱ_t) * δ
  当 t 大（高噪）时，这个系数很大 → 小的噪声预测误差 → 大的坐标误差
  当 t 小（低噪）时，系数小 → 影响可控

输出：每个 t 值一行，每行5列：
  Col1: GT x0  Col2: 加噪后 x_t  Col3: 真实ε还原  Col4: 小扰动  Col5: 大扰动

用法：
    python visualize_noise_sensitivity.py \\
        --data data/jsonl/test_graph_dataset_10k.jsonl \\
        --sample_idx 0 \\
        --out outputs/noise_sensitivity.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':   7,
    'axes.titlesize': 7,
})

# ── 余弦 schedule（与 diffusion.py 一致）─────────────────────────────────────

def make_cosine_schedule(T=1000):
    t = torch.arange(T + 1) / T
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:]          # ᾱ_0 … ᾱ_{T-1}

# ── 坐标归一化（复制自 eval_iou.center_at_origin）──────────────────────────

def center_at_origin(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = coords[mask > 0]
    if len(valid) == 0:
        return coords
    cx, cy = valid[:, 0].mean(), valid[:, 1].mean()
    out = coords.copy()
    out[mask > 0, 0] -= cx
    out[mask > 0, 1] -= cy
    return out

# ── 节点散点图 + 邻接边 ────────────────────────────────────────────────────

def draw_nodes(ax, coords, adj=None, n=None, title="", rmse=None, color='steelblue'):
    if n is None:
        n = len(coords)
    xy = coords[:n]
    if adj is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0.5:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                            color='#999', lw=0.5, zorder=0)
    ax.scatter(xy[:, 0], xy[:, 1], s=18, color=color, zorder=2, edgecolors='k', linewidths=0.3)
    ax.set_aspect('equal')
    ax.axis('off')
    if rmse is not None:
        ax.set_title(f"{title}\nRMSE={rmse:.2f}", fontsize=6)
    else:
        ax.set_title(title, fontsize=6)

# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='data/jsonl/test_graph_dataset_10k.jsonl')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--out',        default='outputs/noise_sensitivity.png')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── 读一条样本 ────────────────────────────────────────────────────────────
    with open(args.data) as f:
        for _ in range(args.sample_idx + 1):
            line = f.readline()
    rec = json.loads(line)
    n   = int(rec['n_nodes'])
    raw = np.array(rec['node_coords'][:n], dtype=np.float32)
    adj = np.array(rec['adj_matrix'], dtype=np.float32)[:n, :n]
    np.fill_diagonal(adj, 0)
    prompt = rec.get('prompt', '').replace('\n', ' ')[:80]

    x0 = center_at_origin(raw, np.ones(n, dtype=np.float32))   # [n, 2]，单位：像素尺度
    x0_t = torch.tensor(x0, dtype=torch.float32)               # [n, 2]

    # ── 噪声 schedule ─────────────────────────────────────────────────────────
    alphas_bar = make_cosine_schedule(T=1000)   # [1000]

    # 计算 t=1000 的理论放大系数（供参考）
    # error_factor(t) = sqrt((1-ᾱ_t)/ᾱ_t)

    # 要分析的时间步
    T_values = [100, 300, 500, 700, 900]

    # 三种扰动强度（δ 的标准差，相对于坐标的绝对像素单位）
    delta_sigmas = [0.0, 0.5, 2.0]    # 0 = 真实噪声（完美）

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    n_rows = len(T_values)
    n_cols = 2 + len(delta_sigmas)          # x0, x_t, 三种恢复

    col_labels = ['GT x₀', 'x_t (加噪)'] + [
        f'δ=0 (完美)' if s == 0 else f'δ~N(0,{s}²)'
        for s in delta_sigmas
    ]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.2 + 0.8),
        squeeze=False,
    )
    fig.suptitle(
        f'DDPM 噪声灵敏度实验  —  sample #{args.sample_idx}  n={n}\n'
        f'"{prompt}"\n'
        f'误差 = sqrt((1−ᾱₜ)/ᾱₜ) × δ_norm',
        fontsize=7, y=0.98
    )

    # 列标题
    for ci, lbl in enumerate(col_labels):
        axes[0, ci].annotate(
            lbl, xy=(0.5, 1.18), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=7, fontweight='bold',
            annotation_clip=False,
        )

    for ri, t in enumerate(T_values):
        ab = float(alphas_bar[t - 1])       # ᾱ_t  (1-indexed → 0-indexed -1)
        factor = math.sqrt((1 - ab) / max(ab, 1e-8))

        # 前向扩散：生成真实噪声
        eps_true = torch.randn_like(x0_t)
        x_t = math.sqrt(ab) * x0_t + math.sqrt(1 - ab) * eps_true

        # 行标签
        row_label = (
            f't={t}\n'
            f'ᾱ={ab:.4f}\n'
            f'err_factor={factor:.2f}'
        )

        # Col 0: GT x0
        draw_nodes(axes[ri, 0], x0, adj, n,
                   title=row_label, color='steelblue')
        if ri == 0:
            pass   # 列标题已在上方标注

        # Col 1: x_t（加噪）
        x_t_np = x_t.numpy()
        draw_nodes(axes[ri, 1], x_t_np, adj, n,
                   title=f'', color='#cc7722')

        # Cols 2+: 用 ε+δ 恢复 x0
        for ci, sigma in enumerate(delta_sigmas):
            if sigma == 0.0:
                eps_pred = eps_true
            else:
                delta = sigma * torch.randn_like(eps_true)
                eps_pred = eps_true + delta

            # 重建公式：x0_hat = (x_t - sqrt(1-ᾱ)*ε_pred) / sqrt(ᾱ)
            x0_hat = (x_t - math.sqrt(1 - ab) * eps_pred) / max(math.sqrt(ab), 1e-4)
            x0_hat_np = x0_hat.numpy()

            rmse = float(np.sqrt(np.mean((x0_hat_np[:n] - x0[:n]) ** 2)))
            # 理论误差（如果 delta 是高斯）
            if sigma > 0:
                theoretic = factor * sigma * math.sqrt(2)   # RMS of 2D gaussian
                title_extra = f'理论≈{theoretic:.2f}'
            else:
                title_extra = '(应=0)'

            draw_nodes(axes[ri, 2 + ci], x0_hat_np, adj, n,
                       title=title_extra,
                       rmse=rmse,
                       color='#2ca02c' if sigma == 0 else '#d62728')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"保存至 {args.out}")

    # ── 打印数值表格 ───────────────────────────────────────────────────────────
    print("\n=== 噪声灵敏度数值汇总 ===")
    print(f"{'t':>5} {'ᾱ_t':>8} {'err_factor':>12}", end='')
    for s in delta_sigmas[1:]:
        print(f"  {'RMSE(δ='+str(s)+')':>14}", end='')
    print()
    alphas_bar_t = make_cosine_schedule(T=1000)
    for t in T_values:
        ab = float(alphas_bar_t[t - 1])
        factor = math.sqrt((1 - ab) / max(ab, 1e-8))
        row = f"{t:>5} {ab:>8.4f} {factor:>12.3f}"
        for s in delta_sigmas[1:]:
            theoretic = factor * s * math.sqrt(2)
            row += f"  {'~'+str(round(theoretic,2)):>14}"
        print(row)


if __name__ == '__main__':
    main()
