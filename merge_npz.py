"""
合并两个 eval_infer.py 生成的 NPZ 文件。

用法：
    python merge_npz.py \
        --parts outputs/eval/infer_part0.npz outputs/eval/infer_part1.npz \
        --out   outputs/eval/infer_all_5roll.npz
"""

import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parts', nargs='+', required=True, help='待合并的 NPZ 文件列表（按顺序）')
    p.add_argument('--out',   required=True, help='输出 NPZ 路径')
    return p.parse_args()


def main():
    args = parse_args()
    parts = [np.load(p) for p in args.parts]

    # 按行拼接所有数组，标量字段取第一个
    keys_concat = ['sample_indices', 'seeds',
                   'gt_coords', 'gt_combo_ids', 'gt_adj', 'gt_mask', 'gt_n_nodes',
                   'pred_coords', 'pred_combo_ids']
    merged = {}
    for k in keys_concat:
        arrays = [p[k] for p in parts if k in p]
        if arrays:
            merged[k] = np.concatenate(arrays, axis=0)

    merged['rolls'] = parts[0]['rolls']   # 标量，所有 part 相同

    N = len(merged['gt_n_nodes'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **merged)
    print(f'合并完成：{N} 条样本 → {args.out}')


if __name__ == '__main__':
    main()
