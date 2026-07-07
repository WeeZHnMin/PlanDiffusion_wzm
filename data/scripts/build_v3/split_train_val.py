"""
将 jsonl 数据集按 9:1 划分为训练集和验证集。

用法：
  python -m data.scripts.split_train_val --input data/jsonl/graphs_160k.jsonl
  python -m data.scripts.split_train_val --input data/jsonl/graphs_160k.jsonl --ratio 0.1 --seed 42
"""

import argparse
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='输入 jsonl 文件路径')
    p.add_argument('--ratio', type=float, default=0.1, help='验证集比例，默认 0.1')
    p.add_argument('--seed',  type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    stem    = in_path.stem
    out_dir = in_path.parent

    train_path = out_dir / f'{stem}_train.jsonl'
    val_path   = out_dir / f'{stem}_val.jsonl'

    lines = in_path.read_text(encoding='utf-8').splitlines()
    lines = [l for l in lines if l.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    n_val   = max(1, int(len(lines) * args.ratio))
    n_train = len(lines) - n_val

    train_path.write_text('\n'.join(lines[:n_train]) + '\n', encoding='utf-8')
    val_path.write_text(  '\n'.join(lines[n_train:]) + '\n', encoding='utf-8')

    print(f'总计：{len(lines)} 条')
    print(f'训练：{n_train} 条  →  {train_path}')
    print(f'验证：{n_val}   条  →  {val_path}')


if __name__ == '__main__':
    main()
