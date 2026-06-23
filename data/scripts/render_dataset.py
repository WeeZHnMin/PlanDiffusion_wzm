"""
从 JSONL 数据集随机抽样渲染房间类型像素图。

Usage:
    python -m data.scripts.render_dataset \\
        --input  data/jsonl/final_graph_dataset_v3.jsonl \\
        --vocab  node_diffusion_cross_att/type_combo_vocab_old.json \\
        --out    outputs/render_v3 \\
        --n      5000
"""

import argparse
import json
import random
from pathlib import Path

from node_diffusion_cross_att.render import load_vocab, render_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--vocab',  default='node_diffusion_cross_att/type_combo_vocab_old.json')
    p.add_argument('--out',    default='outputs/render_v3')
    p.add_argument('--n',      type=int, default=5000, help='抽样数量，0=全部')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_combo = load_vocab(Path(args.vocab))

    print(f'读取: {args.input}')
    with open(args.input, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f'总条数: {len(rows)}')

    n = args.n if args.n > 0 else len(rows)
    random.seed(args.seed)
    samples = random.sample(rows, min(n, len(rows)))
    print(f'抽样: {len(samples)} 条 → {out_dir}')

    ok = err = 0
    for i, sample in enumerate(samples):
        name = sample.get('image', f"sample_{i}").replace('.png', '')
        out_path = out_dir / f"{name}.png"
        try:
            render_sample(sample, id_to_combo, out_path)
            ok += 1
        except Exception as e:
            err += 1
            print(f'  ERR [{i}] {name}: {e}')
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(samples)}  ok={ok} err={err}', flush=True)

    print(f'\n完成  ok={ok}  err={err}  → {out_dir}')


if __name__ == '__main__':
    main()
