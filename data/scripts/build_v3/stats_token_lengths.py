"""
统计 JSONL 中所有文本描述的 BERT token 长度分布。

输出：
  - p50 / p95 / p99 / max
  - 超过 128 / 192 / 256 token 的比例

用法：
  python -m data.scripts.stats_token_lengths
  python -m data.scripts.stats_token_lengths --jsonl data/jsonl/final_graph_dataset_v3.jsonl
"""

import argparse
import json
import time

import numpy as np
from transformers import BertTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', default='data/jsonl/final_graph_dataset_v3.jsonl')
    p.add_argument('--bert',  default='models/bert-base-uncased')
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert)

    lengths = []
    t0 = time.perf_counter()

    with open(args.jsonl, encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec    = json.loads(line)
            prompt = rec.get('prompt', '').replace('\n', ' ').strip()
            ids    = tokenizer(prompt, add_special_tokens=True)['input_ids']
            lengths.append(len(ids))

            if (line_no + 1) % 50000 == 0:
                print(f'  已处理 {line_no+1} 条  ({time.perf_counter()-t0:.1f}s)')

    lengths = np.array(lengths)
    total   = len(lengths)
    print(f'\n总样本数: {total}')
    print(f'耗时: {time.perf_counter()-t0:.1f}s\n')

    print('── 长度分布 ──────────────────────────────')
    for pct in [50, 90, 95, 99, 100]:
        print(f'  p{pct:3d}: {int(np.percentile(lengths, pct))} tokens')

    print('\n── 超过阈值的比例 ────────────────────────')
    for threshold in [128, 192, 256]:
        n_over = int((lengths > threshold).sum())
        print(f'  > {threshold:3d} tokens: {n_over:6d} / {total}  ({100*n_over/total:.2f}%)')


if __name__ == '__main__':
    main()
