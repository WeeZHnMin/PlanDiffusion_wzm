"""
将 aug_50k_graph.jsonl（坐标+类型）和 aug_50k_captions.jsonl（MiMo caption）
按图片文件名对齐，合并成最终训练用 JSONL。

用法：
    python merge_aug.py \\
        --graph    data/jsonl/aug_50k_graph.jsonl \\
        --captions data/jsonl/aug_50k_captions.jsonl \\
        --out      data/jsonl/aug_50k_final.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--graph',    default='data/jsonl/aug_50k_graph.jsonl')
    p.add_argument('--captions', default='data/jsonl/aug_50k_captions.jsonl')
    p.add_argument('--out',      default='data/jsonl/aug_50k_final.jsonl')
    return p.parse_args()


def main():
    args = parse_args()

    # 读取 captions，按文件名建索引
    print(f'读取 captions: {args.captions}')
    cap_map = {}
    with open(args.captions, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('ok') and row.get('file') and row.get('caption'):
                cap_map[row['file']] = row['caption']
    print(f'  有效 caption 数: {len(cap_map)}')

    # 读取 graph，按 image 字段匹配 caption
    print(f'读取 graph: {args.graph}')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = matched = skipped = 0
    with open(args.graph, encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            img_name = row.get('image', '')
            caption  = cap_map.get(img_name, '')
            if not caption:
                skipped += 1
                continue
            row['prompt'] = caption
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            matched += 1

    print(f'\n完成：总共 {total} 条，匹配 {matched} 条，跳过 {skipped} 条')
    print(f'输出 → {out_path}')


if __name__ == '__main__':
    main()
