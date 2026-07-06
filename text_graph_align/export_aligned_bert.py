"""
把 bert_aligned_best.pt 的权重合并进 BERT config，
保存成标准 HuggingFace 目录，之后直接用 --bert 指向该目录即可。

用法：
  python -m text_graph_align.export_aligned_bert \
      --base_bert  models/bert-base-uncased \
      --aligned_pt checkpoints/text_graph_align/bert_aligned_best.pt \
      --output     checkpoints/text_graph_align/bert_aligned
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_bert',   default='models/bert-base-uncased')
    p.add_argument('--aligned_pt',  required=True,
                   help='bert_aligned_best.pt 路径')
    p.add_argument('--output',      required=True,
                   help='输出目录，保存合并后的完整 BERT')
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 加载结构
    model = BertModel.from_pretrained(args.base_bert)

    # 替换权重
    sd = torch.load(args.aligned_pt, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'missing  ({len(missing)}): {missing[:3]}')
    if unexpected:
        print(f'unexpected ({len(unexpected)}): {unexpected[:3]}')

    # 保存完整模型（config + weights）
    model.save_pretrained(str(out))

    # 复制 tokenizer 文件
    tokenizer = BertTokenizer.from_pretrained(args.base_bert)
    tokenizer.save_pretrained(str(out))

    print(f'saved -> {out}')
    print('使用方法: --bert', out)


if __name__ == '__main__':
    main()
