"""
把 text_tower.pt 里的 BERT 权重转换成 HuggingFace 格式。

Usage:
    python -m text_graph_align.convert_to_hf \
        --src  checkpoints/align/text_tower.pt \
        --bert models/bert-base-uncased \
        --dst  models/bert-base-finetune
"""
import argparse
import shutil
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',  default='checkpoints/align/text_tower.pt')
    p.add_argument('--bert', default='models/bert-base-uncased')
    p.add_argument('--dst',  default='models/bert-base-finetune')
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # 复制 config / tokenizer 文件
    for fname in os.listdir(args.bert):
        if fname.endswith('.bin') or fname.endswith('.pt'):
            continue
        shutil.copy(os.path.join(args.bert, fname),
                    os.path.join(args.dst,  fname))
    print(f"已复制 config/tokenizer 文件到 {args.dst}")

    # 提取 bert.* 权重，去掉 bert. 前缀
    src = torch.load(args.src, map_location='cpu')
    tower = src['text_tower'] if 'text_tower' in src else src

    bert_state = {}
    for k, v in tower.items():
        if k.startswith('bert.'):
            new_k = k[len('bert.'):]
            bert_state[new_k] = v

    print(f"提取到 {len(bert_state)} 个 BERT 参数块")
    torch.save(bert_state, os.path.join(args.dst, 'pytorch_model.bin'))
    print(f"已保存到 {args.dst}/pytorch_model.bin")


if __name__ == '__main__':
    main()
