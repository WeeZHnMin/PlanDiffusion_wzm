"""
从对齐模型中提取 text_tower 权重并单独保存。

Usage:
    python -m text_graph_align.extract_text_tower \
        --src  checkpoints/align/align_latest.pt \
        --dst  checkpoints/align/text_tower.pt
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='checkpoints/align/align_latest.pt')
    p.add_argument('--dst', default='checkpoints/align/text_tower.pt')
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    full = ckpt['model']

    text_tower = {
        k[len('text_tower.'):]: v
        for k, v in full.items()
        if k.startswith('text_tower.')
    }

    print(f"提取到 {len(text_tower)} 个参数块：")
    for k, v in text_tower.items():
        print(f"  {k:60s}  {tuple(v.shape)}")

    torch.save({'step': ckpt.get('step', 0), 'text_tower': text_tower}, args.dst)
    print(f"\n已保存到 {args.dst}")


if __name__ == '__main__':
    main()
