"""
从 HuggingFace 私有数据集仓库下载训练数据。

用法：
  python download_dataset.py --hf_token YOUR_TOKEN
  HF_TOKEN=YOUR_TOKEN python download_dataset.py
"""

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",  default="", help="HF token（也可用 HF_TOKEN 环境变量）")
    p.add_argument("--repo_id",   default="wzmmmm/node_diffusion_150k")
    p.add_argument("--filename",  default="graph_dataset.npz")
    p.add_argument("--out_dir",   default="data/processed/node_diffusion_cross_att")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise SystemExit("错误：请通过 --hf_token 或 HF_TOKEN 环境变量提供 token")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename

    print(f"仓库  : {args.repo_id}")
    print(f"文件  : {args.filename}")
    print(f"保存至: {out_path}")

    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(out_dir),
        force_download=False,   # 已存在则跳过
    )
    print(f"完成: {local}")


if __name__ == "__main__":
    main()
