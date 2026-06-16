"""
将本地数据集文件上传到 HuggingFace 私有数据集仓库。

用法：
  python upload_dataset.py --hf_token YOUR_TOKEN
  HF_TOKEN=YOUR_TOKEN python upload_dataset.py

  # 自定义路径
  python upload_dataset.py --hf_token YOUR_TOKEN \
      --src data/processed/node_diffusion_cross_att/graph_dataset.npz
"""

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token", default="", help="HF token（也可用 HF_TOKEN 环境变量）")
    p.add_argument("--repo_id",  default="wzmmmm/node_diffusion_150k")
    p.add_argument("--src",      default="data/processed/node_diffusion_cross_att/graph_dataset.npz",
                   help="本地文件路径")
    p.add_argument("--dest",     default="graph_dataset.npz",
                   help="上传到仓库中的文件名")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise SystemExit("错误：请通过 --hf_token 或 HF_TOKEN 环境变量提供 token")

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"错误：文件不存在 → {src}")

    size_gb = src.stat().st_size / 1e9
    print(f"仓库  : {args.repo_id}")
    print(f"文件  : {src}  ({size_gb:.2f} GB)")
    print(f"目标  : {args.dest}")
    print("上传中 ...")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=args.dest,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="upload graph_dataset.npz",
    )
    print("上传完成")


if __name__ == "__main__":
    main()
