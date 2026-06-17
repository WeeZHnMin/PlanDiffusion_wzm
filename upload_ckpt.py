"""
将本地权重和训练日志上传到 HuggingFace 模型仓库。

用法：
  python upload_ckpt.py --hf_token YOUR_TOKEN
  python upload_ckpt.py --hf_token YOUR_TOKEN --ckpt checkpoints/node_diffusion_cross_att/latest.pt
"""

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token", default="", help="HF token（也可用 HF_TOKEN 环境变量）")
    p.add_argument("--repo_id",  default="wzmmmm/plandiff-cross-att")
    p.add_argument("--ckpt",     default="checkpoints/node_diffusion_cross_att/latest.pt")
    p.add_argument("--log",      default="checkpoints/node_diffusion_cross_att/train_log.jsonl")
    p.add_argument("--step",     default="", help="commit message 里的步数标注（可选）")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise SystemExit("错误：请通过 --hf_token 或 HF_TOKEN 环境变量提供 token")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(args.repo_id, private=True, repo_type="model", exist_ok=True)

    step_tag = f" step={args.step}" if args.step else ""

    ckpt = Path(args.ckpt)
    if ckpt.exists():
        size_gb = ckpt.stat().st_size / 1e9
        print(f"上传权重: {ckpt}  ({size_gb:.2f} GB) ...")
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo="latest.pt",
            repo_id=args.repo_id,
            commit_message=f"upload latest.pt{step_tag}",
        )
        print("  权重上传完成")
    else:
        print(f"权重文件不存在，跳过: {ckpt}")

    log = Path(args.log)
    if log.exists():
        print(f"上传日志: {log} ...")
        api.upload_file(
            path_or_fileobj=str(log),
            path_in_repo="train_log.jsonl",
            repo_id=args.repo_id,
            commit_message=f"upload train_log.jsonl{step_tag}",
        )
        print("  日志上传完成")
    else:
        print(f"日志文件不存在，跳过: {log}")

    print(f"\n完成 → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
