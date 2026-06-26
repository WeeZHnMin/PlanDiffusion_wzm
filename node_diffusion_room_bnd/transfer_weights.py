"""
从 node_diffusion_room_tri 的 checkpoint 迁移权重到 node_diffusion_room_bnd 模型。

规则：
  - 名称和形状完全匹配的参数 → 直接复制
  - 新增的 bnd_attn 参数（layers.*.bnd_attn.*）→ 保留随机初始化

用法：
  python -m node_diffusion_room_bnd.transfer_weights \
      --src  checkpoints/node_diffusion_room_tri/latest.pt \
      --dst  checkpoints/node_diffusion_room_bnd/init_from_tri.pt
"""

import argparse
from pathlib import Path

import torch

from .model import NodeDiffusionTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",  default="checkpoints/node_diffusion_room_tri/latest.pt",
                   help="三流模型 checkpoint 路径")
    p.add_argument("--dst",  default="checkpoints/node_diffusion_room_bnd/init_from_tri.pt",
                   help="迁移后权重保存路径")
    p.add_argument("--bert", default="models/bert-base-uncased")
    p.add_argument("--model_channels", type=int, default=384)
    p.add_argument("--num_layers",     type=int, default=6)
    p.add_argument("--num_heads",      type=int, default=6)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu")

    # ── 加载源 checkpoint ──────────────────────────────────────────────────
    print(f"加载源模型: {args.src}")
    ckpt   = torch.load(args.src, map_location=device)
    src_sd = ckpt["model"]
    if any(k.startswith("module.") for k in src_sd):
        src_sd = {k[7:]: v for k, v in src_sd.items()}
    print(f"  源模型参数数量: {len(src_sd)}")

    # ── 构建新模型（四流）────────────────────────────────────────────────
    print("构建 bnd 模型（四流注意力）...")
    model = NodeDiffusionTransformer(
        model_channels=args.model_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        bert_name=args.bert,
    )
    dst_sd = model.state_dict()

    # ── 逐键迁移 ─────────────────────────────────────────────────────────
    copied   = []
    skipped  = []
    new_keys = []

    for key, val in dst_sd.items():
        if key in src_sd and src_sd[key].shape == val.shape:
            dst_sd[key] = src_sd[key]
            copied.append(key)
        elif key in src_sd:
            skipped.append((key, src_sd[key].shape, val.shape))
        else:
            new_keys.append(key)

    model.load_state_dict(dst_sd)

    print(f"\n迁移结果：")
    print(f"  已复制 : {len(copied)} 个参数")
    print(f"  随机初始化（新增）: {len(new_keys)} 个参数")
    if new_keys:
        for k in new_keys:
            print(f"    + {k}")
    if skipped:
        print(f"  形状不匹配（跳过）: {len(skipped)}")
        for k, s_sh, d_sh in skipped:
            print(f"    ! {k}  src={s_sh} dst={d_sh}")

    # ── 保存 ─────────────────────────────────────────────────────────────
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "step":  ckpt.get("step", 0),
    }, dst_path)
    print(f"\n保存 → {dst_path}  (step={ckpt.get('step', 0)})")


if __name__ == "__main__":
    main()
