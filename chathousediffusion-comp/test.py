import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import os
import pickle

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, seed_torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--milestone', type=str, default='latest',
                   help='加载 model-{milestone}.pt，可用 latest / best')
    p.add_argument('--results',   default='./results/ours_v1')
    p.add_argument('--data_root', default='../data/chathousediffusion/chat_train')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.results, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    params["diffusion_dict"]["sampling_timesteps"] = 200
    model     = Unet(**params["unet_dict"])
    diffusion = GaussianDiffusion(model, **params["diffusion_dict"])

    trainer = Trainer(
        diffusion,
        f"{args.data_root}/images",
        f"{args.data_root}/masks",
        f"{args.data_root}/texts",
        **params["trainer_dict"],
        results_folder=args.results,
        train_num_workers=0,
        mode="val",
        val_size=None,          # 全量测试集
    )

    seed_torch()
    print(f"测试集共 {len(trainer.val_ds)} 条，milestone={args.milestone}")
    trainer.val(load_model=args.milestone)
