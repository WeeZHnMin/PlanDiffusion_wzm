import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import json
import os
import pickle

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, seed_torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--milestone', type=str, default='latest',
                   help='加载 model-{milestone}.pt，可用 latest / best')
    p.add_argument('--results',   default='./results/ours_v1')
    p.add_argument('--data_root', default='../data/chathousediffusion/chat_train')
    p.add_argument('--ddim_steps', type=int, nargs='+', default=[200],
                   help='One or more DDIM sampling step counts, e.g. --ddim_steps 50 200 500.')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.results, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    seed_torch()
    summary = []
    for ddim_steps in args.ddim_steps:
        params["diffusion_dict"]["sampling_timesteps"] = ddim_steps
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

        print(f"测试集共 {len(trainer.val_ds)} 条，milestone={args.milestone}, ddim_steps={ddim_steps}")
        micro_iou = trainer.val(load_model=args.milestone, output_suffix=f"ddim{ddim_steps}")
        summary.append({"milestone": args.milestone, "ddim_steps": ddim_steps, "micro_iou": float(micro_iou)})
        del trainer, diffusion, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.results, f"ddim_eval_{args.milestone}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"DDIM evaluation summary saved to {summary_path}")
