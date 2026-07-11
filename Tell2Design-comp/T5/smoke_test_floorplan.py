import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("T5_SKIP_RUNTIME_VERSION_CHECK", "1")

import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from arguments import DataTrainingArguments
from datasets import load_dataset
from evaluate import evaluate, print_results


def parse_args():
    p = argparse.ArgumentParser(description="Minimal floorplan train/eval smoke test.")
    p.add_argument(
        "--tokenizer",
        default="/home/wzm/PlanDiffusion_wzm/models/t5-v1_1-base",
        help="Tokenizer path or model name.",
    )
    p.add_argument(
        "--model",
        default="/home/wzm/PlanDiffusion_wzm/models/t5-v1_1-base",
        help="Model path or model name.",
    )
    p.add_argument(
        "--source-data-dir",
        default="data",
        help="Directory containing floorplan/floorplan_train.json and floorplan_dev.json.",
    )
    p.add_argument(
        "--smoke-data-dir",
        default="smoke_data",
        help="Temporary data dir to store the sliced smoke-test json files.",
    )
    p.add_argument("--train-n", type=int, default=8, help="Number of train samples to keep.")
    p.add_argument("--dev-n", type=int, default=4, help="Number of dev samples to keep.")
    p.add_argument("--batch-size", type=int, default=2, help="Train/eval batch size.")
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-output-length", type=int, default=512)
    p.add_argument("--boundary-in-where", default="Encoder")
    p.add_argument("--exp", default="smoke")
    p.add_argument("--output-format-type", default="original")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the single training step.",
    )
    return p.parse_args()


def slice_json(src: Path, dst: Path, keep_n: int):
    payload = json.loads(src.read_text(encoding="utf-8"))
    sliced = payload[:keep_n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(sliced, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(sliced)


def build_data_args(args, data_dir: str):
    return DataTrainingArguments(
        datasets="floorplan",
        data_dir=data_dir,
        train_split="train",
        input_format="plain",
        output_format="floorplan",
        output_format_type=args.output_format_type,
        boundary_in_where=args.boundary_in_where,
        exp=args.exp,
        max_seq_length=args.max_input_length,
        max_output_seq_length=args.max_output_length,
        max_seq_length_eval=args.max_input_length,
        max_output_seq_length_eval=args.max_output_length,
        overwrite_cache=True,
    )


def feature_to_dict(feature):
    payload = {}
    for key, value in feature.__dict__.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def collate_features(features):
    batch = {}
    keys = ["input_ids", "attention_mask", "label_ids"]
    optional_keys = ["boundary_ids", "boundary_mask", "decoder_attention_mask"]
    for key in optional_keys:
        if key in features[0]:
            keys.append(key)
    for key in keys:
        values = [feat[key] for feat in features]
        batch[key] = torch.tensor(values, dtype=torch.long)
    return batch


def main():
    args = parse_args()

    source_floorplan_dir = Path(args.source_data_dir) / "floorplan"
    smoke_floorplan_dir = Path(args.smoke_data_dir) / "floorplan"
    kept_train = slice_json(
        source_floorplan_dir / "floorplan_train.json",
        smoke_floorplan_dir / "floorplan_train.json",
        args.train_n,
    )
    kept_dev = slice_json(
        source_floorplan_dir / "floorplan_dev.json",
        smoke_floorplan_dir / "floorplan_dev.json",
        args.dev_n,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    data_args = build_data_args(args, args.smoke_data_dir)

    train_ds = load_dataset(
        dataset_name="floorplan",
        data_args=data_args,
        tokenizer=tokenizer,
        split="train",
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        shuffle=False,
        is_eval=False,
    )
    dev_ds = load_dataset(
        dataset_name="floorplan",
        data_args=data_args,
        tokenizer=tokenizer,
        split="dev",
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        shuffle=False,
        is_eval=True,
    )

    assert len(train_ds) > 0, "train smoke dataset is empty"
    assert len(dev_ds) > 0, "dev smoke dataset is empty"

    model = T5ForConditionalGeneration.from_pretrained(args.model, config=T5Config.from_pretrained(args.model))
    device = torch.device(args.device)
    model.to(device)
    model.train()

    batch = collate_features([
        feature_to_dict(train_ds[0]),
        feature_to_dict(train_ds[1 if len(train_ds) > 1 else 0]),
    ])
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["label_ids"],
    )
    loss = outputs.loss
    loss.backward()

    print(f"smoke_train_len = {len(train_ds)}")
    print(f"smoke_dev_len = {len(dev_ds)}")
    print(f"single_step_loss = {float(loss.detach().cpu()):.6f}")
    print(f"batch_shape = {tuple(batch['input_ids'].shape)}")
    print(f"kept_train = {kept_train}, kept_dev = {kept_dev}")

    model.zero_grad(set_to_none=True)
    model.eval()
    metrics = evaluate(
        model=model,
        dataset_name="floorplan",
        data_args=data_args,
        tokenizer=tokenizer,
        split="dev",
        seed=0,
        gpu=0 if device.type == "cuda" else -1,
        batch_size=args.batch_size,
        output_dir=None,
    )
    print("eval_metrics =")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print_results(metrics)


if __name__ == "__main__":
    main()
