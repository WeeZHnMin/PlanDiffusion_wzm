import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from arguments import DataTrainingArguments


def parse_args():
    p = argparse.ArgumentParser(description="Validate Tell2Design floorplan dataset loading.")
    p.add_argument(
        "--tokenizer",
        default="/home/wzm/PlanDiffusion_wzm/models/t5-v1_1-base",
        help="Tokenizer name or local path.",
    )
    p.add_argument("--data-dir", default="data", help="T5 data root containing floorplan/*.json.")
    p.add_argument("--split", choices=["train", "dev", "test"], default="train")
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-output-length", type=int, default=512)
    p.add_argument("--input-format", default="plain")
    p.add_argument("--output-format", default="floorplan")
    p.add_argument("--output-format-type", default="short")
    p.add_argument("--boundary-in-where", default="Encoder")
    p.add_argument("--exp", default="debug")
    p.add_argument("--show-rooms", type=int, default=3, help="How many rooms from the first sample to print.")
    return p.parse_args()


def expected_json_path(data_dir: str, split: str) -> Path:
    name = {
        "train": "floorplan_train.json",
        "dev": "floorplan_dev.json",
        "test": "floorplan_test.json",
    }[split]
    return Path(data_dir) / "floorplan" / name


def main():
    args = parse_args()
    os.environ.setdefault("T5_SKIP_RUNTIME_VERSION_CHECK", "1")
    json_path = expected_json_path(args.data_dir, args.split)
    if not json_path.exists():
        print(f"[error] expected dataset json not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import load_dataset
    except Exception as exc:
        print("[error] failed to import Tell2Design dataset code.", file=sys.stderr)
        print(f"detail: {exc}", file=sys.stderr)
        if "sacremoses" in str(exc):
            print("hint: install missing dependency with `pip install sacremoses`", file=sys.stderr)
        raise

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    data_args = DataTrainingArguments(
        datasets="floorplan",
        data_dir=args.data_dir,
        train_split="train",
        input_format=args.input_format,
        output_format=args.output_format,
        output_format_type=args.output_format_type,
        boundary_in_where=args.boundary_in_where,
        exp=args.exp,
        max_seq_length=args.max_input_length,
        max_output_seq_length=args.max_output_length,
        overwrite_cache=True,
    )

    ds = load_dataset(
        dataset_name="floorplan",
        data_args=data_args,
        tokenizer=tokenizer,
        split=args.split,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        shuffle=False,
        is_eval=False,
    )

    print(f"json_path = {json_path}")
    print(f"dataset_len = {len(ds)}")

    if len(ds) == 0:
        print("[warn] dataset loaded but is empty")
        return

    ex = ds.get_example(0)
    feat = ds[0]

    print(f"example_id = {ex.id}")
    print(f"n_rooms = {len(ex.rooms) if ex.rooms else 0}")
    print(f"boundary_token_count = {len(ex.boundary_tokens) if ex.boundary_tokens else 0}")
    print(f"input_nonpad = {sum(feat.attention_mask)}")
    print(f"label_nonpad = {sum(1 for x in feat.label_ids if x != 0)}")
    print(f"text_head = {' '.join(ex.tokens[:30]) if ex.tokens else ''}")
    print(f"boundary_tokens_head = {ex.boundary_tokens[:20] if ex.boundary_tokens else []}")

    rooms_preview = []
    for room in (ex.rooms or [])[:args.show_rooms]:
        rooms_preview.append({
            "type": room.type,
            "bbox": [room.x_min, room.y_min, room.x_max, room.y_max],
            "location": room.location,
            "relation": room.relation,
            "size": room.size,
            "aspect_ratio": room.aspect_ratio,
            "private": room.private,
        })
    print("rooms_preview =")
    print(json.dumps(rooms_preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
