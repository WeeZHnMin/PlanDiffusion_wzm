"""
Sample N records from jsonl, translate Chinese prompts to English,
save as a new jsonl file for use with English BERT.

Translation model: Helsinki-NLP/opus-mt-zh-en (MarianMT, ~300MB)
Downloaded automatically on first run.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--out",        type=Path, default=Path("data/jsonl/train_nodes_100_en.jsonl"))
    parser.add_argument("--model-dir",  type=Path, default=Path("models/opus-mt-zh-en"),
                        help="local path to save/load the translation model")
    parser.add_argument("--n-samples",  type=int,  default=100)
    parser.add_argument("--batch-size", type=int,  default=16)
    parser.add_argument("--seed",       type=int,  default=42)
    return parser.parse_args()


def load_model(model_dir: Path):
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    if model_dir.exists():
        print(f"loading translation model from {model_dir}")
        tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
        model     = MarianMTModel.from_pretrained(str(model_dir))
    else:
        print(f"downloading {model_name} → {model_dir}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(model_dir))
        model.save_pretrained(str(model_dir))
        print("saved.")
    return tokenizer, model


def translate_batch(texts, tokenizer, model, device, max_length=256):
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, max_new_tokens=max_length)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    args = parse_args()
    random.seed(args.seed)

    print("loading records...")
    all_records = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    all_records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    print(f"total={len(all_records)}, sampling {args.n_samples}")
    records = random.sample(all_records, min(args.n_samples, len(all_records)))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_dir)
    model = model.to(device)
    model.eval()

    print(f"translating {len(records)} prompts (batch_size={args.batch_size})...")
    prompts_zh = [r["prompt"] for r in records]
    prompts_en = []
    for i in range(0, len(prompts_zh), args.batch_size):
        batch = prompts_zh[i:i + args.batch_size]
        translated = translate_batch(batch, tokenizer, model, device)
        prompts_en.extend(translated)
        print(f"  {min(i + args.batch_size, len(prompts_zh))}/{len(prompts_zh)}")

    # show a few examples
    print("\n--- translation samples ---")
    for zh, en in zip(prompts_zh[:5], prompts_en[:5]):
        print(f"  ZH: {zh[:80]}")
        print(f"  EN: {en[:80]}")
        print()

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r, en in zip(records, prompts_en):
            rec = dict(r)
            rec["prompt_zh"] = r["prompt"]
            rec["prompt"]    = en
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved {len(records)} records → {args.out}")


if __name__ == "__main__":
    main()
