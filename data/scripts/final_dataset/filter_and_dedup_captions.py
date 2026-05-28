import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove duplicate captions and those exceeding max token length."
    )
    parser.add_argument(
        "--input",
        default="data/jsonl/captions.jsonl",
        help="Input captions jsonl file.",
    )
    parser.add_argument(
        "--output",
        default="data/jsonl/captions_unique.jsonl",
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--tokenizer",
        default="models/bert-base-chinese",
        help="Path to BertTokenizer.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Discard captions whose token count exceeds this value.",
    )
    return parser.parse_args()


def load_records(path: Path):
    records = []
    caption_counts = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            caption = record.get("caption")
            if isinstance(caption, str):
                caption_counts[caption] += 1
    return records, caption_counts


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    records, caption_counts = load_records(input_path)

    kept = 0
    dropped_dup = 0
    dropped_long = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            caption = record.get("caption")
            if not isinstance(caption, str):
                dropped_dup += 1
                continue
            if caption_counts[caption] > 1:
                dropped_dup += 1
                continue
            n_tokens = len(tokenizer.encode(caption, add_special_tokens=True))
            if n_tokens > args.max_tokens:
                dropped_long += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"input records       : {len(records)}")
    print(f"kept                : {kept}")
    print(f"dropped (duplicate) : {dropped_dup}")
    print(f"dropped (>{args.max_tokens} tokens): {dropped_long}")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
