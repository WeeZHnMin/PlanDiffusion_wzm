"""
Remove leading '中，' from prompt in a JSONL file.

Default target:
    data/jsonl/train_nodes.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/jsonl/train_nodes.jsonl"),
        help="Target JSONL file",
    )
    args = parser.parse_args()

    path = args.file
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    total = 0
    changed = 0
    out_lines = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            if isinstance(prompt, str) and prompt.startswith("中，"):
                obj["prompt"] = prompt[2:]
                changed += 1
            out_lines.append(json.dumps(obj, ensure_ascii=False))

    with path.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"total={total}")
    print(f"changed={changed}")
    print(f"updated={path}")


if __name__ == "__main__":
    main()
