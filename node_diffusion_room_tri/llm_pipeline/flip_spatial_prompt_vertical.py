"""Flip vertical spatial words before [SEP] in tri inference JSONL prompts.

This is a post-hoc repair for prompts produced by generate_spatial_prompts.py
when the vertical coordinate convention was inverted. Only the spatial prefix
before the first literal "[SEP]" is edited:

    top <-> bottom
    upper <-> lower

The original natural-language caption after [SEP] and all graph/coordinate
fields are left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_FILES = [
    "outputs/tri_from_llm_graph_ddim500.jsonl",
    "outputs/tri_from_llm_graph_ddpm1000.jsonl",
]

VERTICAL_WORD_RE = re.compile(r"(?<![A-Za-z])(top|bottom|upper|lower)(?![A-Za-z])")
SWAP = {
    "top": "bottom",
    "bottom": "top",
    "upper": "lower",
    "lower": "upper",
}


def swap_vertical_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group(1)
        return SWAP[word]

    return VERTICAL_WORD_RE.sub(repl, text)


def flip_prompt_prefix(prompt: str) -> tuple[str, bool, bool]:
    """Return (new_prompt, has_sep, changed)."""
    before, sep, after = prompt.partition("[SEP]")
    if not sep:
        return prompt, False, False

    flipped_before = swap_vertical_words(before)
    new_prompt = flipped_before + sep + after
    return new_prompt, True, new_prompt != prompt


def process_file(path: Path, dry_run: bool, backup: bool) -> dict[str, int]:
    stats = {
        "rows": 0,
        "changed": 0,
        "missing_prompt": 0,
        "missing_sep": 0,
        "json_errors": 0,
    }
    if not path.exists():
        raise FileNotFoundError(path)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".bak")

    with path.open(encoding="utf-8") as fin:
        fout = None if dry_run else tmp_path.open("w", encoding="utf-8")
        try:
            for line_no, line in enumerate(fin, start=1):
                raw = line.rstrip("\n")
                if not raw:
                    if fout is not None:
                        fout.write(line)
                    continue

                stats["rows"] += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    if fout is not None:
                        fout.write(line)
                    continue

                prompt = row.get("prompt")
                if not isinstance(prompt, str):
                    stats["missing_prompt"] += 1
                else:
                    new_prompt, has_sep, changed = flip_prompt_prefix(prompt)
                    if not has_sep:
                        stats["missing_sep"] += 1
                    if changed:
                        stats["changed"] += 1
                        row["prompt"] = new_prompt

                if fout is not None:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            if fout is not None:
                fout.close()

    if not dry_run:
        if backup:
            backup_path.write_bytes(path.read_bytes())
        tmp_path.replace(path)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=DEFAULT_FILES,
        help="JSONL files to repair in place.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report how many prompts would change.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy before replacing each JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = {k: 0 for k in ["rows", "changed", "missing_prompt", "missing_sep", "json_errors"]}

    for name in args.jsonl:
        path = Path(name)
        stats = process_file(path, dry_run=args.dry_run, backup=args.backup)
        for key, value in stats.items():
            total[key] += value
        mode = "dry-run" if args.dry_run else "updated"
        print(f"{mode}: {path}")
        print(
            "  rows={rows} changed={changed} missing_prompt={missing_prompt} "
            "missing_sep={missing_sep} json_errors={json_errors}".format(**stats)
        )

    print(
        "total: rows={rows} changed={changed} missing_prompt={missing_prompt} "
        "missing_sep={missing_sep} json_errors={json_errors}".format(**total)
    )


if __name__ == "__main__":
    main()
