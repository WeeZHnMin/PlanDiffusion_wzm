"""
Extract original JSONL rows referenced by a mapping file.

Default output:
    data/jsonl/mapped_source_records.jsonl
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MAPPING_FILE = DATA_DIR / "viz_50000" / "mapping.jsonl"
SRC_DIR = DATA_DIR / "Architext_v1" / "train_jsonl"
OUT_FILE = DATA_DIR / "jsonl" / "mapped_source_records.jsonl"


def main() -> None:
    if not MAPPING_FILE.exists():
        raise SystemExit(f"Missing mapping file: {MAPPING_FILE}")

    mapping = []
    wanted = {}
    with MAPPING_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping.append(row)
            src_file = row["source_file"]
            src_line = int(row["source_line"])
            wanted.setdefault(src_file, set()).add(src_line)

    found = {}
    for src_file, line_set in wanted.items():
        src_path = SRC_DIR / src_file
        if not src_path.exists():
            raise SystemExit(f"Missing source file: {src_path}")
        with src_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no in line_set:
                    found[(src_file, line_no)] = line.rstrip("\n")

    missing = []
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for row in mapping:
            key = (row["source_file"], int(row["source_line"]))
            line = found.get(key)
            if line is None:
                missing.append(key)
                continue
            src_obj = json.loads(line)
            out_obj = {
                k: src_obj[k]
                for k in ("prompt", "rooms", "adjacency", "coord_seq", "n_rooms", "n_tokens")
                if k in src_obj
            }
            out_obj["image"] = row["image"]
            out_obj["source_file"] = row["source_file"]
            out_obj["source_line"] = int(row["source_line"])
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    if missing:
        preview = ", ".join([f"{f}:{n}" for f, n in missing[:10]])
        raise SystemExit(f"Missing {len(missing)} mapped rows. Examples: {preview}")

    print(f"Wrote {len(mapping)} rows -> {OUT_FILE}")


if __name__ == "__main__":
    main()
