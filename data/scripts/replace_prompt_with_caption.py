import json
from pathlib import Path

# 将 mapped_node_data.jsonl 里的英文 prompt 替换为中文 caption。
# 匹配方式: record["image"] == caption["file"]。
# 对于 captions.jsonl 里没有的数据，直接丢弃（不写入输出文件）。
# 输出: data/jsonl/mapped_node_data_zh.jsonl
DATA_DIR = Path(__file__).resolve().parent.parent
CAPTIONS_FILE = DATA_DIR / "jsonl" / "captions.jsonl"
SRC_FILE = DATA_DIR / "jsonl" / "mapped_node_data.jsonl"
OUT_FILE = DATA_DIR / "jsonl" / "mapped_node_data_zh.jsonl"


def main() -> None:
    # Build image -> caption mapping.
    captions = {}
    with CAPTIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("ok") and obj.get("caption"):
                captions[obj["file"]] = obj["caption"]

    print(f"loaded {len(captions)} captions")

    hit = 0
    miss = 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with SRC_FILE.open(encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = rec.get("image", "")
            zh = captions.get(img)

            if zh:
                rec["prompt"] = zh
                rec["prompt_lang"] = "zh"
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                hit += 1
            else:
                # Drop records without matched caption.
                miss += 1

    print(f"replaced: {hit}  dropped(no caption): {miss}  -> {OUT_FILE}")


if __name__ == "__main__":
    main()
