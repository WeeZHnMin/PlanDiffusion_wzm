"""
把 mapped_node_data.jsonl 里的英文 prompt 替换成中文 caption。

链接方式: record["image"] == caption["file"]
没找到 caption 的记录保留原英文 prompt，并在 prompt_lang 字段标注。

输出: data/jsonl/mapped_node_data_zh.jsonl
"""

import json
from pathlib import Path

DATA_DIR     = Path(__file__).resolve().parent.parent
CAPTIONS_FILE = DATA_DIR / "jsonl" / "viz_50000_captions_multi.jsonl"
SRC_FILE      = DATA_DIR / "jsonl" / "mapped_node_data.jsonl"
OUT_FILE      = DATA_DIR / "jsonl" / "mapped_node_data_zh.jsonl"


def main():
    # 构建 image → caption 映射
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

    hit = miss = 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SRC_FILE.open(encoding="utf-8") as fin, \
         OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = rec.get("image", "")
            zh  = captions.get(img)
            if zh:
                rec["prompt"] = zh
                rec["prompt_lang"] = "zh"
                hit += 1
            else:
                rec["prompt_lang"] = "en"
                miss += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"替换成功: {hit}  保留英文: {miss}  -> {OUT_FILE}")


if __name__ == "__main__":
    main()
