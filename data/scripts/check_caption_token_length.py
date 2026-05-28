import json
from pathlib import Path
from transformers import BertTokenizer

JSONL_PATH = Path("data/jsonl/captions_unique.jsonl")
MODEL_PATH = Path("models/bert-base-chinese")
OUTPUT_PATH = Path("data/jsonl/captions_over256.jsonl")
MAX_LEN = 256

tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))

over_limit = []
total = 0

with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        record = json.loads(line)
        caption = record.get("caption", "")
        tokens = tokenizer.encode(caption, add_special_tokens=True)
        n = len(tokens)
        if n > MAX_LEN:
            over_limit.append({**record, "token_len": n})

print(f"Total captions: {total}")
print(f"Over {MAX_LEN} tokens: {len(over_limit)}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in over_limit:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved to {OUTPUT_PATH}")
