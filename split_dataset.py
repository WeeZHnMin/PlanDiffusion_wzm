"""
从 test_graph_dataset_30k.jsonl 中切出两份不重叠的数据：
  - 前 8500 条  + test_graph_dataset_10k.jsonl  → test_graph_dataset_18k5.jsonl
  - 接下来 18500 条                              → val_graph_dataset_18k5.jsonl
"""

import random
from pathlib import Path

SRC_30K  = Path("data/jsonl/test_graph_dataset_30k.jsonl")
SRC_10K  = Path("data/jsonl/test_graph_dataset_10k.jsonl")
OUT_TEST = Path("data/jsonl/test_graph_dataset_18k5.jsonl")
OUT_VAL  = Path("data/jsonl/val_graph_dataset_18k5.jsonl")

N_FROM_30K_FOR_TEST = 8500
N_FOR_VAL           = 18500

# 读取 30k 全部行
lines_30k = SRC_30K.read_text(encoding="utf-8").splitlines()
lines_30k = [l for l in lines_30k if l.strip()]
print(f"30k 文件实际行数: {len(lines_30k)}")

assert len(lines_30k) >= N_FROM_30K_FOR_TEST + N_FOR_VAL, \
    f"30k 文件行数不足 {N_FROM_30K_FOR_TEST + N_FOR_VAL}"

# 打乱后切分，保证不重叠
random.seed(42)
indices = list(range(len(lines_30k)))
random.shuffle(indices)

idx_test_extra = indices[:N_FROM_30K_FOR_TEST]
idx_val        = indices[N_FROM_30K_FOR_TEST: N_FROM_30K_FOR_TEST + N_FOR_VAL]

lines_test_extra = [lines_30k[i] for i in idx_test_extra]
lines_val        = [lines_30k[i] for i in idx_val]

# 读取 10k
lines_10k = SRC_10K.read_text(encoding="utf-8").splitlines()
lines_10k = [l for l in lines_10k if l.strip()]
print(f"10k 文件实际行数: {len(lines_10k)}")

# 合并 test
lines_test = lines_10k + lines_test_extra
print(f"test 合并后: {len(lines_test)} 条")
print(f"val:         {len(lines_val)} 条")

OUT_TEST.write_text("\n".join(lines_test) + "\n", encoding="utf-8")
OUT_VAL.write_text("\n".join(lines_val) + "\n", encoding="utf-8")

print(f"\n保存 → {OUT_TEST}")
print(f"保存 → {OUT_VAL}")
