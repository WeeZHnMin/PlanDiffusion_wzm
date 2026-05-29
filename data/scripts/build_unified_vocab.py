"""
构建统一词表：BPE文本词表 + 图结构token

步骤：
1. 从 final_graph_dataset.jsonl 提取所有提示词
2. 用 tokenizers 库训练 BPE，目标文本词表大小 ~12000
3. 把79个图结构token追加到BPE词表后面
4. 保存统一词表到 data/processed/unified_vocab/

输出：
  data/processed/unified_vocab/tokenizer.json   ← HuggingFace tokenizers格式
  data/processed/unified_vocab/vocab_config.json ← 图token偏移等配置
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Punctuation, Digits, UnicodeScripts
from tokenizers.normalizers import NFKC


# ── 旧词表的图token定义（来自type_combo_vocab_old.json）──────────
GRAPH_SPECIAL_TOKENS = [
    "<PAD>",       # 0  → 图PAD（在统一词表里也做PAD）
    "<BOS_G>",     # BOS_图
    "<EOS_G>",     # EOS_图
    "<OPEN>",      # TOK_OPEN  <
    "<CLOSE>",     # TOK_CLOSE >
    "<BREAK>",     # TOK_BREAK /
]

# 32个组合类型token（combo type 1~32）
COMBO_TYPE_TOKENS = [f"<TYPE_{i}>" for i in range(1, 33)]

# 40个节点位置token（node 1~40）
NODE_TOKENS = [f"<NODE_{i}>" for i in range(1, 41)]

ALL_GRAPH_TOKENS = GRAPH_SPECIAL_TOKENS + COMBO_TYPE_TOKENS + NODE_TOKENS
# 总计：6 + 32 + 40 = 78个图token（不含PAD独立算）


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path,
                        default=Path("data/jsonl/final_graph_dataset.jsonl"))
    parser.add_argument("--old-vocab", type=Path,
                        default=Path("data/processed/type_combo_vocab_old.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/unified_vocab"))
    parser.add_argument("--bpe-vocab-size", type=int, default=12000,
                        help="BPE文本词表大小（不含图token）")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="BPE合并的最低频率")
    return parser.parse_args()


def extract_prompts(jsonl_path: Path) -> list[str]:
    prompts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt", "").replace("\n", " ").replace("\r", " ").strip()
            if prompt:
                prompts.append(prompt)
    print(f"提取提示词: {len(prompts)} 条")
    return prompts


def train_bpe(prompts: list[str], vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.normalizer = NFKC()
    # 字符级分词：对中文按字符切，保留数字和标点
    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),
        UnicodeScripts(),
    ])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<PAD>", "<UNK>"],
        show_progress=True,
    )

    # 用内存中的字符串列表训练
    tokenizer.train_from_iterator(prompts, trainer=trainer)
    print(f"BPE训练完成，实际词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


def build_unified_vocab(tokenizer: Tokenizer, old_vocab_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载旧图token配置
    old_vocab = json.loads(old_vocab_path.read_text(encoding="utf-8"))

    bpe_vocab_size = tokenizer.get_vocab_size()

    # 图token从 bpe_vocab_size 开始偏移
    graph_offset = bpe_vocab_size

    # 建立图token映射（对应旧vocab的ID → 新统一ID）
    # 旧vocab: PAD=0, TYPE 1-32, TOK_OPEN=33, TOK_CLOSE=34, TOK_BREAK=35,
    #          BOS=36, EOS=37, NODE_OFFSET=38 (节点1=39...节点40=78)
    old_pad      = 0
    old_bos      = old_vocab["BOS_ID"]       # 36
    old_eos      = old_vocab["EOS_ID"]       # 37
    old_tok_open = old_vocab["TOK_OPEN"]     # 33
    old_tok_close= old_vocab["TOK_CLOSE"]    # 34
    old_tok_break= old_vocab["TOK_BREAK"]    # 35
    old_node_off = old_vocab["NODE_OFFSET"]  # 38
    old_max_nodes= old_vocab["MAX_NODES"]    # 40

    # 新统一词表中图token的ID：
    # graph_offset+0 : PAD_G  (也作为统一PAD)
    # graph_offset+1 : BOS_G
    # graph_offset+2 : EOS_G
    # graph_offset+3 : TOK_OPEN
    # graph_offset+4 : TOK_CLOSE
    # graph_offset+5 : TOK_BREAK
    # graph_offset+6 .. graph_offset+37 : TYPE_1 .. TYPE_32
    # graph_offset+38 .. graph_offset+77: NODE_1 .. NODE_40

    new_pad       = graph_offset + 0
    new_bos       = graph_offset + 1
    new_eos       = graph_offset + 2
    new_tok_open  = graph_offset + 3
    new_tok_close = graph_offset + 4
    new_tok_break = graph_offset + 5
    new_type_start= graph_offset + 6        # TYPE_1 = graph_offset+6
    new_node_start= graph_offset + 6 + 32   # NODE_1 = graph_offset+38

    total_vocab_size = graph_offset + 6 + 32 + 40  # BPE + 特殊 + 类型 + 节点

    # 旧ID → 新ID 的映射表
    old_to_new = {}
    old_to_new[old_pad]       = new_pad
    old_to_new[old_bos]       = new_bos
    old_to_new[old_eos]       = new_eos
    old_to_new[old_tok_open]  = new_tok_open
    old_to_new[old_tok_close] = new_tok_close
    old_to_new[old_tok_break] = new_tok_break
    for t_id in range(1, 33):                # TYPE 1-32
        old_to_new[t_id] = new_type_start + (t_id - 1)
    for n in range(1, old_max_nodes + 1):    # NODE 1-40
        old_node_id = old_node_off + n
        old_to_new[old_node_id] = new_node_start + (n - 1)

    # 保存tokenizer（BPE部分）
    tokenizer_path = output_dir / "bpe_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"BPE tokenizer保存 -> {tokenizer_path}")

    # 保存统一词表配置
    config = {
        "bpe_vocab_size": bpe_vocab_size,
        "graph_offset": graph_offset,
        "total_vocab_size": total_vocab_size,

        # 新统一ID
        "PAD_ID":       new_pad,
        "BOS_ID":       new_bos,
        "EOS_ID":       new_eos,
        "TOK_OPEN":     new_tok_open,
        "TOK_CLOSE":    new_tok_close,
        "TOK_BREAK":    new_tok_break,
        "TYPE_START":   new_type_start,   # TYPE_1的ID
        "NODE_START":   new_node_start,   # NODE_1的ID
        "MAX_NODES":    old_max_nodes,
        "N_TYPES":      32,

        # 旧ID → 新ID映射（用于重新编码现有npz数据）
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
    }
    config_path = output_dir / "vocab_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"词表配置保存 -> {config_path}")
    print(f"\n统一词表大小: {total_vocab_size}")
    print(f"  BPE文本部分: 0 ~ {bpe_vocab_size - 1}  ({bpe_vocab_size} 个)")
    print(f"  图特殊token: {new_pad} ~ {new_tok_break}  (PAD/BOS/EOS/OPEN/CLOSE/BREAK)")
    print(f"  组合类型:    {new_type_start} ~ {new_type_start+31}  (32个)")
    print(f"  节点token:   {new_node_start} ~ {new_node_start+39}  (40个)")


def main():
    args = parse_args()
    prompts = extract_prompts(args.jsonl)
    tokenizer = train_bpe(prompts, args.bpe_vocab_size, args.min_frequency)
    build_unified_vocab(tokenizer, args.old_vocab, args.output_dir)
    print("\n完成！")


if __name__ == "__main__":
    main()
