"""
从 bert-base-uncased 提取前 K 个词，构建 WordPiece tokenizer，
并生成与 node_diffusion 训练代码兼容的 vocab_config.json。

输出（两个位置各一份）：
  node_diffusion/unified_vocab_wp/
  data/processed/wordpiece_vocab_10k/
    wp_tokenizer.json  -- WordPiece 分词器（tokenizers 库格式）
    vocab_config.json  -- graph token 偏移配置
    vocab.txt          -- 每行一个 token

用法：
    python -m data.scripts.build_wordpiece_vocab
    python -m data.scripts.build_wordpiece_vocab --top-k 8000
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer


MAX_NODES = 40
N_TYPES   = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-vocab", default="models/bert-base-uncased/vocab.txt")
    p.add_argument("--top-k",     type=int, default=10000, help="使用 vocab 前 K 个词")
    p.add_argument("--out-dirs",  nargs="+", default=[
        "node_diffusion/unified_vocab_wp",
        "data/processed/wordpiece_vocab_10k",
    ])
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.src_vocab)
    tokens = [t.strip() for t in src.read_text(encoding="utf-8").splitlines() if t.strip()]
    tokens = tokens[: args.top_k]
    vocab  = {token: idx for idx, token in enumerate(tokens)}
    print(f"词表大小: {len(tokens)}  首: {tokens[0]}  末: {tokens[-1]}")

    unk = "[UNK]" if "[UNK]" in vocab else tokens[0]
    tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token=unk))
    tokenizer.normalizer    = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder       = WordPieceDecoder()

    g = len(tokens)
    config = {
        "wp_vocab_size":   g,
        "graph_offset":    g,
        "total_vocab_size": g + 6 + N_TYPES + MAX_NODES,
        "PAD_ID":          g + 0,
        "BOS_ID":          g + 1,
        "EOS_ID":          g + 2,
        "TOK_OPEN":        g + 3,
        "TOK_CLOSE":       g + 4,
        "TOK_BREAK":       g + 5,
        "TYPE_START":      g + 6,
        "NODE_START":      g + 6 + N_TYPES,
        "MAX_NODES":       MAX_NODES,
        "N_TYPES":         N_TYPES,
    }

    for out_str in args.out_dirs:
        out = Path(out_str)
        out.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out / "wp_tokenizer.json"))
        (out / "vocab_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        (out / "vocab.txt").write_text(
            "
".join(tokens) + "
", encoding="utf-8"
        )
        print(f"写出: {out}")

    sample = "The kitchen is on the left, connected to the bedrooms via a corridor."
    enc = tokenizer.encode(sample)
    print(f"
测试: "{sample}"")
    print(f"  tokens : {enc.tokens}")
    print(f"  长度   : {len(enc.ids)}")
    print(f"
vocab_config:")
    for k, v in config.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
