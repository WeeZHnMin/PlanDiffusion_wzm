"""
基于 BERT 前 10k WordPiece 词表，追加84个图专用 token，
生成 llm_graph 训练所需的词表文件。

输出：
  llm_graph/vocab/wp_tokenizer.json  — WordPiece 分词器
  llm_graph/vocab/vocab.txt          — 文本词表
  llm_graph/vocab/vocab_config.json  — 完整配置（含图 token 偏移）

用法：
    python -m llm_graph.build_vocab
    python -m llm_graph.build_vocab --top-k 10000 --max-nodes 40
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-tokenizer", default="node_diffusion/unified_vocab_wp/wp_tokenizer.json")
    p.add_argument("--src-vocab",     default="node_diffusion/unified_vocab_wp/vocab.txt")
    p.add_argument("--top-k",         type=int, default=10000)
    p.add_argument("--max-nodes",     type=int, default=40)
    p.add_argument("--max-text-len",  type=int, default=128)
    p.add_argument("--max-seq-len",   type=int, default=384)
    p.add_argument("--out",           default="llm_graph/vocab")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = args.top_k  # graph token 起始偏移
    config = {
        "wp_vocab_size": args.top_k,
        "PAD_ID":        g + 0,
        "BOS_ID":        g + 1,
        "EOS_ID":        g + 2,
        "SEP_ID":        g + 3,
        "N_START":       g + 4,                        # 节点数 token（k=1~MAX_NODES）
        "NODE_START":    g + 4 + args.max_nodes,       # 节点 ID token（j=0~MAX_NODES-1）
        "MAX_NODES":     args.max_nodes,
        "MAX_TEXT_LEN":  args.max_text_len,
        "MAX_SEQ_LEN":   args.max_seq_len,
        "VOCAB_SIZE":    g + 4 + args.max_nodes * 2,
        "note":          "N=k -> N_START+(k-1), node_j -> NODE_START+j",
    }

    shutil.copy(args.src_tokenizer, out / "wp_tokenizer.json")
    shutil.copy(args.src_vocab,     out / "vocab.txt")
    (out / "vocab_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"词表大小: {args.top_k}  图 token: 84  总计: {config['VOCAB_SIZE']}")
    print(f"写出: {out}")
    for k, v in config.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
