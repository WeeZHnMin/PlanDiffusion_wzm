"""
批量推理：用 θ₁（LLM 图拓扑生成）对测试集所有文本生成邻接矩阵，
同时用 BERT 重新编码文本，输出供 θ₂（节点坐标扩散）使用的 npz。

输出 npz 包含：
  adj_matrix    [N, 40, 40]  生成的邻接矩阵（0/1）
  node_mask     [N, 40]      有效节点掩码
  n_nodes       [N]          每条样本节点数
  valid         [N]          生成是否合法
  prompt_tokens [N, T]       BERT input_ids
  prompt_mask   [N, T]       BERT attention_mask

用法：
  python -m llm_graph.infer_batch \\
      --ckpt checkpoints/llm_graph/stage2/20260603_045254/best.pt \\
      --out  data/processed/node_diffusion_cross_att/gen_adj_test.npz

  # 只跑前100条调试
  python -m llm_graph.infer_batch --ckpt ... --out ... --max_samples 100
"""

import argparse
import numpy as np
import torch

from llm_graph.infer_stage1 import (
    load_model, load_dataset, get_prefix_and_gt,
    generate, parse_sequence, MAX_NODES, BOS_ID,
)

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260603_045254/best.pt')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree_test_10k.npz')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--bert',        default='models/bert-base-uncased',
                   help='BERT 模型路径或名称，用于重新编码文本')
    p.add_argument('--out',         default='data/processed/node_diffusion_cross_att/gen_adj_test.npz')
    p.add_argument('--max_samples', type=int, default=0,
                   help='最多处理条数，0=全量')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def decode_bpe_text(token_ids: list, tokenizer) -> str:
    """将 BPE token ID 序列解码回原始文本字符串（去除 BOS_G）。"""
    ids = [t for t in token_ids if t < 10000]  # 只保留文本 token，过滤图结构 token
    return tokenizer.decode(ids)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)
    all_tokens, all_lengths, all_textlens = load_dataset(args.data)

    from tokenizers import Tokenizer
    bpe_tok = Tokenizer.from_file(args.vocab)

    from transformers import BertTokenizer
    bert_tok = BertTokenizer.from_pretrained(args.bert)
    print(f'BERT tokenizer loaded from {args.bert}')

    N = len(all_tokens)
    if args.max_samples > 0:
        N = min(N, args.max_samples)
    print(f'处理样本数: {N}')

    adj_out    = np.zeros((N, MAX_NODES, MAX_NODES), dtype=np.uint8)
    mask_out   = np.zeros((N, MAX_NODES),            dtype=np.uint8)
    n_nodes_out= np.zeros(N,                          dtype=np.int32)
    valid_out  = np.zeros(N,                          dtype=bool)
    ptok_out   = np.zeros((N, MAX_BERT_LEN),          dtype=np.int64)
    pmask_out  = np.zeros((N, MAX_BERT_LEN),          dtype=np.float32)

    n_valid = 0
    for i in range(N):
        prefix, _ = get_prefix_and_gt(i, all_tokens, all_lengths, all_textlens)

        gen_seq = generate(model, prefix, device,
                           max_new_tokens=200, temperature=args.temperature)
        parsed  = parse_sequence(gen_seq)

        if parsed['valid']:
            n = parsed['n_nodes']
            adj = np.array(parsed['adj'], dtype=np.uint8)
            adj_out[i, :n, :n] = adj
            mask_out[i, :n]    = 1
            n_nodes_out[i]     = n
            valid_out[i]       = True
            n_valid += 1

        # 解码 BPE 文本（去掉末尾的 BOS_G）并用 BERT 重新编码
        text_ids = prefix[:-1]  # prefix = [...text tokens..., BOS_G]，去掉 BOS_G
        text     = decode_bpe_text(text_ids, bpe_tok)
        enc = bert_tok(
            text,
            max_length=MAX_BERT_LEN,
            padding='max_length',
            truncation=True,
        )
        ptok_out[i]  = enc['input_ids']
        pmask_out[i] = enc['attention_mask']

        if (i + 1) % 500 == 0 or i == N - 1:
            print(f'  [{i+1:5d}/{N}]  valid={n_valid}  valid_rate={n_valid/(i+1):.3f}')

    np.savez(
        args.out,
        adj_matrix    = adj_out,
        node_mask     = mask_out,
        n_nodes       = n_nodes_out,
        valid         = valid_out,
        prompt_tokens = ptok_out,
        prompt_mask   = pmask_out,
    )
    print(f'\n保存完成 → {args.out}')
    print(f'有效样本: {n_valid}/{N}  ({n_valid/N*100:.1f}%)')


if __name__ == '__main__':
    main()
