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

  # 批次并行推理（默认 batch_size=16）
  python -m llm_graph.infer_batch --ckpt ... --out ... --batch_size 32
"""

import argparse
import os
import numpy as np
import torch

from llm_graph.infer_stage1 import (
    load_model, load_dataset, get_prefix_and_gt,
    parse_sequence, has_triangle, node_degrees,
    MAX_NODES, BOS_ID, PAD_ID, EOS_ID, SEP_ID,
    N_START, NODE_START, VOCAB_SIZE,
)

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260614_155601/latest.pt')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree_test_10k.npz')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--out',         default='data/processed/node_diffusion_cross_att/gen_adj_test.npz')
    p.add_argument('--max_samples', type=int, default=0, help='最多处理条数，0=全量')
    p.add_argument('--batch_size',  type=int, default=16, help='并行推理批次大小')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def decode_bpe_text(token_ids: list, tokenizer) -> str:
    ids = [t for t in token_ids if t < 10000]
    return tokenizer.decode(ids)


# ── 批次自回归生成 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(model, prefix_list, device, max_new_tokens=200, temperature=1.0,
                   use_c1=True, use_c2=True, use_c3=True, use_c4=True, use_c5=True):
    """
    对一批前缀并行自回归生成，带完整约束。
    prefix_list : list of list[int]，各前缀长度可不同（动态 padding）
    返回        : list of list[int]，完整生成序列（含前缀）
    """
    B = len(prefix_list)
    NEG_INF = float('-inf')

    # 每条序列的完整 token 列表（生成过程中动态追加）
    seqs         = [list(p) for p in prefix_list]
    finished     = [False] * B
    phases       = ['N_tok'] * B
    Ns           = [0]    * B
    parent_cnts  = [0]    * B
    run_adjs     = [None] * B

    def _build_batch(indices):
        """将 indices 中的序列 left-pad 到同一长度，返回 (input_ids, attn_mask)。"""
        max_len = max(len(seqs[b]) for b in indices)
        ids, masks = [], []
        for b in indices:
            pad = max_len - len(seqs[b])
            ids.append([PAD_ID] * pad + seqs[b])
            masks.append([0] * pad + [1] * len(seqs[b]))
        return (torch.tensor(ids,   dtype=torch.long, device=device),
                torch.tensor(masks, dtype=torch.long, device=device))

    def _sample(logits, mask):
        logits = logits.clone()
        logits[mask] = NEG_INF
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    for _ in range(max_new_tokens):
        active = [b for b in range(B) if not finished[b]]
        if not active:
            break

        # ── 主批次 forward ─────────────────────────────────────────────────────
        ids, attn = _build_batch(active)
        logits_all = model(input_ids=ids, attention_mask=attn).logits[:, -1, :].float()

        edge_second = []   # (active_pos, b, first_node_id) 需要第二次 forward 的边

        for ai, b in enumerate(active):
            logits = logits_all[ai]
            if temperature != 1.0:
                logits = logits / temperature

            ph = phases[b]

            # ── N_tok 阶段 ───────────────────────────────────────────────────
            if ph == 'N_tok':
                mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
                mask[N_START: N_START + MAX_NODES] = False
                if use_c5:
                    mask[N_START: N_START + 8] = True
                nid = _sample(logits, mask)
                Ns[b]       = nid - N_START + 1
                run_adjs[b] = [[0] * Ns[b] for _ in range(Ns[b])]
                phases[b]   = 'parents' if Ns[b] > 1 else 'edges'
                parent_cnts[b] = 0
                seqs[b].append(nid)

            # ── parents 阶段 ─────────────────────────────────────────────────
            elif ph == 'parents':
                mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
                if use_c1:
                    for j in range(min(parent_cnts[b] + 1, Ns[b])):
                        mask[NODE_START + j] = False
                else:
                    for j in range(Ns[b]):
                        mask[NODE_START + j] = False
                if not use_c2:
                    mask[SEP_ID] = False
                nid = _sample(logits, mask)

                if NODE_START <= nid < NODE_START + Ns[b]:
                    p = nid - NODE_START
                    k = parent_cnts[b] + 1
                    if 0 <= k < Ns[b]:
                        run_adjs[b][k][p] = run_adjs[b][p][k] = 1
                    parent_cnts[b] += 1
                seqs[b].append(nid)

                if use_c2 and parent_cnts[b] == Ns[b] - 1:
                    seqs[b].append(SEP_ID)
                    phases[b] = 'edges'
                elif parent_cnts[b] >= Ns[b]:
                    seqs[b].append(SEP_ID)
                    phases[b] = 'edges'
                elif nid == SEP_ID:
                    phases[b] = 'edges'

            # ── edges 阶段（第一个 token）───────────────────────────────────
            elif ph == 'edges':
                mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
                for j in range(Ns[b]):
                    mask[NODE_START + j] = False
                if use_c4 and all(d >= 2 for d in node_degrees(run_adjs[b])):
                    mask[EOS_ID] = False
                elif not use_c4:
                    mask[EOS_ID] = False
                nid = _sample(logits, mask)

                if nid == EOS_ID:
                    seqs[b].append(nid)
                    finished[b] = True
                else:
                    seqs[b].append(nid)           # 先追加第一个节点 token
                    edge_second.append((b, nid))  # 标记需要第二次 forward

        # ── edges 阶段：第二个 token 子批次 forward ────────────────────────
        if edge_second:
            sub_idx = [b for b, _ in edge_second]
            ids2, attn2 = _build_batch(sub_idx)
            logits2_all = model(input_ids=ids2, attention_mask=attn2).logits[:, -1, :].float()

            for si, (b, first_tok) in enumerate(edge_second):
                first  = first_tok - NODE_START
                logits2 = logits2_all[si]
                if temperature != 1.0:
                    logits2 = logits2 / temperature

                # C3：禁三角环
                mask2 = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
                for j in range(Ns[b]):
                    skip_tri = use_c3 and has_triangle(run_adjs[b], first, j)
                    if j != first and not run_adjs[b][first][j] and not skip_tri:
                        mask2[NODE_START + j] = False

                # fallback：全部会成环时接受三角环
                if mask2.all():
                    for j in range(Ns[b]):
                        if j != first and not run_adjs[b][first][j]:
                            mask2[NODE_START + j] = False

                if not mask2.all():
                    sec_tok = _sample(logits2, mask2)
                    sec = sec_tok - NODE_START
                    if 0 <= sec < Ns[b]:
                        run_adjs[b][first][sec] = run_adjs[b][sec][first] = 1
                    seqs[b].append(sec_tok)
                # else: first 已与所有节点相连，跳过

    return seqs


# ── 主函数 ────────────────────────────────────────────────────────────────────

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
    BS = args.batch_size
    print(f'处理样本数: {N}  batch_size: {BS}')

    adj_out    = np.zeros((N, MAX_NODES, MAX_NODES), dtype=np.uint8)
    mask_out   = np.zeros((N, MAX_NODES),            dtype=np.uint8)
    n_nodes_out= np.zeros(N,                          dtype=np.int32)
    valid_out  = np.zeros(N,                          dtype=bool)
    ptok_out   = np.zeros((N, MAX_BERT_LEN),          dtype=np.int64)
    pmask_out  = np.zeros((N, MAX_BERT_LEN),          dtype=np.float32)

    n_valid = 0
    for b_start in range(0, N, BS):
        b_end    = min(b_start + BS, N)
        indices  = list(range(b_start, b_end))

        # 准备前缀
        prefixes = []
        for i in indices:
            prefix, _ = get_prefix_and_gt(i, all_tokens, all_lengths, all_textlens)
            prefixes.append(prefix)

        # 批次自回归生成
        gen_seqs = generate_batch(model, prefixes, device,
                                  max_new_tokens=200,
                                  temperature=args.temperature)

        # 解析 + BERT 重编码
        for local_i, i in enumerate(indices):
            gen_seq = gen_seqs[local_i]
            parsed  = parse_sequence(gen_seq)

            if parsed['valid']:
                n = parsed['n_nodes']
                adj = np.array(parsed['adj'], dtype=np.uint8)
                adj_out[i, :n, :n] = adj
                mask_out[i, :n]    = 1
                n_nodes_out[i]     = n
                valid_out[i]       = True
                n_valid += 1

            text_ids = prefixes[local_i][:-1]
            text     = decode_bpe_text(text_ids, bpe_tok)
            enc = bert_tok(text, max_length=MAX_BERT_LEN,
                           padding='max_length', truncation=True)
            ptok_out[i]  = enc['input_ids']
            pmask_out[i] = enc['attention_mask']

        if b_end % 500 < BS or b_end == N:
            print(f'  [{b_end:5d}/{N}]  valid={n_valid}  valid_rate={n_valid/b_end:.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
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
