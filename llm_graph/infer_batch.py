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
      --ckpt checkpoints/llm_graph/stage2/20260614_155601/latest.pt \\
      --out  data/processed/node_diffusion_cross_att/gen_adj_test.npz

  # 批次并行推理（默认 batch_size=16）
  python -m llm_graph.infer_batch --ckpt ... --out ... --batch_size 32
"""

import argparse
import json
import os
import numpy as np
import torch

from llm_graph.infer_stage1 import (
    load_model,
    parse_sequence, has_triangle, node_degrees,
    MAX_NODES, BOS_ID, PAD_ID, EOS_ID, SEP_ID,
    N_START, NODE_START, VOCAB_SIZE,
    encode_text,
)

MAX_BERT_LEN = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260614_155601/latest.pt')
    p.add_argument('--data',        default='data/jsonl/test_graph_dataset_10k.jsonl')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--out',         default='data/processed/node_diffusion_cross_att/gen_adj_test.npz')
    p.add_argument('--max_samples', type=int, default=0, help='最多处理条数，0=全量')
    p.add_argument('--batch_size',  type=int, default=16, help='并行推理批次大小')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


# ── 批次自回归生成 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(model, prefix_list, device, max_new_tokens=200, temperature=1.0,
                   use_c1=True, use_c2=True, use_c3=True, use_c4=True, use_c5=True):
    """
    对一批前缀并行自回归生成，带完整约束，使用 KV cache 加速。
    每步每个样本恰好生成一个 token，KV cache 始终对齐。
    phases: N_tok → parents → force_sep → edges_first → edges_second → (循环)
    prefix_list : list of list[int]，各前缀长度可不同（右 padding）
    返回        : list of list[int]，完整生成序列（含前缀）
    """
    B = len(prefix_list)
    NEG_INF = float('-inf')
    _raw   = model.module if hasattr(model, 'module') else model
    _VOCAB = _raw.config.vocab_size
    MAX_LEN = max(len(p) for p in prefix_list) + max_new_tokens + 4

    # ── 右 padding buffer ─────────────────────────────────────────────────────
    ids_buf  = torch.full((B, MAX_LEN), PAD_ID, dtype=torch.long, device=device)
    attn_buf = torch.zeros((B, MAX_LEN),         dtype=torch.long, device=device)
    seq_lens = []
    for b, p in enumerate(prefix_list):
        L = len(p)
        ids_buf[b, :L]  = torch.tensor(p, dtype=torch.long, device=device)
        attn_buf[b, :L] = 1
        seq_lens.append(L)

    seqs          = [list(p) for p in prefix_list]
    finished      = [False] * B
    # phases: N_tok | parents | force_sep | edges_first | edges_second
    phases        = ['N_tok'] * B
    Ns            = [0]    * B
    parent_cnts   = [0]    * B
    run_adjs      = [None] * B
    pending_first = [None] * B   # edges_second 阶段：上一步选的第一个 edge 节点

    def _sample(logits, mask):
        logits = logits.clone()
        logits[mask] = NEG_INF
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    def _append(b, tok):
        pos = seq_lens[b]
        ids_buf[b, pos]  = tok
        attn_buf[b, pos] = 1
        seq_lens[b]      = pos + 1
        seqs[b].append(tok)

    # ── 前缀一次性 forward → 初始 KV cache ───────────────────────────────────
    max_pre = max(seq_lens)
    out0    = model(
        input_ids      = ids_buf[:, :max_pre],
        attention_mask = attn_buf[:, :max_pre],
        use_cache      = True,
    )
    past_kv = out0.past_key_values
    # 各样本前缀长度不同，取各自最后一个真实 token 的 logits
    init_logits = torch.stack([
        out0.logits[b, seq_lens[b] - 1, :].float() for b in range(B)
    ])   # [B, vocab]

    # ── 生成循环：每步每样本恰好 1 token ─────────────────────────────────────
    logits_all = init_logits
    for step in range(max_new_tokens):
        active = [b for b in range(B) if not finished[b]]
        if not active:
            break

        # ── 按 phase 决定每个样本的下一个 token ──────────────────────────────
        next_toks = [None] * B   # 将要 append 的 token；finished 样本填 PAD

        for b in active:
            logits = logits_all[b]
            if temperature != 1.0:
                logits = logits / temperature
            ph = phases[b]

            # N_tok
            if ph == 'N_tok':
                mask = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                mask[N_START: N_START + MAX_NODES] = False
                if use_c5:
                    mask[N_START: N_START + 8] = True
                nid = _sample(logits, mask)
                Ns[b]          = nid - N_START + 1
                run_adjs[b]    = [[0] * Ns[b] for _ in range(Ns[b])]
                phases[b]      = 'parents' if Ns[b] > 1 else 'edges_first'
                parent_cnts[b] = 0
                next_toks[b]   = nid

            # parents
            elif ph == 'parents':
                mask = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                if use_c1:
                    # C1：parent of node k 必须来自 [0, k-1]
                    # use_c2=False 时不受 Ns[b] 上限约束，可超过预测 N
                    max_p = parent_cnts[b] + 1 if not use_c2 else min(parent_cnts[b] + 1, Ns[b])
                    for j in range(max_p):
                        mask[NODE_START + j] = False
                else:
                    cap = MAX_NODES if not use_c2 else Ns[b]
                    for j in range(cap):
                        mask[NODE_START + j] = False
                if not use_c2:
                    mask[SEP_ID] = False   # 允许模型自己输出 SEP
                nid = _sample(logits, mask)
                if NODE_START <= nid < NODE_START + MAX_NODES:
                    p = nid - NODE_START
                    k = parent_cnts[b] + 1
                    # 动态扩展邻接矩阵（use_c2=False 时节点数可超过 Ns[b]）
                    if k >= len(run_adjs[b]):
                        new_n = k + 1
                        new_adj = [[0] * new_n for _ in range(new_n)]
                        for r in range(len(run_adjs[b])):
                            for c in range(len(run_adjs[b])):
                                new_adj[r][c] = run_adjs[b][r][c]
                        run_adjs[b] = new_adj
                        Ns[b] = new_n
                    if 0 <= p < k:
                        run_adjs[b][k][p] = run_adjs[b][p][k] = 1
                    parent_cnts[b] += 1
                # 判断是否需要强制 SEP
                if use_c2 and parent_cnts[b] == Ns[b] - 1:
                    phases[b] = 'force_sep'
                elif (not use_c2 and nid == SEP_ID) or \
                     (use_c2 and (parent_cnts[b] >= Ns[b] or nid == SEP_ID)):
                    phases[b] = 'edges_first'
                    if nid == SEP_ID:
                        next_toks[b] = nid
                        continue
                next_toks[b] = nid

            # force_sep：强制输出 SEP，不采样
            elif ph == 'force_sep':
                phases[b]    = 'edges_first'
                next_toks[b] = SEP_ID

            # edges_first：选第一个 edge 节点或 EOS
            elif ph == 'edges_first':
                mask = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                degrees = node_degrees(run_adjs[b])
                c4_ok   = all(d >= 2 for d in degrees)
                if use_c4 and not c4_ok:
                    for j in range(Ns[b]):
                        if degrees[j] < 2:
                            mask[NODE_START + j] = False
                else:
                    for j in range(Ns[b]):
                        mask[NODE_START + j] = False
                    if c4_ok or not use_c4:
                        mask[EOS_ID] = False
                nid = _sample(logits, mask)
                next_toks[b] = nid
                if nid == EOS_ID:
                    finished[b] = True
                else:
                    pending_first[b] = nid - NODE_START
                    phases[b]        = 'edges_second'

            # edges_second：用 C3 约束选第二个节点
            elif ph == 'edges_second':
                first = pending_first[b]
                mask2 = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                for j in range(Ns[b]):
                    skip_tri = use_c3 and has_triangle(run_adjs[b], first, j)
                    if j != first and not run_adjs[b][first][j] and not skip_tri:
                        mask2[NODE_START + j] = False
                if mask2.all():   # fallback：全是三角时放开 C3
                    for j in range(Ns[b]):
                        if j != first and not run_adjs[b][first][j]:
                            mask2[NODE_START + j] = False
                if not mask2.all():
                    sec_tok = _sample(logits, mask2)
                    sec = sec_tok - NODE_START
                    if 0 <= sec < Ns[b]:
                        run_adjs[b][first][sec] = run_adjs[b][sec][first] = 1
                    next_toks[b] = sec_tok
                else:
                    next_toks[b] = PAD_ID   # first 已全连，跳过（罕见）
                phases[b] = 'edges_first'

        # ── 追加 token 并推进 KV cache ────────────────────────────────────────
        for b in range(B):
            tok = next_toks[b]
            if tok is None:
                tok = PAD_ID   # finished 样本填 PAD 保持 buffer 对齐
            _append(b, tok)

        cur_max  = max(seq_lens)
        new_toks = ids_buf[:, cur_max - 1: cur_max]   # [B, 1]，刚追加的 token
        out = model(
            input_ids       = new_toks,
            attention_mask  = attn_buf[:, :cur_max],
            past_key_values = past_kv,
            use_cache       = True,
        )
        past_kv    = out.past_key_values
        logits_all = out.logits[:, -1, :].float()   # [B, vocab]

    return seqs


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)

    from transformers import BertTokenizer
    bert_tok = BertTokenizer.from_pretrained(args.bert)
    print(f'BERT tokenizer loaded from {args.bert}')

    # 读取 JSONL
    print(f'读取数据集: {args.data}')
    rows = []
    with open(args.data, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    N = len(rows)
    if args.max_samples > 0:
        N = min(N, args.max_samples)
        rows = rows[:N]
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
        b_end   = min(b_start + BS, N)
        indices = list(range(b_start, b_end))

        # 准备前缀：从 JSONL 的 prompt 字段编码
        prefixes = []
        texts    = []
        for i in indices:
            text = rows[i]['prompt']
            texts.append(text)
            prefix = encode_text(text, args.vocab) + [BOS_ID]
            prefixes.append(prefix)

        # 批次自回归生成
        gen_seqs = generate_batch(model, prefixes, device,
                                  max_new_tokens=200,
                                  temperature=args.temperature)

        # 解析 + BERT 编码
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

            enc = bert_tok(texts[local_i], max_length=MAX_BERT_LEN,
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
