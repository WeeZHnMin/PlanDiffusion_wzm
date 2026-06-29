"""
原始批量推理（无 KV cache），从 6851e0c 恢复，用于对比实验。
"""

import torch

from llm_graph.infer_stage1 import (
    parse_sequence, has_triangle, node_degrees,
    MAX_NODES, BOS_ID, PAD_ID, EOS_ID, SEP_ID,
    N_START, NODE_START,
)


@torch.no_grad()
def generate_batch(model, prefix_list, device, max_new_tokens=200, temperature=1.0,
                   use_c1=True, use_c2=True, use_c3=True, use_c4=True, use_c5=True):
    """
    对一批前缀并行自回归生成，带完整约束（无 KV cache，每步全量 forward）。
    prefix_list : list of list[int]，各前缀长度可不同（动态 padding）
    返回        : list of list[int]，完整生成序列（含前缀）
    """
    B = len(prefix_list)
    NEG_INF = float('-inf')
    _raw   = model.module if hasattr(model, 'module') else model
    _VOCAB = _raw.config.vocab_size

    seqs        = [list(p) for p in prefix_list]
    finished    = [False] * B
    phases      = ['N_tok'] * B
    Ns          = [0]    * B
    parent_cnts = [0]    * B
    run_adjs    = [None] * B

    def _build_batch(indices):
        max_len = max(len(seqs[b]) for b in indices)
        ids, masks, pos_ids = [], [], []
        for b in indices:
            L   = len(seqs[b])
            pad = max_len - L
            ids.append([PAD_ID] * pad + seqs[b])
            masks.append([0] * pad + [1] * L)
            pos_ids.append([0] * pad + list(range(L)))
        return (torch.tensor(ids,     dtype=torch.long, device=device),
                torch.tensor(masks,   dtype=torch.long, device=device),
                torch.tensor(pos_ids, dtype=torch.long, device=device))

    def _sample(logits, mask):
        logits = logits.clone()
        logits[mask] = NEG_INF
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    for _ in range(max_new_tokens):
        active = [b for b in range(B) if not finished[b]]
        if not active:
            break

        ids, attn, pos = _build_batch(active)
        logits_all = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits[:, -1, :].float()

        edge_second = []

        for ai, b in enumerate(active):
            logits = logits_all[ai]
            if temperature != 1.0:
                logits = logits / temperature

            ph = phases[b]

            if ph == 'N_tok':
                mask = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                mask[N_START: N_START + MAX_NODES] = False
                if use_c5:
                    mask[N_START: N_START + 8] = True
                nid = _sample(logits, mask)
                Ns[b]       = nid - N_START + 1
                run_adjs[b] = [[0] * Ns[b] for _ in range(Ns[b])]
                phases[b]   = 'parents' if Ns[b] > 1 else 'edges'
                parent_cnts[b] = 0
                seqs[b].append(nid)

            elif ph == 'parents':
                mask = torch.ones(_VOCAB, dtype=torch.bool, device=device)
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

            elif ph == 'edges':
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

                if nid == EOS_ID:
                    seqs[b].append(nid)
                    finished[b] = True
                else:
                    seqs[b].append(nid)
                    edge_second.append((b, nid))

        if edge_second:
            sub_idx = [b for b, _ in edge_second]
            ids2, attn2, pos2 = _build_batch(sub_idx)
            logits2_all = model(input_ids=ids2, attention_mask=attn2, position_ids=pos2).logits[:, -1, :].float()

            for si, (b, first_tok) in enumerate(edge_second):
                first   = first_tok - NODE_START
                logits2 = logits2_all[si]
                if temperature != 1.0:
                    logits2 = logits2 / temperature

                mask2 = torch.ones(_VOCAB, dtype=torch.bool, device=device)
                for j in range(Ns[b]):
                    skip_tri = use_c3 and has_triangle(run_adjs[b], first, j)
                    if j != first and not run_adjs[b][first][j] and not skip_tri:
                        mask2[NODE_START + j] = False

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

    return seqs
