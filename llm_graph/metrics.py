"""
评估指标计算：按序列段分别计算准确率。

序列结构（stage2）：
  [text tokens] BOS_G [N_tok] [parent_1 ... parent_{N-1}] SEP [i1 j1 ...] EOS_G

vocab 常量（和 vocab_config.json 对应）：
  N_START   = 12004   N=k → 12003+k
  SEP_ID    = 12003
  NODE_START= 12044
  EOS_ID    = 12002
  BOS_ID    = 12001
"""

import torch


N_START    = 12004
SEP_ID     = 12003
NODE_START = 12044
EOS_ID     = 12002
BOS_ID     = 12001


def compute_metrics(logits: torch.Tensor,
                    labels: torch.Tensor,
                    text_lens: torch.Tensor) -> dict:
    """
    logits : (B, L-1, V)  — teacher forcing，预测 tokens[1:]
    labels : (B, L-1)     — tokens[1:]，-100 处忽略
    text_lens: (B,)       — 文本部分长度（含 BOS_G）

    返回各段准确率：
      acc_all    : 全部图 token 的准确率
      acc_N      : N_tok 预测准确率
      acc_parent : 父节点序列准确率
      acc_edge   : 补边 token 准确率
    """
    B, L, V = logits.shape
    pred = logits.argmax(-1)   # (B, L)
    valid = labels.ne(-100)    # (B, L)

    def seg_acc(mask):
        total = (mask & valid).sum().item()
        if total == 0:
            return float('nan')
        correct = ((pred == labels) & mask & valid).sum().item()
        return correct / total

    # ── 全部图 token ──────────────────────────────────────────────────
    # labels 里 -100 的位置已经是文本部分，所以 valid 就是图部分
    acc_all = seg_acc(valid)

    # ── 逐样本定位各段 ────────────────────────────────────────────────
    mask_N      = torch.zeros_like(valid)
    mask_parent = torch.zeros_like(valid)
    mask_edge   = torch.zeros_like(valid)

    for i in range(B):
        tl = int(text_lens[i])
        # labels 是 tokens[1:] 的 shift，位置 j 对应 tokens[j+1]
        # BOS_G 在 tokens[tl-1]，N_tok 在 tokens[tl]，即 labels[tl-1]

        n_pos    = tl - 1           # labels 里 N_tok 的位置（预测目标）
        # 找 SEP 位置（在 labels 里）
        row      = labels[i]
        sep_positions = (row == SEP_ID).nonzero(as_tuple=True)[0]
        eos_positions = (row == EOS_ID).nonzero(as_tuple=True)[0]

        if len(sep_positions) == 0 or len(eos_positions) == 0:
            continue

        sep_pos = int(sep_positions[0])
        eos_pos = int(eos_positions[0])

        # N_tok
        if 0 <= n_pos < L:
            mask_N[i, n_pos] = True

        # 父节点：n_pos+1 到 sep_pos-1
        if n_pos + 1 <= sep_pos - 1:
            mask_parent[i, n_pos + 1: sep_pos] = True

        # 补边：sep_pos+1 到 eos_pos-1
        if sep_pos + 1 <= eos_pos - 1:
            mask_edge[i, sep_pos + 1: eos_pos] = True

    return {
        'acc_all':    acc_all,
        'acc_N':      seg_acc(mask_N),
        'acc_parent': seg_acc(mask_parent),
        'acc_edge':   seg_acc(mask_edge),
    }
