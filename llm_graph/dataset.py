"""
生成树+补边格式的数据集（共用于 stage1 和 stage2）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TreeGraphDataset(Dataset):
    """
    读取 text_graph_tree.npz。

    tokens   : (N, MAX_SEQ_LEN)  int32
    lengths  : (N,)              int32  序列实际长度
    text_lens: (N,)              int32  文本部分长度（含 BOS_G）
               graph 部分从 text_lens[i] 开始（即 BOS_G 后面的 N_tok 开始）
    """

    def __init__(self, npz_path: str, stage: int = 2):
        """
        stage=1: 只用图序列（text_lens 以后的部分），忽略文本前缀
        stage=2: 完整 text+graph 序列
        """
        d = np.load(npz_path)
        self.tokens    = d['tokens'].astype(np.int32)     # (N, L)
        self.lengths   = d['lengths'].astype(np.int32)    # (N,)
        self.text_lens = d['text_lens'].astype(np.int32)  # (N,)
        self.stage     = stage
        print(f'TreeGraphDataset: {len(self.tokens)} samples  stage={stage}')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length   = int(self.lengths[idx])
        text_len = int(self.text_lens[idx])

        if self.stage == 1:
            # 只取图序列部分（从 BOS_G 开始）
            graph_start = text_len - 1  # BOS_G 位置
            seq = self.tokens[idx, graph_start:length]
            return (torch.tensor(seq, dtype=torch.long),
                    0)   # text_len=0，loss 从头算
        else:
            seq = self.tokens[idx, :length]
            return (torch.tensor(seq, dtype=torch.long),
                    text_len)


def collate_fn(batch, pad_id: int):
    seqs      = [b[0] for b in batch]
    text_lens = [b[1] for b in batch]
    max_len   = max(s.shape[0] for s in seqs)  # 动态 padding，节省显存

    tokens = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask   = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        tokens[i, :len(seq)] = seq
        mask[i, :len(seq)]   = 1

    return tokens, mask, torch.tensor(text_lens, dtype=torch.long)


def make_loader(npz_path, batch_size, stage=2, pad_id=10000,
                shuffle=True, num_workers=0):
    ds = TreeGraphDataset(npz_path, stage=stage)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
