"""
FloorplanGraphDataset：从 graph_dataset.npz 加载数据。

每条样本返回：
  X          : (40, 32)   节点类型 one-hot（32种combo类型，padding行全0）
  E          : (40, 40, 2) 边 one-hot（0=无边，1=有边）
  node_mask  : (40,)      bool，True=有效节点
  prompt_tokens: (128,)   int64，BPE token IDs
  prompt_lens: int        文本实际长度
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class FloorplanGraphDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.adj        = d['adj_matrix'].astype(np.int32)       # (N, 40, 40)
        self.node_mask  = d['node_mask'].astype(np.bool_)         # (N, 40)
        self.combo_ids  = d['node_combo_ids'].astype(np.int32)   # (N, 40), 1-32=type, 0=pad
        self.ptokens    = d['prompt_tokens'].astype(np.int64)    # (N, 128)
        self.plens      = d['prompt_lens'].astype(np.int32)      # (N,)
        print(f'FloorplanGraphDataset: {len(self.adj)} samples from {npz_path}')

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        # 节点类型 one-hot：combo_id 1-32 → class 0-31；padding(0) → 全零行
        ids = self.combo_ids[idx]                                    # (40,)
        mask = self.node_mask[idx]                                   # (40,)
        # 有效节点: ids[i]-1 作为 class index（0-based）
        valid_ids = np.where(mask, ids - 1, 0).astype(np.int64)    # (40,), 0-based
        X = F.one_hot(torch.from_numpy(valid_ids), num_classes=32).float()  # (40, 32)
        X[~torch.from_numpy(mask)] = 0.0                            # padding 行清零

        # 边 one-hot：0=无边，1=有边
        adj = torch.from_numpy(self.adj[idx].astype(np.int64))     # (40, 40)
        E = F.one_hot(adj, num_classes=2).float()                  # (40, 40, 2)

        node_mask = torch.from_numpy(mask)                          # (40,) bool

        ptokens = torch.from_numpy(self.ptokens[idx])               # (128,)
        plen    = int(self.plens[idx])

        return X, E, node_mask, ptokens, plen


def make_loader(npz_path, batch_size, shuffle=True,
                num_workers=4, persistent_workers=True):
    ds = FloorplanGraphDataset(npz_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=(num_workers > 0),
                      prefetch_factor=(2 if num_workers > 0 else None),
                      persistent_workers=(persistent_workers and num_workers > 0))
