"""
NodeDataset for node_diffusion_room（三流版）：
cond 中附带 room_membership [N, MAX_ROOMS] 和 adj_matrix [N, N]。

npz 中必须含有 room_membership 和 adj_matrix 字段（由 build_graph_npz.py 生成）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NodeDataset(Dataset):
    """
    Each sample returns:
      x    : FloatTensor [2, 40]   coordinates (x,y)
      cond : dict with
               node_mask       [40]              1=有效节点
               room_membership [40, MAX_ROOMS]   二值，节点-环隶属矩阵
               adj_matrix      [40, 40]          二值邻接矩阵
               prompt_tokens   [T]               BERT input_ids
               prompt_mask     [T]               1=有效token，0=PAD
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords        = d['node_coords'].astype(np.float32)
        self.node_mask     = d['node_mask'].astype(np.uint8)
        self.prompt_tokens = d['prompt_tokens'].astype(np.int64)
        self._prompt_mask  = d['prompt_mask'].astype(np.float32) \
                             if 'prompt_mask' in d else None
        self._prompt_lens  = d['prompt_lens'].astype(np.int32) \
                             if 'prompt_lens' in d else None

        for field in ('room_membership', 'adj_matrix'):
            if field not in d:
                raise KeyError(
                    f"npz '{npz_path}' 缺少 {field} 字段。\n"
                    "请重新运行 node_diffusion_room copy/build_graph_npz.py 生成新 npz。"
                )
        self.room_membership = d['room_membership'].astype(np.float32)  # [N, 40, MAX_ROOMS]
        self.adj_matrix      = d['adj_matrix'].astype(np.float32)       # [N, 40, 40]
        print(f"NodeDataset(TriStream adj+room+global): {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy()    # [2, 40]
        if self._prompt_mask is not None:
            prompt_mask = self._prompt_mask[idx]
        else:
            T = self.prompt_tokens.shape[1]
            l = int(self._prompt_lens[idx])
            prompt_mask = np.zeros(T, dtype=np.float32)
            prompt_mask[:l] = 1.0
        cond = {
            'node_mask':       self.node_mask[idx].astype(np.float32),
            'room_membership': self.room_membership[idx],
            'adj_matrix':      self.adj_matrix[idx],
            'prompt_tokens':   self.prompt_tokens[idx],
            'prompt_mask':     prompt_mask,
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


def load_node_data(npz_path_or_dataset, batch_size, shuffle=True, sampler=None):
    if isinstance(npz_path_or_dataset, NodeDataset):
        dataset = npz_path_or_dataset
    else:
        dataset = NodeDataset(npz_path_or_dataset)
    if sampler is not None:
        shuffle = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        sampler=sampler, num_workers=2, drop_last=True)
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1
