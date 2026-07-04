"""
AlignDataset: 读取 node_diffusion_room_tri/build_graph_npz.py 生成的 npz。

npz 中需包含：
  node_coords   [N, MAX_NODES, 2]   float32
  adj_matrix    [N, MAX_NODES, MAX_NODES]  uint8
  node_mask     [N, MAX_NODES]       int32
  prompt_tokens [N, MAX_TEXT_LEN]   int32   BERT token ids
  prompt_mask   [N, MAX_TEXT_LEN]   int32   1=有效 token
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AlignDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(str(npz_path))
        self.node_coords   = data['node_coords'].astype(np.float32)
        self.adj_matrix    = data['adj_matrix'].astype(np.float32)
        self.node_mask     = data['node_mask'].astype(np.float32)
        self.prompt_tokens = data['prompt_tokens'].astype(np.int64)
        self.prompt_mask   = data['prompt_mask'].astype(np.float32)
        print(f"AlignDataset: {len(self.node_mask)} 条  {npz_path}")

    def __len__(self):
        return len(self.node_mask)

    def __getitem__(self, idx):
        return {
            'node_coords':   torch.from_numpy(self.node_coords[idx]),
            'adj_matrix':    torch.from_numpy(self.adj_matrix[idx]),
            'node_mask':     torch.from_numpy(self.node_mask[idx]),
            'prompt_tokens': torch.from_numpy(self.prompt_tokens[idx]),
            'prompt_mask':   torch.from_numpy(self.prompt_mask[idx]),
        }


def load_align_data(npz_path, batch_size, shuffle=True, num_workers=4):
    ds     = AlignDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=(num_workers > 0))
    return ds, loader
