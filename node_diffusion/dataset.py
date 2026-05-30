import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NodeDataset(Dataset):
    """
    Loads layout data from layout_dataset.npz.

    Each sample returns:
      x    : FloatTensor [2, 40]   coordinates (x,y)
      cond : dict with
               adj_matrix    [40, 40]  0/1邻接矩阵
               node_mask     [40]      1=有效节点
               prompt_tokens [T]       BPE token IDs
               prompt_mask   [T]       1=有效token，0=PAD
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords        = d['node_coords'].astype(np.float32)   # [N, 40, 2]
        self.adj_matrix    = d['adj_matrix'].astype(np.float32)    # [N, 40, 40]
        self.node_mask     = d['node_mask'].astype(np.float32)     # [N, 40]
        self.prompt_tokens = d['prompt_tokens'].astype(np.int64)   # [N, T]
        self.prompt_lens   = d['prompt_lens'].astype(np.int32)     # [N]
        T = self.prompt_tokens.shape[1]
        # prompt_mask: 1=有效，0=PAD
        self.prompt_mask = np.zeros((len(self.prompt_tokens), T), dtype=np.float32)
        for i, l in enumerate(self.prompt_lens):
            self.prompt_mask[i, :l] = 1.0
        print(f"NodeDataset: {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy()    # [2, 40]
        cond = {
            'adj_matrix':    self.adj_matrix[idx],    # [40, 40]
            'node_mask':     self.node_mask[idx],      # [40]
            'prompt_tokens': self.prompt_tokens[idx],  # [T]
            'prompt_mask':   self.prompt_mask[idx],    # [T]
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


def load_node_data(npz_path, batch_size, shuffle=True):
    dataset = NodeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=2, drop_last=True)
    while True:
        yield from loader
