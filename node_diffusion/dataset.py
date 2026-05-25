import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NodeDataset(Dataset):
    """
    Loads preprocessed node-coordinate data from .npz cache
    (produced by data/scripts/preprocess_nodes.py).

    Each sample returns:
      x    : FloatTensor [2, 40]   coordinates (2 = x,y; 40 = max nodes)
      cond : dict with
               adj_matrix  [40, 40]
               node_mask   [40]
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords     = d['coords'].astype(np.float32)      # [N, 40, 2]
        self.adj_matrix = d['adj_matrix'].astype(np.float32)  # [N, 40, 40]
        self.node_mask  = d['node_mask'].astype(np.float32)   # [N, 40]
        print(f"NodeDataset: {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x   = self.coords[idx].T.copy()    # [2, 40]
        cond = {
            'adj_matrix': self.adj_matrix[idx],   # [40, 40]
            'node_mask':  self.node_mask[idx],     # [40]
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


def load_node_data(npz_path, batch_size, shuffle=True):
    dataset = NodeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=2, drop_last=True)
    while True:
        yield from loader
