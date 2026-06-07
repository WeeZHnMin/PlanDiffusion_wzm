import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NodeDataset(Dataset):
    """
    Loads graph diffusion data from graph_dataset.npz.

    Each sample returns:
      x    : FloatTensor [2, 40]   coordinates (x,y)
      cond : dict with
               adj_matrix    [40, 40]  0/1邻接矩阵
               node_mask     [40]      1=有效节点
               node_types    [40]      节点类型 ID (1-32，0=padding)
               prompt_tokens [T]       BPE token IDs
               prompt_mask   [T]       1=有效token，0=PAD
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords        = d['node_coords'].astype(np.float32)   # [N, 40, 2]
        self.adj_matrix    = d['adj_matrix'].astype(np.uint8)      # [N, 40, 40] uint8 节省75%内存
        self.node_mask     = d['node_mask'].astype(np.uint8)       # [N, 40]     uint8 节省75%内存
        self.node_types    = d['node_combo_ids'].astype(np.int64)  # [N, 40]
        self.prompt_tokens = d['prompt_tokens'].astype(np.int64)   # [N, T]
        self.prompt_lens   = d['prompt_lens'].astype(np.int32)     # [N]
        # prompt_mask 按需在 __getitem__ 中生成，不预分配
        print(f"NodeDataset: {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy()    # [2, 40]
        T = self.prompt_tokens.shape[1]
        l = int(self.prompt_lens[idx])
        prompt_mask = np.zeros(T, dtype=np.float32)
        prompt_mask[:l] = 1.0
        cond = {
            'adj_matrix':    self.adj_matrix[idx].astype(np.float32),
            'node_mask':     self.node_mask[idx].astype(np.float32),
            'node_types':    self.node_types[idx],
            'prompt_tokens': self.prompt_tokens[idx],
            'prompt_mask':   prompt_mask,
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


def load_node_data(npz_path, batch_size, shuffle=True):
    dataset = NodeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=2, drop_last=True)
    while True:
        yield from loader
