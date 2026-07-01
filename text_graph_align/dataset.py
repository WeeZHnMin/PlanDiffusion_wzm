"""
AlignDataset: 读取 room_type_clf build_npz 生成的 npz 格式。

复用 room_type_clf 数据集（已预计算 BERT text_hidden），
训练时只需跑 text_proj，不再调用 BERT。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AlignDataset(Dataset):
    def __init__(self, npz_path):
        npz_path = str(npz_path)
        data = np.load(npz_path)
        self.node_mask       = data['node_mask'].astype(np.float32)
        self.adj_matrix      = data['adj_matrix'].astype(np.float32)
        self.room_membership = data['room_membership'].astype(np.float32)
        self.text_idx        = data['text_idx'].astype(np.int64)
        self.text_attn_mask  = data['text_attn_mask']           # [U, T] bool

        text_hidden_path = npz_path.replace('.npz', '.text_hidden.npy')
        self.text_hidden = np.load(text_hidden_path)            # [U, T, 768] fp16
        print(f"AlignDataset: {len(self.node_mask)} 条  "
              f"unique_prompts={len(self.text_hidden)}  {npz_path}")

    def __len__(self):
        return len(self.node_mask)

    def __getitem__(self, idx):
        uid = self.text_idx[idx]
        return {
            'node_mask':       torch.from_numpy(self.node_mask[idx]),
            'adj_matrix':      torch.from_numpy(self.adj_matrix[idx]),
            'room_membership': torch.from_numpy(self.room_membership[idx]),
            'text_hidden':     torch.from_numpy(
                                   self.text_hidden[uid].astype(np.float32)),
            'text_attn_mask':  torch.from_numpy(
                                   self.text_attn_mask[uid].astype(np.float32)),
        }


def load_align_data(npz_path, batch_size, shuffle=True, num_workers=4):
    ds = AlignDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=(num_workers > 0))
    return ds, loader
