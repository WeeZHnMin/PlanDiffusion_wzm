"""
AlignDataset: 读取 room_type_clf build_npz 生成的 npz。

npz 中需包含：
  text_input_ids  [U, T] int32   去重后的 token IDs
  text_attn_mask  [U, T] bool    1=有效 token
  text_idx        [N,]   int32   每条样本对应的 unique 索引
  node_mask       [N, 40]
  adj_matrix      [N, 40, 40]
  room_membership [N, 40, MAX_ROOMS]

文本编码器从零训练，不再依赖预计算 BERT。
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
        self.text_input_ids  = data['text_input_ids'].astype(np.int64)        # [U, T]
        self.text_attn_mask  = data['text_input_attn_mask'].astype(np.float32) # [U, T]
        print(f"AlignDataset: {len(self.node_mask)} 条  "
              f"unique_prompts={len(self.text_input_ids)}  {npz_path}")

    def __len__(self):
        return len(self.node_mask)

    def __getitem__(self, idx):
        uid = self.text_idx[idx]
        return {
            'node_mask':       torch.from_numpy(self.node_mask[idx]),
            'adj_matrix':      torch.from_numpy(self.adj_matrix[idx]),
            'room_membership': torch.from_numpy(self.room_membership[idx]),
            'input_ids':       torch.from_numpy(self.text_input_ids[uid]),
            'attn_mask':       torch.from_numpy(self.text_attn_mask[uid]),
        }


def load_align_data(npz_path, batch_size, shuffle=True, num_workers=4):
    ds = AlignDataset(npz_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=(num_workers > 0))
    return ds, loader
