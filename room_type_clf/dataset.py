"""
RoomTypeDataset: 读取 build_npz.py 生成的 npz 文件。

字段：
  node_mask       [N, 40]            float32
  adj_matrix      [N, 40, 40]        float32
  room_membership [N, 40, MAX_ROOMS] float32
  prompt_tokens   [N, 192]           int64
  prompt_mask     [N, 192]           float32
  type_labels     [N, 40]            int64   padding=-1
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192
COMBO_VOCAB_PATH = str(Path(__file__).parent / 'type_combo_vocab_old.json')


def load_combo_vocab(path=COMBO_VOCAB_PATH):
    """返回 (num_types, combo_to_id)。num_types = N_TYPES（最大 combo_id）。"""
    with open(path, encoding='utf-8') as f:
        v = json.load(f)
    return v['N_TYPES'], v['combo_to_id']


class RoomTypeDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.node_mask       = data['node_mask'].astype(np.float32)
        self.adj_matrix      = data['adj_matrix'].astype(np.float32)
        self.room_membership = data['room_membership'].astype(np.float32)
        self.prompt_tokens   = data['prompt_tokens'].astype(np.int64)
        self.prompt_mask     = data['prompt_mask'].astype(np.float32)
        self.type_labels     = data['type_labels'].astype(np.int64)
        print(f"RoomTypeDataset: {len(self.node_mask)} 条  {npz_path}")

    def __len__(self):
        return len(self.node_mask)

    def __getitem__(self, idx):
        return {
            'node_mask':       torch.from_numpy(self.node_mask[idx]),
            'adj_matrix':      torch.from_numpy(self.adj_matrix[idx]),
            'room_membership': torch.from_numpy(self.room_membership[idx]),
            'prompt_tokens':   torch.from_numpy(self.prompt_tokens[idx]),
            'prompt_mask':     torch.from_numpy(self.prompt_mask[idx]),
            'type_labels':     torch.from_numpy(self.type_labels[idx]),
        }


def load_data(npz_path, batch_size, shuffle=True):
    dataset = RoomTypeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=4, pin_memory=True, persistent_workers=True,
                         drop_last=True)
    return dataset, loader
