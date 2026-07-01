"""
RoomTypeDataset: 读取 build_npz.py 生成的 npz。

text_hidden / text_attn_mask 按 text_idx 索引，去重存储。
训练时只过 text_proj，不再调用 BERT。
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
    """返回 (num_types, combo_to_id)。num_types = N_TYPES+1，使 ID 1~N_TYPES 均合法。"""
    with open(path, encoding='utf-8') as f:
        v = json.load(f)
    return v['N_TYPES'] + 1, v['combo_to_id']


class RoomTypeDataset(Dataset):
    def __init__(self, npz_path):
        npz_path = str(npz_path)
        data = np.load(npz_path)
        self.node_mask       = data['node_mask'].astype(np.float32)
        self.adj_matrix      = data['adj_matrix'].astype(np.float32)
        self.room_membership = data['room_membership'].astype(np.float32)
        self.type_labels     = data['type_labels'].astype(np.int64)
        self.text_idx        = data['text_idx'].astype(np.int64)
        self.text_attn_mask  = data['text_attn_mask']                    # [U, T] bool

        # text_hidden 单独存为 .text_hidden.npy
        text_hidden_path = npz_path.replace('.npz', '.text_hidden.npy')
        self.text_hidden = np.load(text_hidden_path)                     # [U, T, 768] fp16
        print(f"RoomTypeDataset: {len(self.node_mask)} 条  "
              f"unique_prompts={len(self.text_hidden)}  {npz_path}")

    def __len__(self):
        return len(self.node_mask)

    def __getitem__(self, idx):
        uid = self.text_idx[idx]
        return {
            'node_mask':       torch.from_numpy(self.node_mask[idx]),
            'adj_matrix':      torch.from_numpy(self.adj_matrix[idx]),
            'room_membership': torch.from_numpy(self.room_membership[idx]),
            'type_labels':     torch.from_numpy(self.type_labels[idx]),
            'text_hidden':     torch.from_numpy(
                                   self.text_hidden[uid].astype(np.float32)),   # [T, 768]
            'text_attn_mask':  torch.from_numpy(
                                   self.text_attn_mask[uid].astype(np.float32)), # [T]
        }


def load_data(npz_path, batch_size, shuffle=True):
    dataset = RoomTypeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=4, pin_memory=True, persistent_workers=True,
                         drop_last=True)
    return dataset, loader
