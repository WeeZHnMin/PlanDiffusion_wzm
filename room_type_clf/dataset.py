"""
RoomTypeDataset: 从 jsonl 加载数据，输出节点类型分类所需的张量。

每条样本返回：
  node_mask        [N]          1=有效节点
  adj_matrix       [N, N]       二值邻接矩阵
  room_membership  [N, MAX_ROOMS] 节点-环隶属矩阵
  prompt_tokens    [T]          BERT input_ids
  prompt_mask      [T]          1=有效 token
  type_labels      [N]          节点类型 id（padding 节点=-1，忽略）
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .model import _assign_room_membership_single, MAX_ROOMS

MAX_NODES    = 40
MAX_TEXT_LEN = 192


COMBO_VOCAB_PATH = 'node_diffusion_room_tri/type_combo_vocab_old.json'


def load_combo_vocab(path=COMBO_VOCAB_PATH):
    """加载组合类型词表，返回 (num_types, combo_to_id)。
    combo_to_id: {'[1]': 1, '[2]': 2, ...}  ID 从 1 开始。
    num_types = N_TYPES（最大 combo_id，用于 nn.Embedding / CrossEntropyLoss）。
    """
    with open(path, encoding='utf-8') as f:
        v = json.load(f)
    return v['N_TYPES'], v['combo_to_id']


class RoomTypeDataset(Dataset):
    def __init__(self, jsonl_path, bert_name='models/bert-base-uncased',
                 combo_vocab_path=COMBO_VOCAB_PATH):
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        with open(jsonl_path, encoding='utf-8') as f:
            self.records = [json.loads(l) for l in f if l.strip()]

        self.num_types, _ = load_combo_vocab(combo_vocab_path)
        print(f"RoomTypeDataset: {len(self.records)} 条  num_types={self.num_types}  {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        n   = min(int(rec['n_nodes']), MAX_NODES)

        # 邻接矩阵
        adj_raw = np.array(rec['adj_matrix'], dtype=np.float32)[:n, :n]
        np.fill_diagonal(adj_raw, 0)
        adj_pad = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        adj_pad[:n, :n] = adj_raw

        # node_mask
        mask_np = np.zeros(MAX_NODES, dtype=np.float32)
        mask_np[:n] = 1.0

        # room_membership
        mb_np = np.zeros((MAX_NODES, MAX_ROOMS), dtype=np.float32)
        mb_np[:n] = _assign_room_membership_single(adj_raw.astype(bool), n)

        # 文本 token
        prompt = rec.get('prompt', '').replace('\n', ' ').strip()
        enc    = self.tokenizer(prompt, max_length=MAX_TEXT_LEN,
                                padding='max_length', truncation=True)
        ptok   = np.array(enc['input_ids'],      dtype=np.int64)
        pmsk   = np.array(enc['attention_mask'], dtype=np.float32)

        # 节点类型标签：直接用 node_combo_ids，padding 节点=-1
        combo_ids = rec.get('node_combo_ids', [])
        labels    = np.full(MAX_NODES, -1, dtype=np.int64)
        for i in range(n):
            labels[i] = int(combo_ids[i]) if i < len(combo_ids) else 0

        return {
            'node_mask':       torch.from_numpy(mask_np),
            'adj_matrix':      torch.from_numpy(adj_pad),
            'room_membership': torch.from_numpy(mb_np),
            'prompt_tokens':   torch.from_numpy(ptok),
            'prompt_mask':     torch.from_numpy(pmsk),
            'type_labels':     torch.from_numpy(labels),
        }


def load_data(jsonl_path, bert_name, batch_size, shuffle=True):
    dataset = RoomTypeDataset(jsonl_path, bert_name=bert_name)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=4, pin_memory=True, persistent_workers=True,
                         drop_last=True)
    return dataset, loader


# ── NPZ 版 Dataset（build_npz.py 预处理后使用，速度更快）────────────────────────

class RoomTypeNpzDataset(Dataset):
    """直接读取 build_npz.py 生成的 npz 文件，省去运行时 tokenize 和 room_membership 计算。"""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.node_mask       = data['node_mask'].astype(np.float32)     # [N, 40]
        self.adj_matrix      = data['adj_matrix'].astype(np.float32)    # [N, 40, 40]
        self.room_membership = data['room_membership']                  # [N, 40, MAX_ROOMS]
        self.prompt_tokens   = data['prompt_tokens'].astype(np.int64)   # [N, 192]
        self.prompt_mask     = data['prompt_mask'].astype(np.float32)   # [N, 192]
        self.type_labels     = data['type_labels'].astype(np.int64)     # [N, 40]
        print(f"RoomTypeNpzDataset: {len(self.node_mask)} 条  {npz_path}")

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


def load_npz_data(npz_path, batch_size, shuffle=True):
    dataset = RoomTypeNpzDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=4, pin_memory=True, persistent_workers=True,
                         drop_last=True)
    return dataset, loader
