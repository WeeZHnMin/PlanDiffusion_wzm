import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

MAX_NODES    = 40
MAX_TEXT_LEN = 192


class AlignDataset(Dataset):
    def __init__(self, jsonl_path, bert_name='models/bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.records = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        print(f"AlignDataset: {len(self.records)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        n   = int(rec['n_nodes'])

        # 坐标质心归零
        raw = np.array(rec['node_coords'][:n], dtype=np.float32)
        raw = raw - raw.mean(axis=0)
        coords = np.zeros((MAX_NODES, 2), dtype=np.float32)
        coords[:n] = raw

        # 邻接矩阵
        adj_raw = np.array(rec['adj_matrix'], dtype=np.float32)[:n, :n]
        np.fill_diagonal(adj_raw, 0)
        adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        adj[:n, :n] = adj_raw

        # 节点 mask
        mask = np.zeros(MAX_NODES, dtype=np.float32)
        mask[:n] = 1.0

        # 文本 token
        prompt = rec.get('prompt', '').replace('\n', ' ').strip()
        enc = self.tokenizer(
            prompt, add_special_tokens=True,
            max_length=MAX_TEXT_LEN, padding='max_length', truncation=True,
        )
        input_ids      = np.array(enc['input_ids'],      dtype=np.int64)
        attention_mask = np.array(enc['attention_mask'], dtype=np.float32)

        return {
            'input_ids':      torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'coords':         torch.from_numpy(coords),
            'adj':            torch.from_numpy(adj),
            'mask':           torch.from_numpy(mask),
        }


def load_align_data(jsonl_path, batch_size, bert_name='models/bert-base-uncased',
                    shuffle=True, num_workers=4):
    ds = AlignDataset(jsonl_path, bert_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True)
    while True:
        yield from loader
