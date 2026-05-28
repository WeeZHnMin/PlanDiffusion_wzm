import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

# 坐标归一化常数：coords / COORD_SCALE → [-1, 1]
COORD_SCALE = 160.0


class NodeDataset(Dataset):
    """graph_only 模式：从旧格式 npz 加载（含 node_mask 字段）。"""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords     = d['coords'].astype(np.float32)
        self.adj_matrix = d['adj_matrix'].astype(np.float32)
        self.node_mask  = d['node_mask'].astype(np.float32)
        print(f"NodeDataset: {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy() / COORD_SCALE
        cond = {
            'adj_matrix': self.adj_matrix[idx],
            'node_mask':  self.node_mask[idx],
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


class NodeTextDataset(Dataset):
    """text+graph 模式：从主 npz + jsonl 加载。prompt 超长截断。"""

    def __init__(self, npz_path, jsonl_path, bert_path,
                 max_text_len=256, augment=5):
        d       = np.load(npz_path)
        self.coords     = d['coords'].astype(np.float32)
        self.adj_matrix = d['adj_matrix'].astype(np.float32)
        n_nodes         = d['n_nodes'].astype(np.int32)

        N, MAX_N = len(n_nodes), self.coords.shape[1]
        self.node_mask = np.zeros((N, MAX_N), dtype=np.float32)
        for i, n in enumerate(n_nodes):
            self.node_mask[i, :n] = 1.0

        prompts = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for _ in range(augment):
                    prompts.append(rec['prompt'])
        assert len(prompts) == N, f'prompt数({len(prompts)}) 与样本数({N}) 不匹配'
        self.prompts      = prompts
        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len
        print(f"NodeTextDataset: {N} samples")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x   = torch.from_numpy(self.coords[idx].T.copy() / COORD_SCALE)
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        cond = {
            'adj_matrix': torch.from_numpy(self.adj_matrix[idx]),
            'node_mask':  torch.from_numpy(self.node_mask[idx]),
            'text_ids':   enc['input_ids'].squeeze(0),
            'text_mask':  enc['attention_mask'].squeeze(0),
        }
        return x, cond


class HFNodeTextDataset(Dataset):
    """text+graph 模式 · HF Hub。prompt 超长截断。"""

    def __init__(self, bert_path, max_text_len=256, repo_id='wzmmmm/plan-diffusion'):
        from datasets import load_dataset
        print(f'从 HuggingFace Hub 加载: {repo_id} ...')
        ds = load_dataset(repo_id, split='train')
        self.coords     = ds['coords']
        self.adj_matrix = ds['adj_matrix']
        self.n_nodes    = ds['n_nodes']
        self.prompts    = ds['prompt']
        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len
        print(f'HFNodeTextDataset: {len(self.prompts)} samples')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        n      = self.n_nodes[idx]
        coords = np.array(self.coords[idx], dtype=np.float32)
        x      = torch.from_numpy(coords.T / COORD_SCALE)

        adj       = torch.tensor(self.adj_matrix[idx], dtype=torch.float32)
        node_mask = torch.zeros(coords.shape[0], dtype=torch.float32)
        node_mask[:n] = 1.0

        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        cond = {
            'adj_matrix': adj,
            'node_mask':  node_mask,
            'text_ids':   enc['input_ids'].squeeze(0),
            'text_mask':  enc['attention_mask'].squeeze(0),
        }
        return x, cond


def load_node_data(npz_path, batch_size, shuffle=True):
    dataset = NodeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=2, drop_last=True)
    while True:
        yield from loader


def load_node_text_data(npz_path, jsonl_path, bert_path,
                        batch_size, max_text_len=256, shuffle=True):
    dataset = NodeTextDataset(npz_path, jsonl_path, bert_path, max_text_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, drop_last=True)
    while True:
        yield from loader


def load_hf_node_text_data(bert_path, batch_size, max_text_len=256, shuffle=True):
    dataset = HFNodeTextDataset(bert_path, max_text_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, drop_last=True)
    while True:
        yield from loader
