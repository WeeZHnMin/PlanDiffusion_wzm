import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

# 坐标归一化常数：coords / COORD_SCALE → [-1, 1]
# 坐标中心化后实测范围约 [-153, 153]，取 160 留余量
COORD_SCALE = 160.0


class NodeDataset(Dataset):
    """
    graph_only 模式：从旧格式 npz 加载（含 node_mask 字段）。
    返回 x [2,40]，cond {adj_matrix, node_mask}
    """

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.coords     = d['coords'].astype(np.float32)
        self.adj_matrix = d['adj_matrix'].astype(np.float32)
        self.node_mask  = d['node_mask'].astype(np.float32)
        print(f"NodeDataset: {len(self.coords)} samples from {npz_path}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].T.copy() / COORD_SCALE   # [2, 40]，归一化到 [-1, 1]
        cond = {
            'adj_matrix': self.adj_matrix[idx],
            'node_mask':  self.node_mask[idx],
        }
        return torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in cond.items()}


class NodeTextDataset(Dataset):
    """
    text+graph 模式：从主 npz（graph_tokens_combo_5w.npz）+ jsonl 加载。
    返回 x [2,40]，cond {adj_matrix, node_mask, text_ids, text_mask}
    """

    def __init__(self, npz_path, jsonl_path, bert_path,
                 max_text_len=64, augment=5):
        d = np.load(npz_path)
        self.coords     = d['coords'].astype(np.float32)      # [N, 40, 2]
        self.adj_matrix = d['adj_matrix'].astype(np.float32)  # [N, 40, 40]
        n_nodes         = d['n_nodes'].astype(np.int32)        # [N,]

        # 从 n_nodes 生成 node_mask
        N, MAX_N = len(n_nodes), self.coords.shape[1]
        self.node_mask = np.zeros((N, MAX_N), dtype=np.float32)
        for i, n in enumerate(n_nodes):
            self.node_mask[i, :n] = 1.0

        # 从 jsonl 重建 prompt（每条重复 augment 次）
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
        self.prompts = prompts

        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len
        print(f"NodeTextDataset: {N} samples")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.coords[idx].T.copy() / COORD_SCALE)  # [2, 40]，归一化到 [-1, 1]

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
    """
    text+graph 模式 · HF Hub：
    从 wzmmmm/plan-diffusion 加载 coords / adj_matrix / n_nodes / prompt。
    """

    def __init__(self, bert_path, max_text_len=64, repo_id='wzmmmm/plan-diffusion'):
        from datasets import load_dataset
        print(f'从 HuggingFace Hub 加载: {repo_id} ...')
        ds = load_dataset(repo_id, split='train')
        self.coords     = ds['coords']      # list of [40, 2] list
        self.adj_matrix = ds['adj_matrix']  # list of [40, 40] list
        self.n_nodes    = ds['n_nodes']     # list of int
        self.prompts    = ds['prompt']      # list of str

        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len
        print(f'HFNodeTextDataset: {len(self.prompts)} samples')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        n = self.n_nodes[idx]

        # coords: [40, 2] list → [2, 40] tensor，归一化
        coords = np.array(self.coords[idx], dtype=np.float32)   # [40, 2]
        x = torch.from_numpy(coords.T / COORD_SCALE)            # [2, 40]

        # adj_matrix: [40, 40] list → tensor
        adj = torch.tensor(self.adj_matrix[idx], dtype=torch.float32)  # [40, 40]

        # node_mask: 从 n_nodes 推导
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
    """graph_only · 本地。"""
    dataset = NodeDataset(npz_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=2, drop_last=True)
    while True:
        yield from loader


def load_node_text_data(npz_path, jsonl_path, bert_path,
                        batch_size, max_text_len=64, shuffle=True):
    """text+graph · 本地。"""
    dataset = NodeTextDataset(npz_path, jsonl_path, bert_path, max_text_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, drop_last=True)
    while True:
        yield from loader


def load_hf_node_text_data(bert_path, batch_size, max_text_len=64, shuffle=True):
    """text+graph · HF Hub。"""
    dataset = HFNodeTextDataset(bert_path, max_text_len)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, drop_last=True)
    while True:
        yield from loader
