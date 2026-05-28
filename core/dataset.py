"""
Dataset，支持两种数据源：
  local : 从本地 npz + jsonl 加载
  hf    : 从 HuggingFace Hub wzmmmm/plan-diffusion 加载

以及两种训练模式：
  pretrain : 只返回 token 序列
  sft      : 返回 token 序列 + BERT tokenize 后的 prompt
"""

import json
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

HF_REPO = 'wzmmmm/plan-diffusion'


def _filter_by_prompt_len(tokenizer, prompts, max_len):
    """返回 prompt tokenize 后长度 <= max_len 的索引列表。"""
    valid = []
    for i, p in enumerate(prompts):
        if len(tokenizer.encode(p, add_special_tokens=True)) <= max_len:
            valid.append(i)
    return valid


# ── 本地加载 ────────────────────────────────────────────────

class LocalGraphTokenDataset(Dataset):
    """预训练 · 本地：只返回 token 序列。"""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.tokens  = data['tokens']
        self.lengths = data['lengths']

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        return torch.tensor(self.tokens[idx, :length], dtype=torch.long)


class LocalGraphTextDataset(Dataset):
    """SFT · 本地：返回 token 序列 + prompt。超过 max_text_len 的样本丢弃。"""

    def __init__(self, npz_path, jsonl_path, bert_path,
                 max_text_len=256, augment=5):
        data = np.load(npz_path)
        tokens  = data['tokens']
        lengths = data['lengths']

        prompts = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for _ in range(augment):
                    prompts.append(rec['prompt'])
        assert len(prompts) == len(tokens), \
            f'prompt数({len(prompts)}) 与样本数({len(tokens)}) 不匹配'

        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len

        print('过滤超长 prompt...')
        valid = _filter_by_prompt_len(self.tokenizer, prompts, max_text_len)
        self.tokens  = tokens[valid]
        self.lengths = lengths[valid]
        self.prompts = [prompts[i] for i in valid]
        print(f'保留 {len(valid)} / {len(prompts)} 条 (丢弃 {len(prompts)-len(valid)} 条)')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        seq = torch.tensor(self.tokens[idx, :length], dtype=torch.long)
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=False,
            return_tensors='pt',
        )
        return seq, enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)


# ── HuggingFace Hub 加载 ────────────────────────────────────

class HFGraphTokenDataset(Dataset):
    """预训练 · HF Hub：只返回 token 序列。"""

    def __init__(self, repo_id=HF_REPO):
        from datasets import load_dataset
        print(f'从 HuggingFace Hub 加载数据集: {repo_id} ...')
        ds = load_dataset(repo_id, split='train')
        self.tokens  = ds['tokens']
        self.lengths = ds['lengths']

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        return torch.tensor(self.tokens[idx][:length], dtype=torch.long)


class HFGraphTextDataset(Dataset):
    """SFT · HF Hub：返回 token 序列 + prompt。超过 max_text_len 的样本丢弃。"""

    def __init__(self, bert_path, max_text_len=256, repo_id=HF_REPO):
        from datasets import load_dataset
        print(f'从 HuggingFace Hub 加载数据集: {repo_id} ...')
        ds = load_dataset(repo_id, split='train')

        self.tokenizer    = BertTokenizerFast.from_pretrained(bert_path)
        self.max_text_len = max_text_len

        print('过滤超长 prompt...')
        prompts = ds['prompt']
        valid   = _filter_by_prompt_len(self.tokenizer, prompts, max_text_len)
        self.tokens  = [ds['tokens'][i]  for i in valid]
        self.lengths = [ds['lengths'][i] for i in valid]
        self.prompts = [prompts[i]        for i in valid]
        print(f'保留 {len(valid)} / {len(prompts)} 条 (丢弃 {len(prompts)-len(valid)} 条)')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        seq = torch.tensor(self.tokens[idx][:length], dtype=torch.long)
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_text_len,
            padding='max_length',
            truncation=False,
            return_tensors='pt',
        )
        return seq, enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)


# ── 工厂函数 ────────────────────────────────────────────────

def build_dataset(mode, source, cfg):
    if source == 'local':
        if mode == 'pretrain':
            return LocalGraphTokenDataset(cfg['npz_path'])
        else:
            return LocalGraphTextDataset(
                cfg['npz_path'], cfg['jsonl_path'], cfg['bert_path'],
                max_text_len=cfg['max_text_len'],
            )
    else:
        if mode == 'pretrain':
            return HFGraphTokenDataset()
        else:
            return HFGraphTextDataset(
                bert_path=cfg['bert_path'],
                max_text_len=cfg['max_text_len'],
            )


# ── Collate ─────────────────────────────────────────────────

def pretrain_collate(batch, pad_id=0):
    max_len    = max(s.size(0) for s in batch)
    token_ids  = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        token_ids[i, :s.size(0)] = s
        token_mask[i, :s.size(0)] = 1
    return token_ids, token_mask


def sft_collate(batch, pad_id=0):
    seqs, text_ids, text_masks = zip(*batch)
    max_len    = max(s.size(0) for s in seqs)
    token_ids  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    token_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        token_ids[i, :s.size(0)] = s
        token_mask[i, :s.size(0)] = 1
    return token_ids, token_mask, torch.stack(text_ids), torch.stack(text_masks)
