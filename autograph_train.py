"""
从零训练 GPT-2 做平面布局图结构生成（带节点组合类型）
数据：data/processed/graph_tokens_combo_5w.npz
常量从 data/processed/type_combo_vocab.json 加载
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

# ── 从词表文件加载常量 ────────────────────────────────────────
_vocab = json.load(open('data/processed/type_combo_vocab.json', encoding='utf-8'))
MAX_NODES   = _vocab['MAX_NODES']    # 40
PAD_ID      = 0
TOK_OPEN    = _vocab['TOK_OPEN']     # 33
TOK_CLOSE   = _vocab['TOK_CLOSE']    # 34
TOK_BREAK   = _vocab['TOK_BREAK']    # 35
BOS_ID      = _vocab['BOS_ID']       # 36
EOS_ID      = _vocab['EOS_ID']       # 37
NODE_OFFSET = _vocab['NODE_OFFSET']  # 38
VOCAB_SIZE  = _vocab['VOCAB_SIZE']   # 78

# ── 配置 ───────────────────────────────────────────────────
CFG = dict(
    data_path   = 'data/processed/graph_tokens_combo_5w.npz',
    save_dir    = 'checkpoints/autograph_combo',
    batch_size  = 32,
    max_epochs  = 20,
    lr          = 6e-4,
    weight_decay= 0.1,
    grad_clip   = 1.0,
    log_every   = 200,        # 每隔多少 step 打印一次
    save_every  = 2000,       # 每隔多少 step 保存一次
    seed        = 42,
)

# GPT-2 small 规模，对当前任务够用
MODEL_CFG = dict(
    vocab_size          = VOCAB_SIZE,
    n_embd              = 384,
    n_layer             = 8,
    n_head              = 6,
    n_positions         = 256,
    bos_token_id        = BOS_ID,
    eos_token_id        = EOS_ID,
    pad_token_id        = PAD_ID,
    resid_pdrop         = 0.1,
    embd_pdrop          = 0.1,
    attn_pdrop          = 0.1,
)


# ── Dataset ────────────────────────────────────────────────
class GraphTokenDataset(Dataset):
    def __init__(self, tokens, lengths):
        self.tokens  = tokens   # (N, max_len) int32
        self.lengths = lengths  # (N,) int32

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        seq = self.tokens[idx, :length]
        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch):
    """把不等长序列 pad 到本 batch 最长，同时生成 attention_mask。"""
    max_len = max(x.shape[0] for x in batch)
    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    mask   = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return padded, mask


# ── 辅助：token 序列解码成边列表 ───────────────────────────
def decode_tokens(token_list):
    """
    把 token 列表还原成边集合。
    返回 set of (i, j)，i < j，节点编号1-based（原始编号，去掉NODE_OFFSET偏移）。
    跳过类型token（1-7）和结构符号。
    """
    edges = set()
    i = 0
    toks = [t for t in token_list if t not in (BOS_ID, EOS_ID, PAD_ID)]
    prev_node = None
    in_bracket = False
    bracket_node = None

    while i < len(toks):
        t = toks[i]
        if t == TOK_BREAK:
            prev_node = None
            in_bracket = False
        elif t == TOK_OPEN:
            in_bracket = True
            bracket_node = prev_node
        elif t == TOK_CLOSE:
            in_bracket = False
            bracket_node = None
        elif NODE_OFFSET < t <= NODE_OFFSET + MAX_NODES:
            node_id = t - NODE_OFFSET  # 还原为1-based编号
            if in_bracket and bracket_node is not None:
                u, v = min(bracket_node, node_id), max(bracket_node, node_id)
                edges.add((u, v))
            else:
                if prev_node is not None:
                    u, v = min(prev_node, node_id), max(prev_node, node_id)
                    edges.add((u, v))
                prev_node = node_id
        # 1-7 是类型token，直接跳过
        i += 1
    return edges


# ── 验证时生成几条序列并打印 ───────────────────────────────
@torch.no_grad()
def generate_samples(model, device, n=4):
    model.eval()
    init = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
    out = model.generate(
        init,
        do_sample=True,
        top_k=CFG['top_k'],
        max_length=200,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
    )
    results = []
    for i in range(n):
        seq = out[i].tolist()
        edges = decode_tokens(seq)
        # 推断节点数
        nodes = set()
        for u, v in edges:
            nodes.add(u); nodes.add(v)
        results.append(dict(n_nodes=len(nodes), n_edges=len(edges), edges=edges))
    return results


# ── 训练 ───────────────────────────────────────────────────
def train():
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 加载数据
    raw = np.load(CFG['data_path'])
    tokens  = raw['tokens']    # (248295, 256), VOCAB_SIZE=78
    lengths = raw['lengths']   # (248295,)

    full_dataset = GraphTokenDataset(tokens, lengths)
    train_loader = DataLoader(full_dataset, batch_size=CFG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, drop_last=True,
                              num_workers=0)
    print(f'train: {len(full_dataset)}')

    # 建模型（从零初始化）
    config = GPT2Config(**MODEL_CFG)
    model  = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params/1e6:.1f}M')

    optimizer = AdamW(model.parameters(), lr=CFG['lr'],
                      weight_decay=CFG['weight_decay'], betas=(0.9, 0.95))
    total_steps = CFG['max_epochs'] * len(train_loader)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=CFG['lr'] * 0.1)
    loss_fn     = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    os.makedirs(CFG['save_dir'], exist_ok=True)

    global_step = 0

    for epoch in range(CFG['max_epochs']):
        model.train()
        epoch_loss = 0.0

        for batch, attn_mask in train_loader:
            batch     = batch.to(device)
            attn_mask = attn_mask.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            m = attn_mask[:, :-1]

            logits = model(input_ids=x, attention_mask=m).logits
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % CFG['log_every'] == 0:
                print(f'epoch {epoch+1:3d}  step {global_step:5d}  '
                      f'loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}')

            if global_step % CFG['save_every'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(CFG['save_dir'], f'step{global_step}.pt'))

        avg_loss = epoch_loss / len(train_loader)
        print(f'=== epoch {epoch+1} 结束  avg_loss={avg_loss:.4f} ===\n')

    print('训练完成。最优 val_loss:', best_val_loss)


if __name__ == '__main__':
    train()
