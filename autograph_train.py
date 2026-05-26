"""
从零训练 GPT-2 做平面布局图结构生成
数据：data/processed/graph_tokens_train.npz
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

# ── 常量（和 autograph_preprocess.py 保持一致）─────────────
MAX_NODES  = 40
PAD_ID     = 0
TOK_OPEN   = 41   # <
TOK_CLOSE  = 42   # >
TOK_BREAK  = 43   # /
BOS_ID     = 44
EOS_ID     = 45
VOCAB_SIZE = 46

# ── 配置 ───────────────────────────────────────────────────
CFG = dict(
    data_path   = 'data/processed/graph_tokens_train.npz',
    save_dir    = 'checkpoints/autograph',
    val_ratio   = 0.1,        # 10% 做验证集
    batch_size  = 64,
    max_epochs  = 200,
    lr          = 6e-4,
    weight_decay= 0.1,
    grad_clip   = 1.0,
    log_every   = 50,         # 每隔多少 step 打印一次
    val_every   = 500,        # 每隔多少 step 验证一次
    save_every  = 1000,       # 每隔多少 step 保存一次
    gen_samples = 4,          # 验证时生成几张图看看
    top_k       = 10,
    seed        = 42,
)

# GPT-2 small 规模，对当前任务够用
MODEL_CFG = dict(
    vocab_size          = VOCAB_SIZE,
    n_embd              = 384,
    n_layer             = 8,
    n_head              = 6,
    n_positions         = 256,    # 比最长序列 143 留有余量
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
    返回 set of (i, j)，i < j，节点编号从 1 开始。
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
        elif 1 <= t <= MAX_NODES:
            if in_bracket and bracket_node is not None:
                u, v = min(bracket_node, t), max(bracket_node, t)
                edges.add((u, v))
            else:
                if prev_node is not None:
                    u, v = min(prev_node, t), max(prev_node, t)
                    edges.add((u, v))
                prev_node = t
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
    tokens  = raw['tokens']    # (6000, 143)
    lengths = raw['lengths']   # (6000,)

    full_dataset = GraphTokenDataset(tokens, lengths)
    n_val   = int(len(full_dataset) * CFG['val_ratio'])
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG['seed'])
    )
    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'],
                              shuffle=True,  collate_fn=collate_fn, drop_last=True,
                              num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=CFG['batch_size'],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)
    print(f'train: {n_train}  val: {n_val}')

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
    best_val_loss = float('inf')

    for epoch in range(CFG['max_epochs']):
        model.train()
        epoch_loss = 0.0

        for batch, attn_mask in train_loader:
            batch     = batch.to(device)          # (B, L)
            attn_mask = attn_mask.to(device)      # (B, L)
            x = batch[:, :-1]                     # 输入：去掉最后一个
            y = batch[:, 1:]                      # 目标：右移一位
            m = attn_mask[:, :-1]                 # mask 对应输入

            logits = model(input_ids=x, attention_mask=m).logits  # (B, L-1, V)
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

            # 验证
            if global_step % CFG['val_every'] == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vbatch, vmask in val_loader:
                        vbatch = vbatch.to(device)
                        vmask  = vmask.to(device)
                        vx, vy = vbatch[:, :-1], vbatch[:, 1:]
                        vm = vmask[:, :-1]
                        vlogits = model(input_ids=vx, attention_mask=vm).logits
                        val_loss += loss_fn(vlogits.reshape(-1, VOCAB_SIZE), vy.reshape(-1)).item()
                val_loss /= len(val_loader)
                print(f'\n  ── val loss: {val_loss:.4f} ──')

                # 生成几张图看看
                samples = generate_samples(model, device, n=CFG['gen_samples'])
                for s in samples:
                    print(f'     节点={s["n_nodes"]}  边={s["n_edges"]}')
                print()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(CFG['save_dir'], 'best.pt'))
                    print(f'  ✓ 保存最优模型 val_loss={best_val_loss:.4f}\n')

                model.train()

            # 定期保存
            if global_step % CFG['save_every'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(CFG['save_dir'], f'step{global_step}.pt'))

        avg_loss = epoch_loss / len(train_loader)
        print(f'=== epoch {epoch+1} 结束  avg_loss={avg_loss:.4f} ===\n')

    print('训练完成。最优 val_loss:', best_val_loss)


if __name__ == '__main__':
    train()
