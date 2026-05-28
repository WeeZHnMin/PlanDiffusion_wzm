"""
训练脚本，支持两种模式：
  MODE = 'pretrain'  纯 GPT-2 自回归预训练
  MODE = 'sft'       BERT + GPT-2 全量 SFT 微调
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

from model import GraphModel, TextConditionedGraphModel
from dataset import build_dataset, pretrain_collate, sft_collate

# ══════════════════════════════════════════════════════════════
#  切换这里
MODE        = 'pretrain'  # 'pretrain' | 'sft'
DATA_SOURCE = 'local'     # 'local'    | 'hf'
# ══════════════════════════════════════════════════════════════

_vocab      = json.load(open('../data/processed/type_combo_vocab.json', encoding='utf-8'))
PAD_ID      = 0
BOS_ID      = _vocab['BOS_ID']
EOS_ID      = _vocab['EOS_ID']
VOCAB_SIZE  = _vocab['VOCAB_SIZE']

GPT2_CFG = dict(
    vocab_size   = VOCAB_SIZE,
    n_embd       = 384,
    n_layer      = 8,
    n_head       = 6,
    n_positions  = 256,
    bos_token_id = BOS_ID,
    eos_token_id = EOS_ID,
    pad_token_id = PAD_ID,
    resid_pdrop  = 0.1,
    embd_pdrop   = 0.1,
    attn_pdrop   = 0.1,
)

PRETRAIN_CFG = dict(
    npz_path     = '../data/processed/graph_tokens_combo_5w.npz',
    save_dir     = '../checkpoints/pretrain',
    batch_size   = 32,
    max_epochs   = 100,
    lr           = 6e-4,
    weight_decay = 0.1,
    grad_clip    = 1.0,
    log_every    = 200,
    seed         = 42,
)

SFT_CFG = dict(
    npz_path             = '../data/processed/graph_tokens_combo_5w.npz',
    jsonl_path           = '../data/jsonl/mapped_type_data_zh.jsonl',
    bert_path            = '../models/bert-base-chinese',
    pretrained_gpt2_path = '../checkpoints/pretrain/best.pt',  # 预训练权重
    save_dir             = '../checkpoints/sft',
    batch_size           = 32,
    max_epochs           = 100,
    lr                   = 2e-4,   # SFT 用更小学习率
    weight_decay         = 0.1,
    grad_clip            = 1.0,
    log_every            = 200,
    max_text_len         = 64,
    seed                 = 42,
)


def freeze_bert_except_last_n(model, n=2):
    """冻结 BERT 除最后 n 个 transformer 层以外的所有参数。"""
    # 先全部冻结
    for param in model.bert.parameters():
        param.requires_grad = False

    # 解冻最后 n 层
    total_layers = len(model.bert.encoder.layer)
    for layer in model.bert.encoder.layer[total_layers - n:]:
        for param in layer.parameters():
            param.requires_grad = True

    # bert_proj 投影层始终可训练
    for param in model.bert_proj.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'BERT 冻结策略: 仅解冻最后 {n} 层')
    print(f'  可训练参数: {trainable/1e6:.1f}M  冻结参数: {frozen/1e6:.1f}M')


def run_epoch(model, loader, optimizer, scheduler, loss_fn,
              device, mode, epoch, log_every, global_step):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        if mode == 'pretrain':
            token_ids, token_mask = batch
            token_ids  = token_ids.to(device)
            token_mask = token_mask.to(device)
            x, y, m = token_ids[:, :-1], token_ids[:, 1:], token_mask[:, :-1]
            out  = model(x, m)
        else:
            token_ids, token_mask, text_ids, text_mask = batch
            token_ids  = token_ids.to(device)
            token_mask = token_mask.to(device)
            text_ids   = text_ids.to(device)
            text_mask  = text_mask.to(device)
            x, y, m = token_ids[:, :-1], token_ids[:, 1:], token_mask[:, :-1]
            out  = model(x, m, text_ids, text_mask)

        loss = loss_fn(out.logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss  += loss.item()
        global_step += 1

        if global_step % log_every == 0:
            print(f'epoch {epoch+1:3d}  step {global_step:5d}  '
                  f'loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}')

    return epoch_loss, global_step


def train():
    cfg  = PRETRAIN_CFG if MODE == 'pretrain' else SFT_CFG
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'mode: {MODE}  device: {device}')

    if MODE == 'pretrain':
        dataset  = build_dataset(MODE, DATA_SOURCE, cfg)
        collate  = partial(pretrain_collate, pad_id=PAD_ID)
        model    = GraphModel(GPT2_CFG).to(device)
    else:
        dataset  = build_dataset(MODE, DATA_SOURCE, cfg)
        collate  = partial(sft_collate, pad_id=PAD_ID)
        model    = TextConditionedGraphModel(
            bert_path=cfg['bert_path'],
            gpt2_cfg=GPT2_CFG,
            pretrained_gpt2_path=cfg.get('pretrained_gpt2_path'),
        ).to(device)
        freeze_bert_except_last_n(model, n=2)

    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        collate_fn=collate, drop_last=True, num_workers=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {n_params/1e6:.1f}M  样本数: {len(dataset)}  steps/epoch: {len(loader)}')

    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=cfg['lr'],
                        weight_decay=cfg['weight_decay'], betas=(0.9, 0.95))
    total_steps = cfg['max_epochs'] * len(loader)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg['lr'] * 0.1)
    loss_fn     = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best_loss   = float('inf')
    global_step = 0

    for epoch in range(cfg['max_epochs']):
        epoch_loss, global_step = run_epoch(
            model, loader, optimizer, scheduler, loss_fn,
            device, MODE, epoch, cfg['log_every'], global_step,
        )
        avg_loss = epoch_loss / len(loader)
        print(f'=== epoch {epoch+1} 结束  avg_loss={avg_loss:.4f} ===')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'best.pt'))
            print(f'  -> 保存最优模型  best_loss={best_loss:.4f}\n')
        else:
            print()

    print(f'训练完成（{MODE}）。最优 loss: {best_loss}')


if __name__ == '__main__':
    train()
