"""
训练 RoomTypeClassifier。

先用 build_npz.py 预处理数据，再训练：
  python -m room_type_clf.build_npz \\
      --jsonl     data/jsonl/final_graph_dataset_v3.jsonl \\
      --val_jsonl data/jsonl/val_graph_dataset_18k5.jsonl \\
      --augment 4 --output data/processed/room_type_clf/train.npz

  python -m room_type_clf.train \\
      --train data/processed/room_type_clf/train.npz \\
      --val   data/processed/room_type_clf/val.npz \\
      --no_text --gpu 3
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from .dataset import load_data, load_combo_vocab
from .model import RoomTypeClassifier

VAL_SAMPLES = 4096


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train',         required=True, help='训练集 npz')
    p.add_argument('--val',           required=True, help='验证集 npz')
    p.add_argument('--bert',          default='models/bert-base-uncased')
    p.add_argument('--save_dir',      default='checkpoints/room_type_clf')
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--total_steps',   type=int,   default=50000)
    p.add_argument('--log_interval',  type=int,   default=100)
    p.add_argument('--save_interval', type=int,   default=1000)
    p.add_argument('--val_interval',  type=int,   default=1000)
    p.add_argument('--model_channels',type=int,   default=256)
    p.add_argument('--num_layers',    type=int,   default=4)
    p.add_argument('--num_heads',     type=int,   default=4)
    p.add_argument('--gpu',           type=int,   default=None)
    p.add_argument('--resume',        default='')
    p.add_argument('--no_text',       action='store_true',
                   help='禁用文本编码器（baseline 实验）')
    return p


def run_val(model, val_loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_nodes = n_samples = n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            node_mask  = batch['node_mask'].to(device)
            adj        = batch['adj_matrix'].to(device)
            membership = batch['room_membership'].to(device)
            ptok       = batch['prompt_tokens'].to(device)
            pmsk       = batch['prompt_mask'].to(device)
            labels     = batch['type_labels'].to(device)

            logits = model(node_mask, adj, membership, ptok, pmsk)  # [B, N, C]
            B, N, C = logits.shape
            loss = criterion(logits.view(B * N, C), labels.view(B * N))

            valid         = (labels >= 0)
            preds         = logits.argmax(dim=-1)
            total_correct += (preds[valid] == labels[valid]).sum().item()
            total_nodes   += valid.sum().item()
            total_loss    += loss.item()
            n_samples     += B
            n_batches     += 1
            if n_samples >= VAL_SAMPLES:
                break

    model.train()
    return total_loss / max(n_batches, 1), total_correct / max(total_nodes, 1)


def main():
    args   = build_parser().parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 词表
    num_types, _ = load_combo_vocab()
    print(f'组合类型数: {num_types}')

    # 数据
    train_ds, train_loader = load_data(args.train, args.batch_size, shuffle=True)
    _,        val_loader   = load_data(args.val,   batch_size=64,   shuffle=False)

    # 模型
    model = RoomTypeClassifier(
        num_types      = num_types,
        model_channels = args.model_channels,
        num_layers     = args.num_layers,
        num_heads      = args.num_heads,
        bert_name      = args.bert,
        use_text       = not args.no_text,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    no_decay = {'bias', 'norm', 'LayerNorm'}
    opt = AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ], lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f'resumed from step {start_step}')

    log_path = save_dir / 'log.jsonl'
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    print(f'日志: {log_path}')

    def inf_loader():
        while True:
            yield from train_loader

    model.train()
    data_iter    = inf_loader()
    running_loss = running_acc = 0.0
    best_val_acc = 0.0
    t0           = time.perf_counter()

    for step in range(start_step, args.total_steps):
        batch = next(data_iter)

        node_mask  = batch['node_mask'].to(device)
        adj        = batch['adj_matrix'].to(device)
        membership = batch['room_membership'].to(device)
        ptok       = batch['prompt_tokens'].to(device)
        pmsk       = batch['prompt_mask'].to(device)
        labels     = batch['type_labels'].to(device)

        logits = model(node_mask, adj, membership, ptok, pmsk)  # [B, N, C]
        B, N, C = logits.shape
        loss = criterion(logits.view(B * N, C), labels.view(B * N))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            valid = (labels >= 0)
            preds = logits.argmax(dim=-1)
            acc   = (preds[valid] == labels[valid]).float().mean().item()

        running_loss += loss.item()
        running_acc  += acc

        if step % args.log_interval == 0 and step > 0:
            avg_loss = running_loss / args.log_interval
            avg_acc  = running_acc  / args.log_interval
            running_loss = running_acc = 0.0
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(f'step {step:6d} | loss {avg_loss:.4f} | acc {avg_acc:.4f} | {elapsed:.1f}s')
            log_file.write(json.dumps({
                'step': step, 'loss': round(avg_loss, 4),
                'acc': round(avg_acc, 4), 'elapsed': round(elapsed, 1),
            }) + '\n')

        if step % args.save_interval == 0 and step > 0:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'step': step}, save_dir / 'latest.pt')
            print(f'  saved -> {save_dir}/latest.pt')

        if step % args.val_interval == 0 and step > 0:
            val_loss, val_acc = run_val(model, val_loader, criterion, device)
            print(f'  [val] loss {val_loss:.4f} | acc {val_acc:.4f}')
            log_file.write(json.dumps({
                'step': step, 'val_loss': round(val_loss, 4),
                'val_acc': round(val_acc, 4),
            }) + '\n')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'step': step, 'val_acc': val_acc}, save_dir / 'best.pt')
                print(f'  best saved (val_acc={val_acc:.4f})')

    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'step': args.total_steps}, save_dir / 'latest.pt')
    log_file.close()
    print('done.')


if __name__ == '__main__':
    main()
