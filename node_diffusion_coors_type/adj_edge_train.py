"""
Train AdjEdgeModel: prompt -> edge list sequence.

Usage:
    python -m node_diffusion.adj_edge_train \
        --npz data/processed/nodes_train_6k_norm.npz \
        --bert_path models/bert-base-chinese
"""

import os
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .adj_edge_model import (AdjEdgeModel, adj_to_edge_seq,
                              edge_seq_to_adj, BOS, EOS, VOCAB_SIZE)


MAX_SEQ = 120   # BOS + max edges + EOS, padded to this length


class EdgeDataset(Dataset):
    def __init__(self, npz_path, bert_path, max_len=128):
        d = np.load(npz_path, allow_pickle=True)
        self.adj_matrix = d['adj_matrix']
        self.node_mask  = d['node_mask']
        self.prompts    = d['prompts']
        self.tokenizer  = BertTokenizer.from_pretrained(bert_path)
        self.max_len    = max_len

        # precompute edge sequences
        self.edge_seqs = []
        max_edges = 0
        for i in range(len(self.adj_matrix)):
            n = int(self.node_mask[i].sum())
            seq = adj_to_edge_seq(self.adj_matrix[i], n)   # edges + EOS
            self.edge_seqs.append(seq)
            max_edges = max(max_edges, len(seq))

        print(f"EdgeDataset: {len(self.prompts)} samples, "
              f"max edges+EOS={max_edges}, using MAX_SEQ={MAX_SEQ}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.prompts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # build tgt: [BOS, e0, e1, ..., EOS, PAD, PAD, ...]
        seq = self.edge_seqs[idx]                          # edges + EOS
        tgt = np.full(MAX_SEQ, -1, dtype=np.int64)
        tgt[0] = BOS
        length = min(len(seq), MAX_SEQ - 1)
        tgt[1:1 + length] = seq[:length]

        tgt_pad_mask = (tgt == -1).astype(np.float32)
        tgt[tgt == -1] = EOS                               # fill pad with EOS (ignored in loss)

        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'tgt':            torch.from_numpy(tgt).long(),
            'tgt_pad_mask':   torch.from_numpy(tgt_pad_mask),
        }


def edge_accuracy(pred_seq, tgt_adj, n_nodes):
    """
    pred_seq: list of edge indices (before EOS)
    tgt_adj : [40,40] ground truth adj
    Returns precision, recall
    """
    pred_set = set(int(x) for x in pred_seq if int(x) != EOS)
    true_set = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if tgt_adj[i, j] > 0.5:
                from .adj_edge_model import _EDGE2IDX
                true_set.add(_EDGE2IDX[(i, j)])

    if not pred_set:
        return 0.0, 0.0
    precision = len(pred_set & true_set) / len(pred_set)
    recall    = len(pred_set & true_set) / len(true_set) if true_set else 1.0
    return precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',           default='data/processed/nodes_train_6k_norm.npz')
    parser.add_argument('--bert_path',     default='models/bert-base-chinese')
    parser.add_argument('--save_dir',      default='checkpoints/adj_edge_model')
    parser.add_argument('--resume',        default='')
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--total_steps',   type=int,   default=50000)
    parser.add_argument('--log_interval',  type=int,   default=100)
    parser.add_argument('--save_interval', type=int,   default=5000)
    parser.add_argument('--d_model',       type=int,   default=256)
    parser.add_argument('--num_layers',    type=int,   default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = EdgeDataset(args.npz, args.bert_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2, drop_last=True)

    model = AdjEdgeModel(
        bert_path=args.bert_path,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    opt = AdamW(model.decoder.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.decoder.load_state_dict(ckpt['decoder'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['step'] + 1
        print(f"resumed from step {start_step}")

    def inf_loader():
        while True:
            yield from loader

    data = inf_loader()
    model.train()
    running_loss = 0.0

    for step in range(start_step, args.total_steps):
        batch = next(data)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tgt            = batch['tgt'].to(device)
        tgt_pad_mask   = batch['tgt_pad_mask'].to(device)

        loss = model(input_ids, attention_mask, tgt, tgt_pad_mask)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval if step > 0 else running_loss
            running_loss = 0.0

            # evaluate precision/recall on 4 samples
            with torch.no_grad():
                model.eval()
                pred = model.generate(input_ids[:4], attention_mask[:4])  # [4, seq]
                prec_list, rec_list = [], []
                for b in range(4):
                    seq  = pred[b].cpu().numpy().tolist()
                    n    = int(batch['tgt_pad_mask'][b].sum() == 0 or True)
                    # get n_nodes from dataset
                    p, r = edge_accuracy(seq, dataset.adj_matrix[b],
                                         int(dataset.node_mask[b].sum()))
                    prec_list.append(p)
                    rec_list.append(r)
                model.train()

            prec = sum(prec_list) / len(prec_list)
            rec  = sum(rec_list)  / len(rec_list)
            print(f"step {step:6d} | loss {avg:.4f} | prec {prec:.3f} | recall {rec:.3f}")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f'model_{step:06d}.pt')
            torch.save({'decoder': model.decoder.state_dict(),
                        'opt':     opt.state_dict(),
                        'step':    step}, path)
            print(f"  saved -> {path}")

    path = os.path.join(args.save_dir, f'model_{args.total_steps:06d}.pt')
    torch.save({'decoder': model.decoder.state_dict(),
                'opt':     opt.state_dict(),
                'step':    args.total_steps}, path)
    print(f"done. saved -> {path}")


if __name__ == '__main__':
    main()
