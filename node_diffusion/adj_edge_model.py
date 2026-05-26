"""
Autoregressive edge-list generation model.

Vocabulary:
  0~779  : edge index in upper triangle of 40x40 adj matrix
  780    : EOS
  781    : BOS (input only)

Each sample is a variable-length sequence of edge indices ending with EOS.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel

MAX_NODES  = 40
UPPER_I, UPPER_J = np.triu_indices(MAX_NODES, k=1)   # 780 edges
N_EDGES    = len(UPPER_I)                             # 780
EOS        = N_EDGES                                  # 780
BOS        = N_EDGES + 1                              # 781
VOCAB_SIZE = N_EDGES + 2                              # 782

# lookup: (i,j) -> edge index
_EDGE2IDX = {}
for idx, (i, j) in enumerate(zip(UPPER_I.tolist(), UPPER_J.tolist())):
    _EDGE2IDX[(i, j)] = idx


def adj_to_edge_seq(adj, n_nodes):
    """adj [40,40] -> edge index list (sorted) + EOS, as np.int64 array"""
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j] > 0.5:
                edges.append(_EDGE2IDX[(i, j)])
    edges.append(EOS)
    return np.array(edges, dtype=np.int64)


def edge_seq_to_adj(edge_seq, n_nodes=MAX_NODES):
    """edge index list (without EOS) -> adj [40,40]"""
    adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    for idx in edge_seq:
        if idx == EOS:
            break
        if 0 <= idx < N_EDGES:
            i, j = int(UPPER_I[idx]), int(UPPER_J[idx])
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    # zero out padding nodes
    adj[n_nodes:, :] = 0
    adj[:, n_nodes:] = 0
    np.fill_diagonal(adj, 0)
    adj[:n_nodes, :n_nodes] += np.eye(n_nodes)
    return adj


class EdgeDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4,
                 dropout=0.1, max_seq=120):
        super().__init__()
        self.max_seq = max_seq
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb   = nn.Embedding(max_seq + 1, d_model)
        self.enc_proj  = nn.Linear(768, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, enc_out, enc_pad_mask, tgt, tgt_pad_mask):
        """
        enc_out      [B, src, 768]
        enc_pad_mask [B, src]   1=padding
        tgt          [B, seq]   token ids (BOS + edge indices + EOS, right-padded)
        tgt_pad_mask [B, seq]   1=padding position

        Returns logits [B, seq, VOCAB_SIZE]
        """
        B, seq = tgt.shape
        pos  = torch.arange(seq, device=tgt.device).unsqueeze(0)
        emb  = self.token_emb(tgt) + self.pos_emb(pos)

        causal = nn.Transformer.generate_square_subsequent_mask(seq, device=tgt.device)
        mem    = self.enc_proj(enc_out)

        out = self.decoder(
            tgt=emb,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask.bool(),
            memory_key_padding_mask=enc_pad_mask.bool(),
        )
        return self.out_proj(out)

    @torch.no_grad()
    def generate(self, enc_out, enc_pad_mask):
        B   = enc_out.size(0)
        mem = self.enc_proj(enc_out)
        gen = torch.full((B, 1), BOS, dtype=torch.long, device=enc_out.device)
        finished = torch.zeros(B, dtype=torch.bool, device=enc_out.device)

        for _ in range(self.max_seq):
            pos  = torch.arange(gen.size(1), device=enc_out.device).unsqueeze(0)
            emb  = self.token_emb(gen) + self.pos_emb(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(
                gen.size(1), device=enc_out.device)
            out  = self.decoder(emb, mem, tgt_mask=mask,
                                memory_key_padding_mask=enc_pad_mask.bool())
            next_tok = self.out_proj(out[:, -1]).argmax(-1)    # [B]
            next_tok[finished] = EOS
            gen = torch.cat([gen, next_tok.unsqueeze(1)], dim=1)
            finished |= (next_tok == EOS)
            if finished.all():
                break

        return gen[:, 1:]   # remove BOS


class AdjEdgeModel(nn.Module):
    def __init__(self, bert_path='models/bert-base-chinese',
                 d_model=256, nhead=4, num_layers=4, dropout=0.1, max_seq=120):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_path)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.decoder = EdgeDecoder(d_model, nhead, num_layers, dropout, max_seq)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        n = sum(p.numel() for p in self.decoder.parameters())
        print(f"AdjEdgeModel decoder: {n:,} parameters (BERT frozen)")

    def forward(self, input_ids, attention_mask, tgt, tgt_pad_mask):
        """
        tgt          [B, seq]  BOS + edges + EOS + padding(-1 replaced with PAD)
        tgt_pad_mask [B, seq]  1=padding
        loss ignores padding positions via ignore_index=-1
        """
        with torch.no_grad():
            enc = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state

        enc_pad = 1 - attention_mask
        # input  = tgt[:, :-1]  (BOS ... last_edge)
        # target = tgt[:, 1:]   (first_edge ... EOS)
        inp     = tgt[:, :-1].clone()
        target  = tgt[:, 1:].clone()
        inp_pad = tgt_pad_mask[:, :-1]

        # padding positions in target -> -1 (ignored by loss)
        target[tgt_pad_mask[:, 1:].bool()] = -1

        logits = self.decoder(enc, enc_pad, inp, inp_pad)      # [B, seq-1, V]
        loss   = self.loss_fn(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
        return loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        enc     = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        enc_pad = 1 - attention_mask
        return self.decoder.generate(enc, enc_pad)
