"""
TextGraphAlign: CLIP 风格对比预训练

GraphEncoder : 四流注意力 → mean pool → MLP投影头 → L2归一化
TextEncoder  : 从零训练的 Transformer，输入 token IDs → mean pool → MLP投影头 → L2归一化
TextGraphAlign: 两编码器 + 可学习温度 logit_scale，对称 InfoNCE
"""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_ROOMS = 20


# ── Room Membership 分配 ──────────────────────────────────────────────────────

def _assign_room_membership_single(adj, n):
    seen       = set()
    next_rid   = 0
    membership = np.zeros((n, MAX_ROOMS), dtype=np.int32)
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            prev  = {u: -1}
            q     = deque([u])
            found = False
            while q and not found:
                cur = q.popleft()
                for w in range(n):
                    if not adj[cur, w] or w in prev:
                        continue
                    if cur == u and w == v:
                        continue
                    prev[w] = cur
                    if w == v:
                        found = True
                        break
                    q.append(w)
            if not found:
                continue
            cycle = []
            cur   = v
            while cur != -1:
                cycle.append(cur)
                cur = prev[cur]
            key = frozenset(cycle)
            if key in seen:
                continue
            seen.add(key)
            if next_rid >= MAX_ROOMS:
                continue
            for node in cycle:
                membership[node, next_rid] = 1
            next_rid += 1
    return membership


# ── 图编码器基础模块 ──────────────────────────────────────────────────────────

def _attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, -1e4)
    scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout is not None:
        scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k      = d_model // heads
        self.h        = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out      = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q  = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k  = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v  = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        out = _attention(q, k, v, self.d_k, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(out)


class GlobalRoomAttnStream(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k      = d_model // heads
        self.h        = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out      = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, room_membership, pad_mask=None):
        B, N, d = x.shape
        H, d_k  = self.h, self.d_k
        Q = self.q_linear(x).view(B, N, H, d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, N, H, d_k).transpose(1, 2)
        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if pad_mask is not None:
            base_scores = base_scores - pad_mask.unsqueeze(2) * 1e4
        R           = room_membership.shape[2]
        out_accum   = torch.zeros(B, H, N, d_k, device=x.device, dtype=Q.dtype)
        room_counts = torch.zeros(B, 1, N, 1,   device=x.device, dtype=Q.dtype)
        for k in range(R):
            mem_k = room_membership[:, :, k]
            if mem_k.sum() == 0:
                continue
            col_mask    = (1.0 - mem_k).unsqueeze(1).unsqueeze(2) * 1e4
            scores_k    = base_scores - col_mask
            attn_k      = F.softmax(scores_k.float(), dim=-1).to(Q.dtype)
            attn_k      = self.dropout(attn_k)
            out_k       = torch.matmul(attn_k, V)
            node_mask_k = mem_k.unsqueeze(1).unsqueeze(3)
            out_accum   = out_accum   + out_k * node_mask_k
            room_counts = room_counts + node_mask_k
        out = out_accum / room_counts.clamp(min=1.0)
        out = out.transpose(1, 2).contiguous().view(B, N, H * d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1            = nn.LayerNorm(d_model)
        self.norm2            = nn.LayerNorm(d_model)
        self.adj_attn         = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn        = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn      = MultiHeadAttention(heads, d_model, dropout)
        self.global_room_attn = GlobalRoomAttnStream(heads, d_model, dropout)
        self.ff               = FeedForward(d_model, dropout)
        self.dropout          = nn.Dropout(dropout)

    def forward(self, x, room_mask, room_membership, pad_mask, adj_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn        (x2, x2, x2, adj_mask)          +
            self.room_attn       (x2, x2, x2, room_mask)         +
            self.global_attn     (x2, x2, x2, pad_mask)          +
            self.global_room_attn(x2, room_membership, pad_mask)
        )
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 图编码器 ──────────────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    def __init__(self, d_model=256, num_layers=4, num_heads=4,
                 d_embed=256, dropout=0.1):
        super().__init__()
        self.node_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers     = nn.ModuleList(
            [GraphEncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _build_masks(self, adj_matrix, room_membership, node_mask):
        dt = torch.float32
        nm = node_mask.to(dt)
        rm = room_membership.to(dt)
        room_nn  = torch.bmm(rm, rm.transpose(1, 2))
        room_msk = (room_nn == 0).to(dt)
        pad_keys = (1 - nm).unsqueeze(1)
        room_msk = torch.clamp(room_msk + pad_keys, 0, 1)
        am  = adj_matrix.to(dt)
        eye = torch.eye(am.shape[1], device=am.device, dtype=dt).unsqueeze(0)
        adj_msk = torch.clamp(1 - (am + eye).clamp(0, 1) + pad_keys, 0, 1)
        return room_msk, pad_keys, adj_msk

    def forward(self, node_mask, adj_matrix, room_membership):
        B, N = node_mask.shape
        seq  = self.node_token.expand(B, N, -1).clone()
        room_msk, pad_mask, adj_msk = self._build_masks(
            adj_matrix, room_membership, node_mask)
        for layer in self.layers:
            seq = layer(seq, room_msk, room_membership.float(), pad_mask, adj_msk)
        nm     = node_mask.float().unsqueeze(-1)
        pooled = (seq * nm).sum(dim=1) / nm.sum(dim=1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


# ── 文本编码器（从零训练）────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    从零训练的 Transformer 文本编码器。
    输入: input_ids [B, T] + attn_mask [B, T]（1=有效）
    输出: L2归一化 embedding [B, d_embed]
    """

    def __init__(self, vocab_size=10000, d_model=256, num_layers=4,
                 num_heads=4, max_len=192, d_embed=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )
        self.register_buffer('pos_ids', torch.arange(max_len).unsqueeze(0))

    def forward(self, input_ids, attn_mask):
        """
        input_ids : [B, T] int64
        attn_mask : [B, T] float32  1=有效 token
        """
        T   = input_ids.size(1)
        x   = self.tok_emb(input_ids) + self.pos_emb(self.pos_ids[:, :T])
        pad = (attn_mask == 0)                    # [B, T] bool, True=padding
        x   = self.transformer(x, src_key_padding_mask=pad)
        tm  = attn_mask.float().unsqueeze(-1)
        pooled = (x * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


# ── CLIP Loss ─────────────────────────────────────────────────────────────────

def clip_loss(graph_emb, text_emb, logit_scale):
    """对称 InfoNCE，返回 (loss, acc_g2t, acc_t2g)。"""
    scale      = logit_scale.exp().clamp(max=100.0)
    logits_g2t = scale * graph_emb @ text_emb.t()
    logits_t2g = logits_g2t.t()
    B      = graph_emb.shape[0]
    labels = torch.arange(B, device=graph_emb.device)
    loss   = (F.cross_entropy(logits_g2t, labels) +
              F.cross_entropy(logits_t2g, labels)) / 2
    with torch.no_grad():
        acc_g2t = (logits_g2t.argmax(dim=1) == labels).float().mean().item()
        acc_t2g = (logits_t2g.argmax(dim=1) == labels).float().mean().item()
    return loss, acc_g2t, acc_t2g


# ── 主模型 ────────────────────────────────────────────────────────────────────

class TextGraphAlign(nn.Module):
    """
    CLIP 风格文本-图对比预训练。文本编码器从零训练。

    forward 输入:
      node_mask      [B, N]
      adj_matrix     [B, N, N]
      room_membership[B, N, MAX_ROOMS]
      input_ids      [B, T]   int64
      attn_mask      [B, T]   float32  1=有效

    forward 返回: (loss, acc_g2t, acc_t2g)
    """

    def __init__(self, vocab_size=10000, d_model=256, num_layers=4, num_heads=4,
                 max_len=192, d_embed=256, dropout=0.1):
        super().__init__()
        self.graph_enc   = GraphEncoder(d_model, num_layers, num_heads, d_embed, dropout)
        self.text_enc    = TextEncoder(vocab_size, d_model, num_layers, num_heads,
                                       max_len, d_embed, dropout)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        g_params = sum(p.numel() for p in self.graph_enc.parameters())
        t_params = sum(p.numel() for p in self.text_enc.parameters())
        print(f"TextGraphAlign  graph_enc={g_params/1e6:.2f}M  "
              f"text_enc={t_params/1e6:.2f}M  "
              f"total={(g_params+t_params)/1e6:.2f}M  "
              f"d_model={d_model}  layers={num_layers}")

    def encode_graph(self, node_mask, adj_matrix, room_membership):
        return self.graph_enc(node_mask, adj_matrix, room_membership)

    def encode_text(self, input_ids, attn_mask):
        return self.text_enc(input_ids, attn_mask)

    def forward(self, node_mask, adj_matrix, room_membership, input_ids, attn_mask):
        g = self.graph_enc(node_mask, adj_matrix, room_membership)
        t = self.text_enc(input_ids, attn_mask)
        return clip_loss(g, t, self.logit_scale)
