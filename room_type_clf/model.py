"""
RoomTypeClassifier: 文本 + 邻接图 → 每个节点的房间类型分类

输入：
  prompt_tokens / prompt_mask  (BERT)
  adj_matrix     [B, N, N]
  room_membership [B, N, MAX_ROOMS]
  node_mask       [B, N]

输出：
  logits [B, N, num_types]  — 有效节点上做交叉熵

特征提取复用 node_diffusion_room_multi 四流注意力：
  adj_attn + room_attn + global_attn + global_room_attn + cross_attn(文本)

节点初始特征：共享可学习 node_token，靠图结构区分各节点。
"""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

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


# ── 基础模块 ──────────────────────────────────────────────────────────────────

def attention(q, k, v, d_k, mask=None, dropout=None):
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
        out = attention(q, k, v, self.d_k, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(out)


class GlobalRoomAttnStream(nn.Module):
    """按环分组的全局注意力，环内独立 softmax。"""
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
            col_mask = (1.0 - mem_k).unsqueeze(1).unsqueeze(2) * 1e4
            scores_k = base_scores - col_mask
            attn_k   = F.softmax(scores_k.float(), dim=-1).to(Q.dtype)
            attn_k   = self.dropout(attn_k)
            out_k    = torch.matmul(attn_k, V)
            node_mask_k = mem_k.unsqueeze(1).unsqueeze(3)
            out_accum   = out_accum   + out_k * node_mask_k
            room_counts = room_counts + node_mask_k
        out = out_accum / room_counts.clamp(min=1.0)
        out = out.transpose(1, 2).contiguous().view(B, N, H * d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """四流自注意力 + cross_attn(文本，可关闭) + FFN。"""
    def __init__(self, d_model, heads, dropout=0.1, use_text=True):
        super().__init__()
        self.use_text         = use_text
        self.norm1            = nn.LayerNorm(d_model)
        self.norm2            = nn.LayerNorm(d_model)
        self.adj_attn         = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn        = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn      = MultiHeadAttention(heads, d_model, dropout)
        self.global_room_attn = GlobalRoomAttnStream(heads, d_model, dropout)
        self.ff               = FeedForward(d_model, dropout)
        self.dropout          = nn.Dropout(dropout)
        if use_text:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(heads, d_model, dropout)

    def forward(self, x, room_mask, room_membership, text_feat=None, text_mask=None,
                pad_mask=None, adj_mask=None):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn        (x2, x2, x2, adj_mask)          +
            self.room_attn       (x2, x2, x2, room_mask)         +
            self.global_attn     (x2, x2, x2, pad_mask)          +
            self.global_room_attn(x2, room_membership, pad_mask)
        )
        if self.use_text and text_feat is not None:
            x2 = self.norm_cross(x)
            x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class RoomTypeClassifier(nn.Module):
    """
    文本 + 邻接图 → 每个节点的房间类型分类。
    节点无坐标输入，用共享可学习 node_token 作初始特征。
    """

    def __init__(self, num_types, model_channels=256, num_layers=4, num_heads=4,
                 dropout=0.1, bert_name='models/bert-base-uncased', use_text=True):
        super().__init__()
        self.model_channels = model_channels
        self.num_types      = num_types
        self.use_text       = use_text

        # 节点初始特征：共享可学习 token
        self.node_token = nn.Parameter(torch.randn(1, 1, model_channels) * 0.02)

        # BERT 文本编码器（冻结，仅 use_text=True 时加载）
        if use_text:
            self.bert = BertModel.from_pretrained(bert_name)
            for p in self.bert.parameters():
                p.requires_grad = False
            self.text_proj = nn.Linear(self.bert.config.hidden_size, model_channels)

        # 四流 Encoder
        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout, use_text=use_text)
             for _ in range(num_layers)]
        )

        # 分类头
        self.type_head = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, num_types),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"RoomTypeClassifier(use_text={use_text}): "
              f"{trainable:,} trainable / {total:,} total  num_types={num_types}")

    def _build_room_mask(self, room_membership, node_mask):
        dt       = room_membership.dtype
        room_nn  = torch.bmm(room_membership, room_membership.transpose(1, 2))
        mask     = (room_nn == 0).to(dt)
        pad_keys = (1 - node_mask.to(dt)).unsqueeze(1)
        return torch.clamp(mask + pad_keys, 0, 1), pad_keys

    def _build_adj_mask(self, adj_matrix, node_mask):
        dt        = adj_matrix.dtype
        eye       = torch.eye(adj_matrix.shape[1], device=adj_matrix.device, dtype=dt).unsqueeze(0)
        connected = (adj_matrix + eye).clamp(0, 1)
        mask      = 1 - connected
        pad_keys  = (1 - node_mask.to(dt)).unsqueeze(1)
        return torch.clamp(mask + pad_keys, 0, 1)

    def encode_text(self, prompt_tokens, prompt_mask=None):
        bert_attn = prompt_mask if prompt_mask is not None \
                    else (prompt_tokens != 0).long()
        with torch.no_grad():
            text_hidden = self.bert(
                input_ids=prompt_tokens,
                attention_mask=bert_attn.long(),
            ).last_hidden_state
        text_feat = self.text_proj(text_hidden)
        text_mask = (1 - bert_attn.float()).unsqueeze(1)
        return text_feat, text_mask

    def forward(self, node_mask, adj_matrix, room_membership,
                prompt_tokens=None, prompt_mask=None):
        B, N = node_mask.shape
        dt   = torch.float32

        # 节点初始特征：所有节点相同，靠图结构区分
        seq = self.node_token.expand(B, N, -1).clone()

        # mask 构造
        room_membership = room_membership.to(dtype=dt)
        node_mask_f     = node_mask.to(dtype=dt)
        room_mask, pad_mask = self._build_room_mask(room_membership, node_mask_f)
        adj_mask            = self._build_adj_mask(adj_matrix.to(dtype=dt), node_mask_f)

        # 文本编码（仅 use_text=True 时执行）
        if self.use_text and prompt_tokens is not None:
            text_feat, text_mask = self.encode_text(prompt_tokens, prompt_mask)
        else:
            text_feat, text_mask = None, None

        # 四流编码
        for layer in self.layers:
            seq = layer(seq, room_mask, room_membership, text_feat, text_mask,
                        pad_mask, adj_mask)

        # 分类
        return self.type_head(seq)   # [B, N, num_types]
