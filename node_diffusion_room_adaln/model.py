"""
NodeDiffusionTransformer — 三流自注意力 + AdaLN 文本条件

原 cross-attn 改为 AdaLN（Adaptive Layer Norm）：
  每层用 BERT CLS token 投影后的全局文本向量控制
  LayerNorm 的 scale/shift，模型无法忽略文本信号。

EncoderLayer:
  adaLN_mod(text_global) → scale1, shift1, scale2, shift2
  norm1(x) * (1+scale1) + shift1  → tri-stream attn
  norm2(x) * (1+scale2) + shift2  → FFN

adaLN_mod 的输出线性层零初始化（DiT trick），
使训练初期 AdaLN 等价于普通 LayerNorm，不破坏稳定性。
"""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

N_TYPES   = 32
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
            q     = __import__('collections').deque([u])
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


# ── Transformer 基础模块 ──────────────────────────────────────────────────────

def timestep_embedding(timesteps, dim):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                         device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


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
        self.d_k = d_model // heads
        self.h   = heads
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


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    三流自注意力 + AdaLN 文本条件（取代 cross-attn）。

    text_global: [B, d_model]  — BERT CLS 投影后的全局文本向量
    adaLN_mod 输出 4*d: scale1, shift1（注意力前）, scale2, shift2（FFN前）
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

        # AdaLN: text_global → scale1, shift1, scale2, shift2
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model),
        )
        # 零初始化：训练初期等价于普通 LayerNorm
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, text_global, room_mask, pad_mask=None, adj_mask=None):
        # text_global: [B, d]  →  scale/shift: [B, d]
        mod = self.adaLN_mod(text_global)                      # [B, 4*d]
        s1, b1, s2, b2 = mod.chunk(4, dim=-1)                 # each [B, d]

        # 三流注意力（AdaLN 调制）
        x2 = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        x  = x + self.dropout(
            self.adj_attn   (x2, x2, x2, adj_mask)  +
            self.room_attn  (x2, x2, x2, room_mask) +
            self.global_attn(x2, x2, x2, pad_mask)
        )

        # FFN（AdaLN 调制）
        x2 = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    三流自注意力 + AdaLN 文本条件。

    encode_text 返回 (text_global [B, d], None)。
    forward 中 text_feat 参数即 text_global [B, d]，text_mask 忽略。
    """

    def __init__(self, model_channels=384, num_layers=6, num_heads=6,
                 dropout=0.1, bert_name='models/bert-base-uncased',
                 unfreeze_layers=0):
        super().__init__()
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)

        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        n_layers = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[n_layers - unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # CLS token → model_channels
        self.text_global_proj = nn.Linear(self.bert.config.hidden_size, model_channels)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"NodeDiffusionTransformer(tri-stream + AdaLN): {trainable:,} trainable / {total:,} total")

    def _build_room_mask(self, room_membership, node_mask):
        dt       = room_membership.dtype
        room_nn  = torch.bmm(room_membership, room_membership.transpose(1, 2))
        mask     = (room_nn == 0).to(dt)
        pad_keys = (1 - node_mask.to(dt)).unsqueeze(1)
        return torch.clamp(mask + pad_keys, 0, 1), pad_keys

    def _build_adj_mask(self, adj_matrix, node_mask):
        dt  = adj_matrix.dtype
        eye = torch.eye(adj_matrix.shape[1], device=adj_matrix.device, dtype=dt).unsqueeze(0)
        connected = (adj_matrix + eye).clamp(0, 1)
        mask      = 1 - connected
        pad_keys  = (1 - node_mask.to(dt)).unsqueeze(1)
        return torch.clamp(mask + pad_keys, 0, 1)

    def encode_text(self, prompt_tokens, prompt_mask=None):
        """
        BERT CLS token → text_global [B, model_channels]。
        返回 (text_global, None)，接口与 tri 版兼容。
        """
        bert_attn = prompt_mask if prompt_mask is not None \
                    else (prompt_tokens != 0).long()
        with torch.no_grad():
            cls_hidden = self.bert(
                input_ids=prompt_tokens,
                attention_mask=bert_attn,
            ).last_hidden_state[:, 0, :]          # CLS token [B, bert_hidden]
        text_global = self.text_global_proj(cls_hidden)   # [B, model_channels]
        return text_global, None

    def forward(self, x, timesteps, node_mask,
                prompt_tokens=None, prompt_mask=None,
                text_feat=None, text_mask=None,
                room_membership=None, adj_matrix=None, **kwargs):
        del kwargs, text_mask
        B, _, N = x.shape
        x = x.permute(0, 2, 1)               # [B, N, 2]

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)
        node_emb = self.input_emb(x) + t_emb  # [B, N, d]

        dt = node_emb.dtype
        room_membership = room_membership.to(device=x.device, dtype=dt)
        room_mask, pad_mask = self._build_room_mask(
            room_membership, node_mask.to(dt))

        if adj_matrix is not None:
            adj_mask = self._build_adj_mask(
                adj_matrix.to(device=x.device, dtype=dt), node_mask.to(dt))
        else:
            adj_mask = pad_mask

        # text_feat = text_global [B, d]
        if text_feat is not None:
            text_global = text_feat.to(dtype=dt)
        elif prompt_tokens is not None:
            text_global, _ = self.encode_text(prompt_tokens, prompt_mask)
            text_global = text_global.to(dt)
        else:
            text_global = torch.zeros(B, self.model_channels,
                                      device=node_emb.device, dtype=dt)

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, text_global, room_mask, pad_mask, adj_mask)

        return self.coord_head(seq).permute(0, 2, 1)   # [B, 2, N]
