"""
NodeDiffusionTransformer — 四流自注意力 (Quad-Stream Self-Attention)

每个 EncoderLayer 并联四路自注意力，结果直接相加：
  - adj_attn   : 直接相邻节点（局部边约束）
  - room_attn  : 同环所有节点（面约束）
  - global_attn: 全部有效节点（全局布局）
  - bnd_attn   : 轮廓节点互见（建筑外轮廓空间约束）

轮廓节点（is_boundary=1）在整个扩散过程中坐标固定（不加噪声），
其嵌入不携带时间步信息，作为空间参考被所有节点通过 bnd_attn 查询。
non_bnd_mask 屏蔽内部节点作为 Key，使所有节点只能 attend 到轮廓节点。

adj_mask 构造:
  connected = adj_matrix + eye
  adj_mask  = 1 - connected.clamp(0,1)   →  1=屏蔽

room_mask 构造:
  room_nn   = membership @ membership.T
  room_mask = (room_nn == 0)             →  1=屏蔽

non_bnd_mask 构造:
  non_bnd_mask = 1 - is_boundary         →  1=屏蔽（内部节点列），0=可见（轮廓节点列）
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
    四流自注意力层：
      1. adj_attn + room_attn + global_attn + bnd_attn（并联相加）
      2. cross_attn（节点查文本）
      3. ffn
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.bnd_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, room_mask, text_feat, text_mask,
                pad_mask=None, adj_mask=None, non_bnd_mask=None):
        x2 = self.norm1(x)
        attn_out = (
            self.adj_attn   (x2, x2, x2, adj_mask)  +
            self.room_attn  (x2, x2, x2, room_mask) +
            self.global_attn(x2, x2, x2, pad_mask)  +
            self.bnd_attn   (x2, x2, x2, non_bnd_mask)
        )
        x  = x + self.dropout(attn_out)
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    四流自注意力：adj_attn + room_attn + global_attn + bnd_attn。

    cond 中需含有：
      node_mask       [B, N]
      room_membership [B, N, MAX_ROOMS]
      adj_matrix      [B, N, N]   二值邻接矩阵
      prompt_tokens   [B, T]
      prompt_mask     [B, T]
      is_boundary     [B, N]      1=轮廓节点（坐标固定），0=内部节点（去噪）
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
        self.text_proj = nn.Linear(self.bert.config.hidden_size, model_channels)

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
        print(f"NodeDiffusionTransformer(adj+room+global+bnd quad-stream): "
              f"{trainable:,} trainable / {total:,} total")

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

    def _build_non_bnd_mask(self, is_boundary):
        """
        non_bnd_mask[b, i, j] = 1 if node j is interior (blocked as Key)
                               = 0 if node j is boundary (visible as Key)
        所有 Query 行均相同：只有轮廓节点列可见。
        """
        # is_boundary: [B, N], 1=boundary, 0=interior
        # blocked if NOT boundary → (1 - is_boundary)
        return (1 - is_boundary).unsqueeze(1).float()   # [B, 1, N] broadcasts to [B, N, N]

    def forward(self, x, timesteps, node_mask,
                prompt_tokens=None, prompt_mask=None,
                room_membership=None, adj_matrix=None,
                is_boundary=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1)                          # [B, N, 2]

        t_emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                                   # [B, 1, d]

        node_emb = self.input_emb(x)                    # [B, N, d]

        # 轮廓节点不携带时间步信息（坐标固定，与 t 无关）
        if is_boundary is not None:
            non_bnd = (1 - is_boundary.float()).unsqueeze(-1).to(node_emb.dtype)  # [B, N, 1]
            node_emb = node_emb + t_emb * non_bnd
        else:
            node_emb = node_emb + t_emb

        dt = node_emb.dtype
        room_membership = room_membership.to(device=x.device, dtype=dt)
        room_mask, pad_mask = self._build_room_mask(
            room_membership, node_mask.to(dt))

        if adj_matrix is not None:
            adj_mask = self._build_adj_mask(adj_matrix.to(device=x.device, dtype=dt),
                                            node_mask.to(dt))
        else:
            adj_mask = pad_mask

        if is_boundary is not None:
            non_bnd_mask = self._build_non_bnd_mask(
                is_boundary.to(device=x.device, dtype=dt))
        else:
            non_bnd_mask = None  # 无轮廓条件时 bnd_attn 做自由全局注意力

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state
            text_feat = self.text_proj(text_hidden).to(dt)
            text_mask = (1 - bert_attn.float()).unsqueeze(1)
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=dt)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, room_mask, text_feat, text_mask,
                        pad_mask, adj_mask, non_bnd_mask)

        return self.coord_head(seq).permute(0, 2, 1)   # [B, 2, N]
