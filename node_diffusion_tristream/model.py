"""
NodeDiffusionTransformer + 三流注意力 (Tri-Stream Attention)

在双流（adj_attn + global_attn）基础上新增第三路 room_attn：
  - adj_attn  : 直接相邻边（局部边约束）
  - room_attn : 同环所有节点（环内几何约束）
  - global_attn: 全部节点（全局布局）

room_mask 由 room_membership [B,N,MAX_ROOMS] 推导：
  room_nn   = membership @ membership.T   →  [B,N,N]
  room_mask = (room_nn == 0).float()      →  1=屏蔽，0=允许

节点可同时属于多个环（共享角点），room_mask 天然处理多隶属问题。
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
    """
    对单张图的邻接矩阵（numpy bool [n,n]），逐边 BFS 检测最小环，
    返回 [n, MAX_ROOMS] 二值 numpy 矩阵。
    membership[i, r] = 1 表示节点 i 属于第 r 个环。
    一个节点可属于多个环（共享角点），不再先到先得。
    """
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


def assign_room_membership(adj_matrix, node_mask):
    """
    批量计算 room_membership。

    adj_matrix : [B, N, N]  float tensor
    node_mask  : [B, N]     float tensor
    返回       : [B, N, MAX_ROOMS]  FloatTensor，二值
    """
    B, N, _ = adj_matrix.shape
    adj_np   = (adj_matrix > 0.5).cpu().numpy()
    mask_np  = node_mask.cpu().numpy()

    out = torch.zeros(B, N, MAX_ROOMS, dtype=torch.float32)
    for b in range(B):
        n = int(mask_np[b].sum())
        if n < 3:
            continue
        m = _assign_room_membership_single(adj_np[b, :n, :n], n)
        out[b, :n, :] = torch.from_numpy(m)

    return out.to(adj_matrix.device)


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
    双流注意力层：
      1. room_attn + global_attn（并联相加）
         adj_attn 已被 room_attn 完全包含（平面图中每条边必属于某个环），故去除。
      2. cross_attn（节点查文本）
      3. ffn
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.room_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, room_mask, text_feat, text_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.room_attn  (x2, x2, x2, room_mask) +
            self.global_attn(x2, x2, x2, None)
        )
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    三流注意力：adj_attn + room_attn + global_attn。

    room_mask 由 room_membership [B,N,MAX_ROOMS] 推导：
        room_nn   = membership @ membership.T   →  [B,N,N]
        room_mask = (room_nn == 0)              →  1=屏蔽（不共享任何环）

    节点初始特征：node_emb = input_emb(x) + t_emb
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
        print(f"NodeDiffusionTransformer(room_attn+global_attn): {trainable:,} trainable / {total:,} total")

    def _build_room_mask(self, room_membership, node_mask):
        """
        room_membership: [B, N, MAX_ROOMS]
        返回 room_mask  : [B, N, N]，1=屏蔽（两节点不共享任何环）
        """
        room_nn  = torch.bmm(room_membership, room_membership.transpose(1, 2))
        mask     = (room_nn == 0).float()
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(mask + pad_keys, 0, 1)

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None,
                room_membership=None, **kwargs):
        # adj_matrix 仅用于在线计算 room_membership（推理时无预计算的情况）
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1).float()

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)
        node_emb = self.input_emb(x) + t_emb

        if room_membership is None:
            room_membership = assign_room_membership(adj_matrix.float(), node_mask.float())
        room_membership = room_membership.float().to(x.device)
        room_mask = self._build_room_mask(room_membership, node_mask.float())

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state
            text_feat = self.text_proj(text_hidden)
            text_mask = (1 - bert_attn.float()).unsqueeze(1)
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=node_emb.dtype)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, room_mask, text_feat, text_mask)

        return self.coord_head(seq).permute(0, 2, 1)   # [B, 2, N]
