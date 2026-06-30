"""
NodeDiffusionTransformer — 四流自注意力 (Quad-Stream Self-Attention)

每个 EncoderLayer 并联四路自注意力，结果直接相加：
  - adj_attn        : 直接相邻节点（局部边约束，adj_matrix=1 的位置允许）
  - room_attn       : 同环所有节点（面约束，room_membership 推导）
  - global_attn     : 全部有效节点（全局布局，仅屏蔽 padding）
  - global_room_attn: 按环分组的全局注意力（环内独立 softmax，共享角点传递跨环信息）

四路独立学习不同几何归纳偏置，共享 FFN 和 cross-attn。

adj_mask 构造:
  connected = adj_matrix + eye (允许自注意力)
  adj_mask  = 1 - connected.clamp(0,1)   →  1=屏蔽

room_mask 构造:
  room_nn   = membership @ membership.T
  room_mask = (room_nn == 0)             →  1=屏蔽（不共享任何环）

global_room_attn 机制:
  scores = Q @ K^T / sqrt(d_k)           # 计算一次
  for k in rooms:
      scores_k = scores; 屏蔽非环k的 Key 列
      out_k    = softmax(scores_k) @ V   # 环内独立归一化
      属于环k的节点行累加 out_k
  output = 累加结果 / 所属环数            # 按环数平均
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


class GlobalRoomAttnStream(nn.Module):
    """
    四流注意力中的 global_room_attn：
      1. 用独立 QKV 权重计算原始注意力分数 scores [B, H, N, N]（一次）
      2. 对每个房间环 k：屏蔽非环节点的 Key 列 → 独立 softmax → 查询 V
      3. 每个节点的输出 = 其所属环的 out_k 之和 / 所属环数
    共享角点（属于多环）同时参与多个环的注意力，实现跨环间接通信。
    """
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
        """
        x:               [B, N, d]
        room_membership: [B, N, MAX_ROOMS]  float, 1=属于该环
        pad_mask:        [B, 1, N]  float, 1=padding key（屏蔽）
        """
        B, N, d = x.shape
        H, d_k  = self.h, self.d_k

        Q = self.q_linear(x).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.k_linear(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, N, H, d_k).transpose(1, 2)

        # 原始分数（共享，只算一次）[B, H, N, N]
        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 先屏蔽 padding key（在环循环外处理，避免重复计算）
        if pad_mask is not None:
            # pad_mask [B, 1, N] → unsqueeze → [B, 1, 1, N]，广播到 [B, H, N, N]
            base_scores = base_scores - pad_mask.unsqueeze(2) * 1e9

        R = room_membership.shape[2]  # MAX_ROOMS

        out_accum   = torch.zeros(B, H, N, d_k, device=x.device, dtype=Q.dtype)
        room_counts = torch.zeros(B, 1, N, 1,   device=x.device, dtype=Q.dtype)

        for k in range(R):
            mem_k = room_membership[:, :, k]  # [B, N]

            if mem_k.sum() == 0:
                continue

            # 屏蔽不属于环 k 的 Key 列：(1 - mem_k)[B,N] → [B,1,1,N]
            col_mask = (1.0 - mem_k).unsqueeze(1).unsqueeze(2) * 1e9
            scores_k = base_scores - col_mask  # [B, H, N, N]

            attn_k = F.softmax(scores_k.float(), dim=-1).to(Q.dtype)
            attn_k = self.dropout(attn_k)
            out_k  = torch.matmul(attn_k, V)   # [B, H, N, d_k]

            # 只累加属于环 k 的节点行：mem_k[B,N] → [B,1,N,1]
            node_mask_k = mem_k.unsqueeze(1).unsqueeze(3)
            out_accum   = out_accum   + out_k  * node_mask_k
            room_counts = room_counts + node_mask_k

        # 按所属环数平均（防止除零）
        out = out_accum / room_counts.clamp(min=1.0)  # [B, H, N, d_k]

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
    """
    四流自注意力层：
      1. adj_attn + room_attn + global_attn + global_room_attn（并联相加）
      2. cross_attn（节点查文本）
      3. ffn
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1            = nn.LayerNorm(d_model)
        self.norm_cross       = nn.LayerNorm(d_model)
        self.norm2            = nn.LayerNorm(d_model)
        self.adj_attn         = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn        = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn      = MultiHeadAttention(heads, d_model, dropout)
        self.global_room_attn = GlobalRoomAttnStream(heads, d_model, dropout)
        self.cross_attn       = MultiHeadAttention(heads, d_model, dropout)
        self.ff               = FeedForward(d_model, dropout)
        self.dropout          = nn.Dropout(dropout)

    def forward(self, x, room_mask, room_membership, text_feat, text_mask,
                pad_mask=None, adj_mask=None):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn        (x2, x2, x2, adj_mask)          +  # 直接相邻（含自连）
            self.room_attn       (x2, x2, x2, room_mask)         +  # 同环节点
            self.global_attn     (x2, x2, x2, pad_mask)          +  # 全局（屏蔽 padding）
            self.global_room_attn(x2, room_membership, pad_mask)    # 按环分组全局注意力
        )
        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    四流自注意力：adj_attn + room_attn + global_attn + global_room_attn。

    cond 中需含有：
      node_mask       [B, N]
      room_membership [B, N, MAX_ROOMS]
      adj_matrix      [B, N, N]   二值邻接矩阵
      prompt_tokens   [B, T]
      prompt_mask     [B, T]
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
        print(f"NodeDiffusionTransformer(adj+room+global+global_room quad-stream): {trainable:,} trainable / {total:,} total")

    def _build_room_mask(self, room_membership, node_mask):
        """
        room_membership: [B, N, MAX_ROOMS]
        返回:
          room_mask : [B, N, N]  1=屏蔽（不共享环 或 padding key）
          pad_mask  : [B, 1, N]  1=屏蔽（padding key），供 global_attn 使用
        """
        dt       = room_membership.dtype
        room_nn  = torch.bmm(room_membership, room_membership.transpose(1, 2))
        mask     = (room_nn == 0).to(dt)
        pad_keys = (1 - node_mask.to(dt)).unsqueeze(1)          # [B, 1, N]
        room_mask = torch.clamp(mask + pad_keys, 0, 1)
        return room_mask, pad_keys

    def _build_adj_mask(self, adj_matrix, node_mask):
        """
        adj_matrix: [B, N, N] float，1=有边，0=无边
        返回 adj_mask: [B, N, N] 1=屏蔽（非直接相邻 或 padding key）
        对角线视为自连接（允许自注意力）。
        """
        dt  = adj_matrix.dtype
        eye = torch.eye(adj_matrix.shape[1], device=adj_matrix.device, dtype=dt).unsqueeze(0)
        connected = (adj_matrix + eye).clamp(0, 1)   # 有边 or 自身
        mask      = 1 - connected                     # 1=屏蔽
        pad_keys  = (1 - node_mask.to(dt)).unsqueeze(1)          # [B, 1, N]
        return torch.clamp(mask + pad_keys, 0, 1)

    def encode_text(self, prompt_tokens, prompt_mask=None):
        """预计算 BERT 文本特征，推理时在 diffusion 循环外调用一次。"""
        bert_attn = prompt_mask if prompt_mask is not None \
                    else (prompt_tokens != 0).long()
        with torch.no_grad():
            text_hidden = self.bert(
                input_ids=prompt_tokens,
                attention_mask=bert_attn,
            ).last_hidden_state
        text_feat = self.text_proj(text_hidden)
        text_mask = (1 - bert_attn.float()).unsqueeze(1)
        return text_feat, text_mask

    def forward(self, x, timesteps, node_mask,
                prompt_tokens=None, prompt_mask=None,
                text_feat=None, text_mask=None,
                room_membership=None, adj_matrix=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1)                          # [B, N, 2]，保留 AMP dtype

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)
        node_emb = self.input_emb(x) + t_emb            # [B, N, d]

        dt = node_emb.dtype
        room_membership = room_membership.to(device=x.device, dtype=dt)
        room_mask, pad_mask = self._build_room_mask(
            room_membership, node_mask.to(dt))

        if adj_matrix is not None:
            adj_mask = self._build_adj_mask(adj_matrix.to(device=x.device, dtype=dt),
                                            node_mask.to(dt))
        else:
            adj_mask = pad_mask

        # 优先使用预计算的 text_feat（推理时避免每步重复跑 BERT）
        if text_feat is not None:
            text_feat = text_feat.to(dtype=dt)
            if text_mask is not None:
                text_mask = text_mask.to(dtype=dt)
        elif prompt_tokens is not None:
            text_feat, text_mask = self.encode_text(prompt_tokens, prompt_mask)
            text_feat = text_feat.to(dt)
            text_mask = text_mask.to(dt)
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=dt)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, room_mask, room_membership, text_feat, text_mask, pad_mask, adj_mask)

        return self.coord_head(seq).permute(0, 2, 1)   # [B, 2, N]
