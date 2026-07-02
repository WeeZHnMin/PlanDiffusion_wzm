"""
NodeDiffusionTransformer — 三流自注意力 + MMDiT 联合注意力

每个 EncoderLayer：
  1. adj_attn + room_attn + global_attn（并联相加）
  2. JointAttention（MMDiT 风格：节点↔文本双向，各自独立 QKV 权重）
  3. FFN（节点 + 文本各自独立）

JointAttention 联合 key mask [B, N+T]：
  节点 padding（node_mask=0）和文本 padding（prompt_mask=0）一并屏蔽。
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
    """mask: [B, N, N] 或 [B, 1, N]，1=屏蔽。"""
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


# ── MMDiT 联合注意力 ──────────────────────────────────────────────────────────

class JointAttention(nn.Module):
    """
    节点和文本各自独立一套 QKV 权重，但共享拼接后的 KV 序列进行注意力。

    joint_key_mask: [B, N+T]  1=屏蔽（节点 padding + 文本 padding）
    输入:  node_seq [B, N, d]，text_seq [B, T, d]
    输出:  out_node [B, N, d]，out_text [B, T, d]
    """
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h   = heads
        # 节点侧
        self.q_node  = nn.Linear(d_model, d_model)
        self.k_node  = nn.Linear(d_model, d_model)
        self.v_node  = nn.Linear(d_model, d_model)
        self.out_node = nn.Linear(d_model, d_model)
        # 文本侧
        self.q_text  = nn.Linear(d_model, d_model)
        self.k_text  = nn.Linear(d_model, d_model)
        self.v_text  = nn.Linear(d_model, d_model)
        self.out_text = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, q, k_joint, v_joint, key_mask):
        """
        q:        [B, h, L, d_k]
        k_joint:  [B, h, N+T, d_k]
        key_mask: [B, 1, 1, N+T]   1=屏蔽
        """
        scores = torch.matmul(q, k_joint.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(key_mask == 1, -1e4)
        scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
        scores = self.dropout(scores)
        return torch.matmul(scores, v_joint)

    def forward(self, node_seq, text_seq, joint_key_mask):
        B, N, d = node_seq.shape
        T = text_seq.shape[1]

        def proj_split(linear, x, L):
            return linear(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        # 节点侧 QKV
        q_n = proj_split(self.q_node, node_seq, N)
        k_n = proj_split(self.k_node, node_seq, N)
        v_n = proj_split(self.v_node, node_seq, N)

        # 文本侧 QKV
        q_t = proj_split(self.q_text, text_seq, T)
        k_t = proj_split(self.k_text, text_seq, T)
        v_t = proj_split(self.v_text, text_seq, T)

        # 联合 KV [B, h, N+T, d_k]
        k_joint = torch.cat([k_n, k_t], dim=2)
        v_joint = torch.cat([v_n, v_t], dim=2)

        # mask 扩展为 [B, 1, 1, N+T]
        mask = joint_key_mask.unsqueeze(1).unsqueeze(2)

        out_n = self._attn(q_n, k_joint, v_joint, mask)   # [B, h, N, d_k]
        out_t = self._attn(q_t, k_joint, v_joint, mask)   # [B, h, T, d_k]

        out_n = out_n.transpose(1, 2).contiguous().view(B, N, d)
        out_t = out_t.transpose(1, 2).contiguous().view(B, T, d)

        return self.out_node(out_n), self.out_text(out_t)


# ── Encoder Layer ─────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    1. adj_attn + room_attn + global_attn（并联相加）
    2. JointAttention（节点↔文本双向，各自独立 QKV）
    3. FFN（节点 + 文本各自独立）
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # 节点三流
        self.norm1       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        # 联合注意力
        self.norm_node_joint = nn.LayerNorm(d_model)
        self.norm_text_joint = nn.LayerNorm(d_model)
        self.joint_attn      = JointAttention(heads, d_model, dropout)
        # 节点 FFN
        self.norm_node_ff = nn.LayerNorm(d_model)
        self.ff_node      = FeedForward(d_model, dropout)
        # 文本 FFN
        self.norm_text_ff = nn.LayerNorm(d_model)
        self.ff_text      = FeedForward(d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_seq, text_seq,
                room_mask, joint_key_mask,
                pad_mask=None, adj_mask=None):
        # ── 三流节点自注意力 ──────────────────────────────────────────────────
        x2 = self.norm1(node_seq)
        node_seq = node_seq + self.dropout(
            self.adj_attn   (x2, x2, x2, adj_mask) +
            self.room_attn  (x2, x2, x2, room_mask) +
            self.global_attn(x2, x2, x2, pad_mask)
        )

        # ── MMDiT 联合注意力（节点↔文本双向）────────────────────────────────
        x_node = self.norm_node_joint(node_seq)
        x_text = self.norm_text_joint(text_seq)
        out_node, out_text = self.joint_attn(x_node, x_text, joint_key_mask)
        node_seq = node_seq + self.dropout(out_node)
        text_seq = text_seq + self.dropout(out_text)

        # ── FFN ──────────────────────────────────────────────────────────────
        node_seq = node_seq + self.dropout(self.ff_node(self.norm_node_ff(node_seq)))
        text_seq = text_seq + self.dropout(self.ff_text(self.norm_text_ff(text_seq)))

        return node_seq, text_seq


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    三流自注意力 + MMDiT 联合注意力版本。

    cond 中需含有：
      node_mask       [B, N]
      room_membership [B, N, MAX_ROOMS]
      adj_matrix      [B, N, N]
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
        self.type_emb  = nn.Embedding(33, model_channels)  # 0=pad, 1-32=combo types

        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        n_bert = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[n_bert - unfreeze_layers:]:
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
        print(f"NodeDiffusionTransformer(tri-stream + MMDiT joint): "
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

    def encode_text(self, prompt_tokens, prompt_mask=None):
        """
        预计算 BERT 文本特征，推理时在 diffusion 循环外调用一次。

        返回:
          text_seq      [B, T, d]   投影后的文本序列（作为联合注意力初始输入）
          text_key_mask [B, T]      1=屏蔽（padding 位）
        """
        bert_attn = prompt_mask if prompt_mask is not None \
                    else (prompt_tokens != 0).long()
        with torch.no_grad():
            text_hidden = self.bert(
                input_ids=prompt_tokens,
                attention_mask=bert_attn.long(),
            ).last_hidden_state
        text_seq      = self.text_proj(text_hidden)       # [B, T, d]
        text_key_mask = (1 - bert_attn.float())           # [B, T]  1=masked
        return text_seq, text_key_mask

    def forward(self, x, timesteps, node_mask,
                prompt_tokens=None, prompt_mask=None,
                text_feat=None, text_mask=None,
                room_membership=None, adj_matrix=None,
                node_combo_ids=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1)                            # [B, N, 2]

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)
        node_emb = self.input_emb(x) + t_emb              # [B, N, d]
        if node_combo_ids is not None:
            node_emb = node_emb + self.type_emb(
                node_combo_ids.long().clamp(0, 32).to(x.device))  # [B, N, d]

        dt = node_emb.dtype
        room_membership = room_membership.to(device=x.device, dtype=dt)
        room_mask, pad_mask = self._build_room_mask(
            room_membership, node_mask.to(dt))

        if adj_matrix is not None:
            adj_mask = self._build_adj_mask(adj_matrix.to(device=x.device, dtype=dt),
                                            node_mask.to(dt))
        else:
            adj_mask = pad_mask

        # ── 文本序列初始化 ────────────────────────────────────────────────────
        # text_feat/text_mask: 推理时由 encode_text 预计算传入
        # text_mask 在新版中是 [B, T]（1=masked），兼容旧版 [B, 1, T] 自动 squeeze
        if text_feat is not None:
            text_seq = text_feat.to(dtype=dt)
            if text_mask is not None:
                text_key_mask = text_mask.squeeze(1).to(dtype=dt)   # [B, T]
            else:
                text_key_mask = torch.zeros(
                    B, text_seq.shape[1], device=x.device, dtype=dt)
        elif prompt_tokens is not None:
            text_seq, text_key_mask = self.encode_text(prompt_tokens, prompt_mask)
            text_seq      = text_seq.to(dt)
            text_key_mask = text_key_mask.to(dt)
        else:
            T = 1
            text_seq      = torch.zeros(B, T, self.model_channels,
                                        device=node_emb.device, dtype=dt)
            text_key_mask = torch.zeros(B, T, device=node_emb.device, dtype=dt)

        # ── 联合 key mask [B, N+T]：节点 padding + 文本 padding ──────────────
        node_key_mask  = (1 - node_mask.to(dt))                         # [B, N]
        joint_key_mask = torch.cat([node_key_mask, text_key_mask], dim=1)  # [B, N+T]

        # ── 逐层前向（text_seq 在层间流动）──────────────────────────────────
        seq = node_emb
        for layer in self.layers:
            seq, text_seq = layer(seq, text_seq,
                                  room_mask, joint_key_mask,
                                  pad_mask, adj_mask)

        return self.coord_head(seq).permute(0, 2, 1)      # [B, 2, N]
