"""
NodeDiffusionTransformer + Room ID Embedding (实验变体)

与 node_diffusion_cross_att/model.py 的唯一区别：
  在 Node Feature Init 额外注入 room_embed(room_id)，
  room_id 由邻接矩阵拓扑自动推算（BFS最小环检测 + union-find）。

room_id=0 保留为 padding_idx（节点不属于任何环，或填充节点）。
"""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

N_TYPES   = 32
MAX_ROOMS = 20   # 最多同时支持的房间数（多余的截断到 MAX_ROOMS）


# ── Room ID 分配 ──────────────────────────────────────────────────────────────

def _assign_room_ids_single(adj, n):
    """
    对单张图的邻接矩阵（numpy bool [n,n]），用逐边 BFS 检测最小环，
    按"先到先得"为每个节点分配一个 room ID：
      - 每发现一个新的最小环，给它分配一个新 rid
      - 环中尚未分配 room 的节点获得该 rid
      - 已有 room 的节点（共享角点）保留原 rid，不被覆盖

    这样两个房间共享的角点会保留第一个环的 rid，而不会把两个房间合并。

    返回: list[int] 长度 n，值域 [0, MAX_ROOMS]
         0 = 不属于任何环
         1..MAX_ROOMS = 房间编号
    """
    node_rid = [0] * n
    next_rid = 1
    seen = set()

    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            # BFS: u → v，不走直连边 u-v
            prev = {u: -1}
            q = deque([u])
            found = False
            while q and not found:
                cur = q.popleft()
                for w in range(n):
                    if not adj[cur, w] or w in prev:
                        continue
                    if cur == u and w == v:   # 跳过直连边
                        continue
                    prev[w] = cur
                    if w == v:
                        found = True
                        break
                    q.append(w)

            if not found:
                continue

            # 回溯构造环节点列表
            cycle = []
            cur = v
            while cur != -1:
                cycle.append(cur)
                cur = prev[cur]

            key = frozenset(cycle)
            if key in seen:
                continue
            seen.add(key)

            rid = next_rid
            next_rid += 1
            for node in cycle:
                if node_rid[node] == 0:      # 先到先得：未分配才赋值
                    node_rid[node] = rid

    # 将稀疏 rid 映射为 1..K 的连续编号，截断到 MAX_ROOMS
    unique = sorted(set(r for r in node_rid if r > 0))
    remap  = {old: new + 1 for new, old in enumerate(unique)}
    return [min(remap.get(r, 0), MAX_ROOMS) for r in node_rid]


def assign_room_ids(adj_matrix, node_mask):
    """
    批量计算 room_ids。

    adj_matrix : [B, N, N]  float tensor（0/1）
    node_mask  : [B, N]     float tensor（1=有效节点）
    返回       : [B, N]     LongTensor，值域 [0, MAX_ROOMS]
    """
    B, N, _ = adj_matrix.shape
    adj_np  = (adj_matrix > 0.5).cpu().numpy()
    mask_np = node_mask.cpu().numpy()

    out = torch.zeros(B, N, dtype=torch.long)
    for b in range(B):
        n = int(mask_np[b].sum())
        if n < 3:
            continue
        ids = _assign_room_ids_single(adj_np[b, :n, :n], n)
        out[b, :n] = torch.tensor(ids, dtype=torch.long)

    return out.to(adj_matrix.device)


# ── Transformer 基础模块（与 cross_att 完全相同）─────────────────────────────

def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
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
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
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
    1. adj_attn + global_attn (并联相加)
    2. cross_attn (节点查文本)
    3. ffn
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm_cross  = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, adj_mask, text_feat, text_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn(x2, x2, x2, adj_mask) +
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
    与 node_diffusion_cross_att 相比，Node Feature Init 多了 room_embed(room_id)：

        node_emb = input_emb(x) + t_emb + room_embed(room_id)

    room_id 由 assign_room_ids() 从 adj_matrix 在线计算，无需额外数据。
    """

    def __init__(self, model_channels=384, num_layers=6, num_heads=6,
                 dropout=0.1, bpe_vocab_size=None,
                 bert_name='models/bert-base-uncased', unfreeze_layers=0):
        super().__init__()
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)

        # Room ID embedding: 0=padding/no-room, 1..MAX_ROOMS=房间编号
        self.room_embed = nn.Embedding(MAX_ROOMS + 1, model_channels, padding_idx=0)

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
        print(f"NodeDiffusionTransformer(+RoomID): {trainable:,} trainable / {total:,} total parameters")

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, room_ids=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1).float()           # [B, N, 2]

        t_emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                            # [B, 1, d]

        # ── Room ID embedding ─────────────────────────────────────────────────
        # 优先使用 dataset 预计算的 room_ids（快）；
        # 若未提供（推理时直接调用）则在线计算（慢，fallback）。
        if room_ids is None:
            room_ids = assign_room_ids(adj_matrix.float(), node_mask.float())
        room_ids = room_ids.long().to(x.device)
        rid_emb  = self.room_embed(room_ids)                               # [B, N, d]

        node_emb = self.input_emb(x) + t_emb + rid_emb                    # [B, N, d]

        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state               # [B, T, 768]
            text_feat = self.text_proj(text_hidden)
            text_mask = (1 - bert_attn.float()).unsqueeze(1)
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=node_emb.dtype)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, adj_mask, text_feat, text_mask)

        epsilon_coord = self.coord_head(seq).permute(0, 2, 1)  # [B, 2, N]
        return epsilon_coord
