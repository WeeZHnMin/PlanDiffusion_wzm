"""
NodeDiffusionTransformer — BGE-Joint 12层架构

每个 BGEJointLayer:
  1. 节点三流 QKV (adj / room / global) + 文本 QKV (BGE-small 权重初始化)
  2. K_joint = cat([K_adj, K_room, K_global, K_text])  [B, h, 3N+L, d_k]
     V_joint = cat([V_adj, V_room, V_global, V_text])
  3. 各 Q 使用不同 mask 对 K_joint/V_joint 计算：
       Q_adj:    K_adj 部分用邻接 mask，其余只用 padding mask
       Q_room:   K_room 部分用环 mask，其余只用 padding mask
       Q_global: 只用 padding mask
       Q_text:   只用 padding mask
  4. node_out = out_adj + out_room + out_global → Linear → 残差 + LN
     text_out = out_text → Linear → 残差 + LN
  5. 各自 FFN (GELU) + LN
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


# ── 时间步嵌入 ────────────────────────────────────────────────────────────────

def timestep_embedding(timesteps, dim):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                         device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ── BGE Joint Layer ───────────────────────────────────────────────────────────

class BGEJointLayer(nn.Module):
    """
    节点三流 QKV + 文本 QKV 共享 K_joint / V_joint。

    K_joint = cat([K_adj, K_room, K_global, K_text])  [B, h, 3N+L, d_k]
    各 Q 对 K_joint/V_joint 计算，node_out = sum(三流输出)。
    """

    def __init__(self, d_model=384, heads=12, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.h   = heads
        d = d_model

        # 节点三流 QKV
        self.q_adj    = nn.Linear(d, d)
        self.k_adj    = nn.Linear(d, d)
        self.v_adj    = nn.Linear(d, d)

        self.q_room   = nn.Linear(d, d)
        self.k_room   = nn.Linear(d, d)
        self.v_room   = nn.Linear(d, d)

        self.q_global = nn.Linear(d, d)
        self.k_global = nn.Linear(d, d)
        self.v_global = nn.Linear(d, d)

        self.out_node    = nn.Linear(d, d)
        self.norm_node   = nn.LayerNorm(d)
        self.ff_node_up  = nn.Linear(d, d * 2)
        self.ff_node_dn  = nn.Linear(d * 2, d)
        self.norm_node_ff = nn.LayerNorm(d)

        # 文本 QKV（BGE 权重将从外部加载）
        self.q_text   = nn.Linear(d, d)
        self.k_text   = nn.Linear(d, d)
        self.v_text   = nn.Linear(d, d)
        self.out_text    = nn.Linear(d, d)
        self.norm_text   = nn.LayerNorm(d)
        self.ff_text_up  = nn.Linear(d, d * 4)   # BGE intermediate: 384→1536
        self.ff_text_dn  = nn.Linear(d * 4, d)   # 1536→384
        self.norm_text_ff = nn.LayerNorm(d)

        self.dropout = nn.Dropout(dropout)

    def _proj(self, linear, x, L):
        B = x.shape[0]
        return linear(x).view(B, L, self.h, self.d_k).transpose(1, 2)  # [B,h,L,d_k]

    def _attn(self, q, k_joint, v_joint, mask=None):
        scores = torch.matmul(q, k_joint.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        scores = F.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(self.dropout(scores), v_joint)

    def forward(self, node_seq, text_seq,
                adj_joint_mask=None, room_joint_mask=None, global_joint_mask=None):
        """
        adj_joint_mask:  [B, 1, N, 3N+L]  Q_adj 专用
        room_joint_mask: [B, 1, N, 3N+L]  Q_room 专用
        global_joint_mask:[B, 1, 1, 3N+L] Q_global / Q_text 共用（仅 padding）
        """
        B, N, d = node_seq.shape
        L = text_seq.shape[1]

        p = self._proj

        # 构建 joint K/V  [B, h, 3N+L, d_k]
        K_joint = torch.cat([
            p(self.k_adj,    node_seq, N),
            p(self.k_room,   node_seq, N),
            p(self.k_global, node_seq, N),
            p(self.k_text,   text_seq, L),
        ], dim=2)
        V_joint = torch.cat([
            p(self.v_adj,    node_seq, N),
            p(self.v_room,   node_seq, N),
            p(self.v_global, node_seq, N),
            p(self.v_text,   text_seq, L),
        ], dim=2)

        def merge(x, seq_len):
            return x.transpose(1, 2).contiguous().view(B, seq_len, d)

        # 三流各自 mask 不同
        node_attn = self.out_node(
            merge(self._attn(p(self.q_adj,    node_seq, N), K_joint, V_joint, adj_joint_mask),  N) +
            merge(self._attn(p(self.q_room,   node_seq, N), K_joint, V_joint, room_joint_mask), N) +
            merge(self._attn(p(self.q_global, node_seq, N), K_joint, V_joint, global_joint_mask), N)
        )
        # 文本 Q 只用 padding mask
        text_attn = self.out_text(
            merge(self._attn(p(self.q_text, text_seq, L), K_joint, V_joint, global_joint_mask), L)
        )

        # 残差 + post-norm
        node_seq = self.norm_node(node_seq + self.dropout(node_attn))
        text_seq = self.norm_text(text_seq + self.dropout(text_attn))

        # FFN
        node_seq = self.norm_node_ff(node_seq + self.dropout(
            self.ff_node_dn(F.gelu(self.ff_node_up(node_seq)))))
        text_seq = self.norm_text_ff(text_seq + self.dropout(
            self.ff_text_dn(F.gelu(self.ff_text_up(text_seq)))))

        return node_seq, text_seq


# ── 主模型 ────────────────────────────────────────────────────────────────────

class NodeDiffusionTransformer(nn.Module):
    """
    BGE-Joint 12层扩散模型。

    cond 需含:
      node_mask       [B, N]
      room_membership [B, N, MAX_ROOMS]
      adj_matrix      [B, N, N]
      prompt_tokens   [B, T]
      prompt_mask     [B, T]
    """

    def __init__(self, model_channels=384, num_heads=12, dropout=0.1,
                 bert_name='models/bge-small-en-v1.5',
                 freeze_text_emb=True):
        super().__init__()
        self.model_channels = model_channels
        d = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.input_emb = nn.Linear(2, d)
        self.type_emb  = nn.Embedding(33, d)   # 0=pad, 1-32=combo types

        # BGE: token embeddings + 12 encoder layers
        print(f'加载 BGE 权重: {bert_name}')
        bge = BertModel.from_pretrained(bert_name)

        # 复用 BGE 的 token/position/type embedding + LayerNorm
        self.text_embeddings = bge.embeddings

        # 12 层联合注意力
        self.layers = nn.ModuleList([
            BGEJointLayer(d, num_heads, dropout) for _ in range(12)])

        # 将 BGE 文本侧权重加载进每一层
        def _copy(dst, src):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        for i, layer in enumerate(self.layers):
            bl = bge.encoder.layer[i]
            sa = bl.attention.self
            ao = bl.attention.output
            _copy(layer.q_text,  sa.query)
            _copy(layer.k_text,  sa.key)
            _copy(layer.v_text,  sa.value)
            _copy(layer.out_text, ao.dense)
            layer.norm_text.weight.data.copy_(ao.LayerNorm.weight.data)
            layer.norm_text.bias.data.copy_(ao.LayerNorm.bias.data)
            _copy(layer.ff_text_up,  bl.intermediate.dense)
            _copy(layer.ff_text_dn,  bl.output.dense)
            layer.norm_text_ff.weight.data.copy_(bl.output.LayerNorm.weight.data)
            layer.norm_text_ff.bias.data.copy_(bl.output.LayerNorm.bias.data)

        del bge

        if freeze_text_emb:
            for p in self.text_embeddings.parameters():
                p.requires_grad = False

        self.coord_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d // 2),
            nn.Linear(d // 2, 2),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'NodeDiffusionTransformer(BGE-Joint 12L): '
              f'{trainable:,} trainable / {total:,} total')

    def _build_room_mask(self, room_membership, node_mask):
        """[B, N, N]  1=不在同一环或 padding，需屏蔽"""
        dt      = room_membership.dtype
        room_nn = torch.bmm(room_membership, room_membership.transpose(1, 2))
        mask    = (room_nn == 0).to(dt)
        pad_key = (1 - node_mask.to(dt)).unsqueeze(1)          # [B, 1, N]
        return torch.clamp(mask + pad_key, 0, 1)               # [B, N, N]

    def _build_adj_mask(self, adj_matrix, node_mask):
        """[B, N, N]  1=不邻接或 padding，需屏蔽"""
        dt  = adj_matrix.dtype
        eye = torch.eye(adj_matrix.shape[1], device=adj_matrix.device,
                        dtype=dt).unsqueeze(0)
        connected = (adj_matrix + eye).clamp(0, 1)
        mask      = 1 - connected
        pad_key   = (1 - node_mask.to(dt)).unsqueeze(1)
        return torch.clamp(mask + pad_key, 0, 1)               # [B, N, N]

    def encode_text(self, prompt_tokens, prompt_mask=None):
        """
        返回 BGE token embeddings（DDIM 循环外预计算一次）。
        text_feat: [B, T, 384]   text_mask: [B, T]  1=padding
        """
        text_seq = self.text_embeddings(prompt_tokens)        # [B, T, 384]
        if prompt_mask is not None:
            text_key_mask = (1 - prompt_mask.float())
        else:
            text_key_mask = (prompt_tokens == 0).float()
        return text_seq, text_key_mask

    def forward(self, x, timesteps, node_mask,
                prompt_tokens=None, prompt_mask=None,
                text_feat=None, text_mask=None,
                room_membership=None, adj_matrix=None,
                node_combo_ids=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1)   # [B, N, 2]

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)).unsqueeze(1)
        node_seq = self.input_emb(x) + t_emb                  # [B, N, d]
        if node_combo_ids is not None:
            node_seq = node_seq + self.type_emb(
                node_combo_ids.long().clamp(0, 32).to(x.device))

        # ── 文本初始化 ────────────────────────────────────────────────────────
        if text_feat is not None:
            text_seq      = text_feat.to(dtype=node_seq.dtype)
            text_key_mask = (text_mask.squeeze(1) if text_mask is not None
                             else torch.zeros(B, text_feat.shape[1],
                                              device=node_seq.device)).to(node_seq.device)
        elif prompt_tokens is not None:
            text_seq, text_key_mask = self.encode_text(prompt_tokens, prompt_mask)
            text_seq = text_seq.to(node_seq.dtype)
        else:
            text_seq      = torch.zeros(B, 1, self.model_channels,
                                        device=node_seq.device, dtype=node_seq.dtype)
            text_key_mask = torch.zeros(B, 1, device=node_seq.device)

        L = text_seq.shape[1]
        dt = node_seq.dtype

        # ── 结构性 mask（adj / room）[B, N, N] ───────────────────────────────
        room_mb  = room_membership.to(device=node_seq.device, dtype=dt) \
                   if room_membership is not None \
                   else torch.zeros(B, N, MAX_ROOMS, device=node_seq.device, dtype=dt)
        adj_mat  = adj_matrix.to(device=node_seq.device, dtype=dt) \
                   if adj_matrix is not None \
                   else torch.zeros(B, N, N, device=node_seq.device, dtype=dt)

        adj_nn  = self._build_adj_mask(adj_mat,  node_mask.to(dt))   # [B, N, N]
        room_nn = self._build_room_mask(room_mb, node_mask.to(dt))   # [B, N, N]

        node_pad  = (1 - node_mask.float())                           # [B, N]
        text_pad  = text_key_mask.to(node_seq.device)                 # [B, L]

        # 扩展 padding mask 到 [B, N, N] 和 [B, N, L]
        node_pad_nn = node_pad.unsqueeze(1).expand(-1, N, -1)         # [B, N, N]
        text_pad_nl = text_pad.unsqueeze(1).expand(-1, N, -1)         # [B, N, L]

        # Q_adj mask：K_adj 用邻接 mask，其余 padding  [B, 1, N, 3N+L]
        adj_joint_mask = torch.cat(
            [adj_nn, node_pad_nn, node_pad_nn, text_pad_nl], dim=2
        ).unsqueeze(1).bool()

        # Q_room mask：K_room 用环 mask，其余 padding  [B, 1, N, 3N+L]
        room_joint_mask = torch.cat(
            [node_pad_nn, room_nn, node_pad_nn, text_pad_nl], dim=2
        ).unsqueeze(1).bool()

        # Q_global / Q_text mask：只有 padding  [B, 1, 1, 3N+L]
        global_joint_mask = torch.cat(
            [node_pad, node_pad, node_pad, text_pad], dim=1
        ).unsqueeze(1).unsqueeze(2).bool()

        # ── 12 层前向 ─────────────────────────────────────────────────────────
        for layer in self.layers:
            node_seq, text_seq = layer(
                node_seq, text_seq,
                adj_joint_mask, room_joint_mask, global_joint_mask)

        return self.coord_head(node_seq).permute(0, 2, 1)     # [B, 2, N]
