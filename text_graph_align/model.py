"""
TextGraphAlign: CLIP 风格对比预训练

GraphEncoder : node_coords + adj_matrix → adj_attn + global_attn → mean pool → MLP投影头 → L2归一化
TextEncoder  : BERT（解冻最后 unfreeze_layers 层）→ CLS token → MLP投影头 → L2归一化
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


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
    """adj_attn（直接相邻）+ room_attn（同环）+ global_attn（全局）三流。"""

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.room_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff          = FeedForward(d_model, dropout)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, adj_mask, room_mask, pad_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(
            self.adj_attn   (x2, x2, x2, adj_mask)  +
            self.room_attn  (x2, x2, x2, room_mask) +
            self.global_attn(x2, x2, x2, pad_mask)
        )
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


# ── 图编码器 ──────────────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """
    输入: node_coords [B, N, 2]，adj_matrix [B, N, N]，node_mask [B, N]
    输出: L2 归一化 embedding [B, d_embed]
    """

    def __init__(self, d_model=384, num_layers=4, num_heads=6,
                 d_embed=384, dropout=0.1):
        super().__init__()
        self.node_input = nn.Linear(2, d_model)
        self.layers     = nn.ModuleList(
            [GraphEncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_embed),
        )

    def _build_masks(self, adj_matrix, node_mask, room_membership):
        dt  = adj_matrix.dtype
        nm  = node_mask.to(dt)
        pad_msk = (1 - nm).unsqueeze(1)                                   # [B, 1, N]

        eye = torch.eye(adj_matrix.shape[1], device=adj_matrix.device, dtype=dt).unsqueeze(0)
        adj_msk = torch.clamp(1 - (adj_matrix + eye).clamp(0, 1) +
                              pad_msk, 0, 1)                               # [B, N, N]

        room_nn   = torch.bmm(room_membership.to(dt),
                              room_membership.to(dt).transpose(1, 2))     # [B, N, N]
        room_msk  = torch.clamp((room_nn == 0).to(dt) + pad_msk, 0, 1)   # [B, N, N]

        return adj_msk, room_msk, pad_msk

    def forward(self, node_coords, adj_matrix, node_mask, room_membership):
        seq = self.node_input(node_coords.float())
        adj_msk, room_msk, pad_msk = self._build_masks(
            adj_matrix.float(), node_mask.float(), room_membership.float())
        for layer in self.layers:
            seq = layer(seq, adj_msk, room_msk, pad_msk)
        nm     = node_mask.float().unsqueeze(-1)
        pooled = (seq * nm).sum(dim=1) / nm.sum(dim=1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


# ── 文本编码器（BERT）────────────────────────────────────────────────────────

class BertTextEncoder(nn.Module):
    """
    BERT + 解冻最后 unfreeze_layers 层 + MLP投影头 → L2归一化。
    输入: prompt_tokens [B, T] int64, prompt_mask [B, T] float (1=有效)
    输出: L2归一化 embedding [B, d_embed]
    """

    def __init__(self, bert_name='models/bert-base-uncased',
                 unfreeze_layers=4, d_embed=384):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            n_bert = len(self.bert.encoder.layer)
            for layer in self.bert.encoder.layer[n_bert - unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        bert_hidden = self.bert.config.hidden_size   # 768
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden),
            nn.ReLU(),
            nn.Linear(bert_hidden, d_embed),
        )

    def forward(self, prompt_tokens, prompt_mask):
        bert_attn = prompt_mask.long() if prompt_mask is not None \
                    else (prompt_tokens != 0).long()
        if any(p.requires_grad for p in self.bert.parameters()):
            out = self.bert(input_ids=prompt_tokens, attention_mask=bert_attn)
        else:
            with torch.no_grad():
                out = self.bert(input_ids=prompt_tokens, attention_mask=bert_attn)
        cls = out.last_hidden_state[:, 0, :]   # CLS token
        return F.normalize(self.proj(cls), dim=-1)


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
        acc_g2t = (logits_g2t.argmax(dim=1) == labels).float().mean()
        acc_t2g = (logits_t2g.argmax(dim=1) == labels).float().mean()
    return loss, acc_g2t, acc_t2g


# ── 主模型 ────────────────────────────────────────────────────────────────────

class TextGraphAlign(nn.Module):
    """
    CLIP 风格文本-布局对比预训练。

    forward 输入:
      node_coords    [B, N, 2]   float，节点坐标
      adj_matrix     [B, N, N]   float，二值邻接矩阵
      node_mask      [B, N]      float，1=有效节点
      prompt_tokens  [B, T]      int64，BERT token ids
      prompt_mask    [B, T]      float，1=有效 token

    forward 返回: (loss, acc_g2t, acc_t2g)
    """

    def __init__(self, bert_name='models/bert-base-uncased',
                 unfreeze_layers=4, d_model=384, num_layers=4,
                 num_heads=6, d_embed=384, dropout=0.1):
        super().__init__()
        self.graph_enc   = GraphEncoder(d_model, num_layers, num_heads, d_embed, dropout)
        self.text_enc    = BertTextEncoder(bert_name, unfreeze_layers, d_embed)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        g_params  = sum(p.numel() for p in self.graph_enc.parameters())
        t_total   = sum(p.numel() for p in self.text_enc.parameters())
        t_train   = sum(p.numel() for p in self.text_enc.parameters() if p.requires_grad)
        print(f"TextGraphAlign  graph_enc={g_params/1e6:.2f}M  "
              f"bert={t_total/1e6:.2f}M(trainable={t_train/1e6:.2f}M)  "
              f"d_embed={d_embed}  d_model={d_model}  layers={num_layers}")

    def forward(self, node_coords, adj_matrix, node_mask, room_membership,
                prompt_tokens, prompt_mask):
        g = self.graph_enc(node_coords, adj_matrix, node_mask, room_membership)
        t = self.text_enc(prompt_tokens, prompt_mask)
        loss, _, _ = clip_loss(g, t, self.logit_scale)
        return loss

    @torch.no_grad()
    def compute_metrics(self, node_coords, adj_matrix, node_mask, room_membership,
                        prompt_tokens, prompt_mask):
        g = self.graph_enc(node_coords, adj_matrix, node_mask, room_membership)
        t = self.text_enc(prompt_tokens, prompt_mask)
        _, acc_g2t, acc_t2g = clip_loss(g, t, self.logit_scale)
        return acc_g2t.item(), acc_t2g.item()
