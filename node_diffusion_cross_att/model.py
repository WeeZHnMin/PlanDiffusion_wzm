import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

N_TYPES = 32


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
    1. adj_attn   : 节点间邻接局部注意力（保留图结构先验）
    2. cross_attn : 节点查询文本（Q=节点, K/V=BERT特征），文本为强制依赖路径
    3. ffn
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.adj_attn   = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff         = FeedForward(d_model, dropout)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, adj_mask, text_feat, text_mask):
        # x         : [B, N_nodes, d]
        # text_feat : [B, T_text,  d]  (BERT hidden → projected)
        # adj_mask  : [B, N_nodes, N_nodes]  1=masked
        # text_mask : [B, 1, T_text]         1=padding token

        x2 = self.norm1(x)
        x  = x + self.dropout(self.adj_attn(x2, x2, x2, adj_mask))

        x2 = self.norm_cross(x)
        x  = x + self.dropout(self.cross_attn(x2, text_feat, text_feat, text_mask))

        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    """
    Coordinate-only diffusion model — frozen BERT encoder + cross-attention.

    Input  : x             [B, 2, N]   noisy coordinates
    Cond   : adj_matrix    [B, N, N]
             node_mask     [B, N]
             prompt_tokens [B, T]      BERT input_ids
             prompt_mask   [B, T]      BERT attention_mask (1=valid, 0=padding)
    Output : epsilon_coord [B, 2, N]   predicted coordinate noise
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

        # 冻结 BERT 大部分层，仅解冻最后 unfreeze_layers 个 transformer block
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
        print(f"NodeDiffusionTransformer: {trainable:,} trainable / {total:,} total parameters")

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1).float()           # [B, N, 2]

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                            # [B, 1, d]
        node_emb = self.input_emb(x) + t_emb     # [B, N, d]

        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state               # [B, T, 768]
            text_feat = self.text_proj(text_hidden)             # [B, T, d]
            # cross-attn mask: [B, 1, T], 1=padding → broadcast over all nodes
            text_mask = (1 - bert_attn.float()).unsqueeze(1)    # [B, 1, T]
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=node_emb.dtype)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, adj_mask, text_feat, text_mask)

        epsilon_coord = self.coord_head(seq).permute(0, 2, 1)  # [B, 2, N]
        return epsilon_coord
