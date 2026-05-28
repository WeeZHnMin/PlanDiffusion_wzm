import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BERT_HIDDEN = 768


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
        # Use a dtype-safe minimum so AMP/fp16 masking does not overflow.
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, torch.finfo(scores.dtype).min)
    scores = F.softmax(scores, dim=-1)
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
    三流注意力：
      adj_attn    — 只看直接相邻节点（adj_mask）
      global_attn — 看所有有效节点（pad_mask）
      cross_attn  — cross-attend 到文本 hidden states（可选）
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.cross_attn  = MultiHeadAttention(heads, d_model, dropout)
        self.ff      = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, pad_mask, encoder_hidden=None):
        # 图结构双流
        x2 = self.norm1(x)
        x  = x \
           + self.dropout(self.adj_attn(x2, x2, x2, adj_mask)) \
           + self.dropout(self.global_attn(x2, x2, x2, pad_mask))

        # 文本 cross-attention（graph_only 模式传 None 跳过）
        if encoder_hidden is not None:
            x2 = self.norm3(x)
            x  = x + self.dropout(self.cross_attn(x2, encoder_hidden, encoder_hidden))

        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    """
    x0-prediction Transformer，支持两种条件模式：

    graph_only  (encoder_hidden=None):
        只用邻接矩阵 + 节点掩码，退化成原有双流模型。

    text+graph  (encoder_hidden 非 None):
        BERT hidden states 通过 cross-attention 注入每一层，
        adj_matrix 继续作为空间注意力掩码。

    Input  : x               [B, 2, 40]  加噪坐标
    Cond   : adj_matrix      [B, 40, 40]
             node_mask       [B, 40]
             encoder_hidden  [B, L, model_channels]  (可选，文本特征)
    Output : x0_pred         [B, 2, 40]
    """

    def __init__(self, model_channels=256, num_layers=6, num_heads=4,
                 dropout=0.1, text_hidden=BERT_HIDDEN):
        super().__init__()
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)

        # BERT hidden → model_channels 投影（graph_only 时不用）
        self.text_proj = nn.Linear(text_hidden, model_channels)

        self.layers = nn.ModuleList([
            EncoderLayer(model_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeDiffusionTransformer: {n_params:,} parameters")

    def _build_masks(self, adj_matrix, node_mask):
        adj_mask = (1 - adj_matrix)
        pad_keys = (1 - node_mask).unsqueeze(1)
        adj_mask = torch.clamp(adj_mask + pad_keys, 0, 1)
        pad_mask = pad_keys.expand_as(adj_mask)
        return adj_mask, pad_mask

    def forward(self, x, timesteps, adj_matrix, node_mask,
                encoder_hidden=None, **kwargs):
        """
        encoder_hidden : [B, L, BERT_HIDDEN] 或 None
        """
        x = x.permute(0, 2, 1).float()                        # [B, 40, 2]

        t_emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                                         # [B, 1, C]

        out = self.input_emb(x) + t_emb                       # [B, 40, C]

        adj_mask, pad_mask = self._build_masks(
            adj_matrix.float(), node_mask.float()
        )

        # 投影文本特征
        enc = self.text_proj(encoder_hidden) if encoder_hidden is not None else None

        for layer in self.layers:
            out = layer(out, adj_mask, pad_mask, encoder_hidden=enc)

        out = self.output_head(out)                            # [B, 40, 2]
        return out.permute(0, 2, 1)                            # [B, 2, 40]
