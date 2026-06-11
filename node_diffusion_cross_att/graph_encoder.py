"""
GraphEncoder: 将 (adj_matrix, node_coords, node_types, node_mask) 编码为固定维度向量。
用于对比预训练，与 BERT 文本嵌入对齐。
"""

import torch
import torch.nn as nn

from .model import MultiHeadAttention, FeedForward


class GraphEncoderLayer(nn.Module):
    """邻接 masked 自注意力 + FFN（无 cross-attention）。"""

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = MultiHeadAttention(heads, d_model, dropout)
        self.ff      = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask):
        x2 = self.norm1(x)
        x  = x + self.dropout(self.attn(x2, x2, x2, adj_mask))
        x2 = self.norm2(x)
        x  = x + self.dropout(self.ff(x2))
        return x


class GraphEncoder(nn.Module):
    """
    Input:
        coords     : [B, N, 2]   float (原始整数坐标，内部除以 COORD_SCALE 归一化)
        adj_matrix : [B, N, N]   float
        node_mask  : [B, N]      float  (1=有效节点)
        node_types : [B, N]      long   (1~32, padding=0)
    Output:
        [B, embed_dim]  未归一化向量（外部做 F.normalize）
    """

    COORD_SCALE = 256.0   # 坐标归一化常数

    def __init__(self, d_model=384, num_layers=4, num_heads=6,
                 embed_dim=512, n_types=32, dropout=0.1):
        super().__init__()
        self.type_embed  = nn.Embedding(n_types + 1, d_model, padding_idx=0)
        self.coord_embed = nn.Linear(2, d_model)
        self.layers      = nn.ModuleList(
            [GraphEncoderLayer(d_model, num_heads, dropout)
             for _ in range(num_layers)]
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
        )

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def forward(self, coords, adj_matrix, node_mask, node_types):
        coords   = coords.float() / self.COORD_SCALE
        x        = self.coord_embed(coords) + self.type_embed(node_types)
        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        for layer in self.layers:
            x = layer(x, adj_mask)

        mask_f = node_mask.float().unsqueeze(-1)                         # [B, N, 1]
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)    # [B, d_model]
        return self.proj(pooled)                                          # [B, embed_dim]
