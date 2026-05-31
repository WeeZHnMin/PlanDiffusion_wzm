"""
GraphTransformer：节点+边+全局特征联合注意力去噪网络。

输入：
  Xt         : (B, N, Kx)    noisy 节点特征（one-hot）
  Et         : (B, N, N, Ke) noisy 边特征（one-hot）
  y          : (B, dy)       全局特征（文本 embedding）
  node_mask  : (B, N)        bool，True=有效节点
  t          : (B,)          时间步（归一化到 0~1）

输出：
  pred_X : (B, N, Kx)    预测干净节点 logits
  pred_E : (B, N, N, Ke) 预测干净边 logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 文本编码器 ─────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, bpe_vocab_size: int, embed_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(bpe_vocab_size, embed_dim, padding_idx=0)
        self.proj  = nn.Linear(embed_dim, out_dim)

    def forward(self, tokens: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T)  int
        lens   : (B,)    int
        返回   : (B, out_dim)
        """
        x    = self.embed(tokens)                                # (B, T, embed_dim)
        mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lens.unsqueeze(1)
        x    = x * mask.unsqueeze(-1).float()
        denom = lens.float().clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / denom                           # (B, embed_dim)
        return self.proj(pooled)                                 # (B, out_dim)


# ── NodeEdgeBlock ─────────────────────────────────────────────────────────────
class NodeEdgeBlock(nn.Module):
    def __init__(self, dx: int, de: int, dy: int, n_head: int):
        super().__init__()
        assert dx % n_head == 0
        self.dx     = dx
        self.n_head = n_head
        self.df     = dx // n_head

        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM: E → attention score
        self.e_mul = nn.Linear(de, dx)
        self.e_add = nn.Linear(de, dx)

        # FiLM: y → E
        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM: y → X
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # y 更新
        self.x_to_y = nn.Linear(dx, dy)
        self.e_to_y = nn.Linear(dx, dy)   # 从边聚合到 y（先投影到 dx 维度）
        self.y_proj = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)

    def forward(self, X, E, y, node_mask):
        """
        X: (B, N, dx), E: (B, N, N, de), y: (B, dy)
        node_mask: (B, N) bool
        """
        B, N, _ = X.shape
        x_mask  = node_mask.float().unsqueeze(-1)          # (B, N, 1)
        e_mask  = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).float()  # (B, N, N)

        # Q, K, V
        Q = self.q(X).view(B, N, self.n_head, self.df).unsqueeze(2)   # (B, N, 1, H, df)
        K = self.k(X).view(B, N, self.n_head, self.df).unsqueeze(1)   # (B, 1, N, H, df)
        Y = (Q * K) / math.sqrt(self.df)                               # (B, N, N, H, df)

        # 边 FiLM 调制注意力分数
        E1 = self.e_mul(E).view(B, N, N, self.n_head, self.df)        # (B, N, N, H, df)
        E2 = self.e_add(E).view(B, N, N, self.n_head, self.df)
        Y  = Y * (E1 + 1) + E2                                        # (B, N, N, H, df)

        # 更新边特征
        newE_flat = Y.flatten(3)                                        # (B, N, N, dx)
        ye_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)             # (B, 1, 1, dx)
        ye_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        newE   = ye_add + (ye_mul + 1) * newE_flat                     # (B, N, N, dx)
        newE   = self.e_out(newE) * e_mask.unsqueeze(-1)               # (B, N, N, de)

        # Softmax（屏蔽 padding）
        pad_mask = (~node_mask).float() * -1e9                         # (B, N)
        Y_score  = Y.sum(-1)                                           # (B, N, N, H)
        Y_score  = Y_score + pad_mask.unsqueeze(1).unsqueeze(-1)       # mask key
        attn     = F.softmax(Y_score, dim=2)                           # (B, N, N, H)

        # V 聚合
        V = self.v(X).view(B, N, self.n_head, self.df)                # (B, N, H, df)
        wV = (attn.unsqueeze(-1) * V.unsqueeze(1)).sum(2)             # (B, N, H, df)
        wV = wV.flatten(2)                                             # (B, N, dx)

        # 更新节点
        yx_mul = self.y_x_mul(y).unsqueeze(1)
        yx_add = self.y_x_add(y).unsqueeze(1)
        newX   = yx_add + (yx_mul + 1) * wV                           # (B, N, dx)
        newX   = self.x_out(newX) * x_mask                            # (B, N, dx)

        # 更新全局 y
        x_pool = (X * x_mask).sum(1) / x_mask.sum(1).clamp(min=1)    # (B, dx)
        e_pool_raw = (newE_flat * e_mask.unsqueeze(-1)).sum((1, 2)) / e_mask.sum((1, 2)).unsqueeze(-1).clamp(min=1)
        new_y = y + self.x_to_y(x_pool) + self.e_to_y(e_pool_raw)
        new_y = self.y_proj(new_y)

        return newX, newE, new_y


# ── XEyTransformerLayer ────────────────────────────────────────────────────────
class XEyTransformerLayer(nn.Module):
    def __init__(self, dx: int, de: int, dy: int, n_head: int,
                 dim_ffX: int = 512, dim_ffE: int = 128, dim_ffy: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_x1 = nn.LayerNorm(dx)
        self.norm_e1 = nn.LayerNorm(de)
        self.norm_y1 = nn.LayerNorm(dy)
        self.attn    = NodeEdgeBlock(dx, de, dy, n_head)
        self.drop    = nn.Dropout(dropout)

        self.norm_x2 = nn.LayerNorm(dx)
        self.ff_x    = nn.Sequential(nn.Linear(dx, dim_ffX), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ffX, dx))

        self.norm_e2 = nn.LayerNorm(de)
        self.ff_e    = nn.Sequential(nn.Linear(de, dim_ffE), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ffE, de))

        self.norm_y2 = nn.LayerNorm(dy)
        self.ff_y    = nn.Sequential(nn.Linear(dy, dim_ffy), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ffy, dy))

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.attn(self.norm_x1(X), self.norm_e1(E), self.norm_y1(y), node_mask)
        X = X + self.drop(newX)
        E = E + self.drop(newE)
        y = y + self.drop(new_y)
        X = X + self.ff_x(self.norm_x2(X))
        E = E + self.ff_e(self.norm_e2(E))
        y = y + self.ff_y(self.norm_y2(y))
        return X, E, y


# ── GraphTransformer ──────────────────────────────────────────────────────────
class GraphTransformer(nn.Module):
    def __init__(self,
                 x_classes:      int   = 32,
                 e_classes:      int   = 2,
                 bpe_vocab_size: int   = 12000,
                 text_embed_dim: int   = 128,
                 n_layers:       int   = 6,
                 dx:             int   = 256,
                 de:             int   = 64,
                 dy:             int   = 256,
                 n_head:         int   = 4,
                 dim_ffX:        int   = 512,
                 dim_ffE:        int   = 128,
                 dim_ffy:        int   = 512,
                 dropout:        float = 0.1):
        super().__init__()
        self.x_classes = x_classes
        self.e_classes = e_classes

        # 文本编码器 → y
        self.text_encoder = TextEncoder(bpe_vocab_size, text_embed_dim, dy)

        # 时间步嵌入（正弦）→ 加到 y
        self.t_embed = nn.Sequential(nn.Linear(dy, dy), nn.SiLU(), nn.Linear(dy, dy))

        # 输入投影
        self.mlp_in_x = nn.Sequential(nn.Linear(x_classes + 1, dx), nn.ReLU(), nn.Linear(dx, dx))  # +1 for time
        self.mlp_in_e = nn.Sequential(nn.Linear(e_classes,     de), nn.ReLU(), nn.Linear(de, de))
        # y 已由 text_encoder 给出，不需要额外投影

        # Transformer 层
        self.layers = nn.ModuleList([
            XEyTransformerLayer(dx, de, dy, n_head, dim_ffX, dim_ffE, dim_ffy, dropout)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.mlp_out_x = nn.Sequential(nn.Linear(dx, dx), nn.ReLU(), nn.Linear(dx, x_classes))
        self.mlp_out_e = nn.Sequential(nn.Linear(de, de), nn.ReLU(), nn.Linear(de, e_classes))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'GraphTransformer: {n_params:,} parameters')

    def _sinusoidal_t(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, Xt, Et, node_mask, prompt_tokens, prompt_lens, t):
        """
        Xt           : (B, N, Kx)
        Et           : (B, N, N, Ke)
        node_mask    : (B, N) bool
        prompt_tokens: (B, T) int
        prompt_lens  : (B,)   int
        t            : (B,)   float, 0~1
        """
        B, N = Xt.shape[:2]
        x_mask = node_mask.float().unsqueeze(-1)                   # (B, N, 1)
        e_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).float()

        # 时间步嵌入
        t_emb = self._sinusoidal_t(t * 1000, self.t_embed[0].in_features)  # (B, dy)
        # 只取前 dy 维（如果 sinusoidal 维度和 dy 不同则截断/补零）
        if t_emb.shape[1] != self.t_embed[0].in_features:
            raise ValueError(f"t_emb dim {t_emb.shape[1]} != dy {self.t_embed[0].in_features}")
        t_emb = self.t_embed(t_emb)                                # (B, dy)

        # 文本 y
        y = self.text_encoder(prompt_tokens, prompt_lens) + t_emb  # (B, dy)

        # 把时间步也拼到节点特征里（简单有效）
        t_node = t.unsqueeze(1).unsqueeze(2).expand(B, N, 1)      # (B, N, 1)
        Xt_in  = torch.cat([Xt, t_node], dim=-1)                  # (B, N, Kx+1)

        X = self.mlp_in_x(Xt_in) * x_mask                        # (B, N, dx)
        E = self.mlp_in_e(Et)                                      # (B, N, N, de)
        E = E * e_mask.unsqueeze(-1)

        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)

        pred_X = self.mlp_out_x(X) * x_mask                      # (B, N, Kx)
        pred_E = self.mlp_out_e(E) * e_mask.unsqueeze(-1)         # (B, N, N, Ke)

        # 强制边矩阵对称
        pred_E = (pred_E + pred_E.permute(0, 2, 1, 3)) / 2

        return pred_X, pred_E
