import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff      = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, T, adj_mask, pad_mask):
        x2 = self.norm1(x)
        node_x2    = x2[:, T:, :]
        adj_out    = self.adj_attn(node_x2, node_x2, node_x2, adj_mask)
        global_out = self.global_attn(x2, x2, x2, pad_mask)
        x_text  = x[:, :T, :] + self.dropout(global_out[:, :T, :])
        x_nodes = x[:, T:, :] + self.dropout(adj_out) + self.dropout(global_out[:, T:, :])
        x = torch.cat([x_text, x_nodes], dim=1)
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    """
    Coordinate-only diffusion model.

    Input  : x            [B, 2, N]   noisy coordinates
    Cond   : adj_matrix, node_mask, prompt_tokens, prompt_mask
    Output : epsilon_coord [B, 2, N]  predicted coordinate noise
    """

    def __init__(self, model_channels=256, num_layers=6, num_heads=4,
                 dropout=0.1, bpe_vocab_size=10000):
        super().__init__()
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb  = nn.Linear(2, model_channels)
        self.text_embed = nn.Embedding(bpe_vocab_size, model_channels, padding_idx=0)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeDiffusionTransformer: {n_params:,} parameters")

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def _build_pad_mask(self, text_pad, node_pad):
        full_key = torch.cat([text_pad, node_pad], dim=1)
        return full_key.unsqueeze(1).expand(-1, full_key.shape[1], -1)

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B = x.shape[0]
        x = x.permute(0, 2, 1).float()                              # [B, N, 2]

        t_emb    = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                                               # [B, 1, d]

        node_emb = self.input_emb(x) + t_emb                        # [B, N, d]

        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())
        node_pad = (1 - node_mask.float())

        if prompt_tokens is not None:
            text_emb = self.text_embed(prompt_tokens)
            T        = text_emb.shape[1]
            text_pad = (1 - prompt_mask.float()) if prompt_mask is not None \
                       else (prompt_tokens == 0).float()
            pad_mask = self._build_pad_mask(text_pad, node_pad)
            seq      = torch.cat([text_emb, node_emb], dim=1)
        else:
            T        = 0
            pad_mask = node_pad.unsqueeze(1).expand(B, node_emb.shape[1], -1)
            seq      = node_emb

        for layer in self.layers:
            seq = layer(seq, T, adj_mask, pad_mask)

        node_out      = seq[:, T:, :]
        epsilon_coord = self.coord_head(node_out).permute(0, 2, 1)  # [B, 2, N]

        return epsilon_coord
