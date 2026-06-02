import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TYPES = 32  # 节点类型数量（combo ID 1-32，0=padding）


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
    Two attention streams per layer:
      adj_attn    : node-only local attention (adjacency-masked).
      global_attn : full sequence self-attention (text prefix + all nodes).

    Text tokens participate in global_attn but NOT in adj_attn.
    This allows text and nodes to attend to each other bidirectionally
    through global_attn, while preserving graph-structure locality via adj_attn.
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff      = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, T, adj_mask, pad_mask):
        """
        x        : [B, T+N, d]  concatenated text prefix + node features
        T        : int           number of text prefix tokens
        adj_mask : [B, N, N]    node adjacency mask (1=blocked)
        pad_mask : [B, T+N, T+N] global padding mask (1=blocked)
        """
        x2 = self.norm1(x)

        # adj_attn: only on node portion
        node_x2  = x2[:, T:, :]                                    # [B, N, d]
        adj_out  = self.adj_attn(node_x2, node_x2, node_x2, adj_mask)  # [B, N, d]

        # global_attn: full sequence (text + nodes)
        global_out = self.global_attn(x2, x2, x2, pad_mask)        # [B, T+N, d]

        # Update: text positions use only global_attn; node positions use both
        x_text  = x[:, :T, :] + self.dropout(global_out[:, :T, :])
        x_nodes = x[:, T:, :] + self.dropout(adj_out) + self.dropout(global_out[:, T:, :])
        x = torch.cat([x_text, x_nodes], dim=1)

        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    """
    Epsilon-prediction Transformer for node-coordinate diffusion.

    Text tokens are prepended as a prefix to the node sequence so that
    text and nodes attend to each other bidirectionally (global_attn),
    while nodes additionally use adjacency-masked local attention (adj_attn).

    Input  : x             [B, 2, N]    noisy (x,y) coordinates  (N=40)
    Cond   : adj_matrix    [B, N, N]    adjacency (1=connected)
             node_mask     [B, N]       1=valid node, 0=padding
             prompt_tokens [B, T]       WordPiece token IDs
             prompt_mask   [B, T]       1=valid token, 0=PAD
    Output : epsilon       [B, 2, N]    predicted noise
             type_logits   [B, N, 33]   node type logits (0=pad, 1-32=types)
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
        self.type_head = nn.Linear(model_channels, N_TYPES + 1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeDiffusionTransformer: {n_params:,} parameters")

    def _build_adj_mask(self, adj_matrix, node_mask):
        """Build adjacency mask for node-only attention. [B, N, N]"""
        adj_mask  = 1 - adj_matrix
        pad_keys  = (1 - node_mask).unsqueeze(1)
        adj_mask  = torch.clamp(adj_mask + pad_keys, 0, 1)
        return adj_mask

    def _build_pad_mask(self, text_pad, node_pad):
        """
        Build global padding mask for full sequence [B, T+N, T+N].
        text_pad : [B, T]  1=masked
        node_pad : [B, N]  1=masked
        """
        full_key = torch.cat([text_pad, node_pad], dim=1)          # [B, T+N]
        return full_key.unsqueeze(1).expand(-1, full_key.shape[1], -1)  # [B, T+N, T+N]

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B = x.shape[0]
        x = x.permute(0, 2, 1).float()                             # [B, N, 2]

        t_emb     = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)                                              # [B, 1, d]
        node_emb  = self.input_emb(x) + t_emb                      # [B, N, d]

        adj_mask  = self._build_adj_mask(adj_matrix.float(), node_mask.float())
        node_pad  = (1 - node_mask.float())                         # [B, N]

        if prompt_tokens is not None:
            text_emb  = self.text_embed(prompt_tokens)              # [B, T, d]
            T         = text_emb.shape[1]
            text_pad  = (1 - prompt_mask.float()) if prompt_mask is not None \
                        else (prompt_tokens == 0).float()           # [B, T]
            pad_mask  = self._build_pad_mask(text_pad, node_pad)    # [B, T+N, T+N]
            seq       = torch.cat([text_emb, node_emb], dim=1)      # [B, T+N, d]
        else:
            T        = 0
            pad_mask = node_pad.unsqueeze(1).expand(B, node_emb.shape[1], -1)
            seq      = node_emb                                      # [B, N, d]

        for layer in self.layers:
            seq = layer(seq, T, adj_mask, pad_mask)

        node_out    = seq[:, T:, :]                                 # [B, N, d]
        epsilon     = self.coord_head(node_out).permute(0, 2, 1)    # [B, 2, N]
        type_logits = self.type_head(node_out)                      # [B, N, 33]

        return epsilon, type_logits
