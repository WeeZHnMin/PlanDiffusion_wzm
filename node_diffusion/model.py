import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        scores = scores.masked_fill(mask.unsqueeze(1) == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
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
      adj_attn    : attends only to directly connected neighbors (local).
      global_attn : attends to all valid nodes + text tokens (global).
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.adj_attn    = MultiHeadAttention(heads, d_model, dropout)
        self.global_attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff      = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask, pad_mask, text_kv=None, text_key_mask=None):
        """
        x            : (B, 40, d)
        adj_mask     : (B, 40, 40)  1=ignore
        pad_mask     : (B, 40, 40)  1=ignore（节点padding）
        text_kv      : (B, T, d)    文本token嵌入，T=max_text_len
        text_key_mask: (B, T)       1=PAD文本token，0=有效
        """
        x2 = self.norm1(x)

        if text_kv is not None:
            # 拼接节点和文本作为 K/V
            kv = torch.cat([x2, text_kv], dim=1)          # (B, 40+T, d)

            # 组合 mask：节点padding + 文本padding → (B, 40, 40+T)
            # pad_mask 当前是 (B, 40, 40)，取第一行（每个query看到相同的key padding）
            node_key_pad = pad_mask[:, :1, :]              # (B, 1, 40)
            text_key_pad = text_key_mask.unsqueeze(1)      # (B, 1, T)
            combined_mask = torch.cat(
                [node_key_pad.expand(-1, x.shape[1], -1),
                 text_key_pad.expand(-1, x.shape[1], -1)],
                dim=2
            )                                              # (B, 40, 40+T)
            global_out = self.global_attn(x2, kv, kv, combined_mask)
        else:
            global_out = self.global_attn(x2, x2, x2, pad_mask)

        x = (
            x
            + self.dropout(self.adj_attn(x2, x2, x2, adj_mask))
            + self.dropout(global_out)
        )
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class NodeDiffusionTransformer(nn.Module):
    """
    Epsilon-prediction Transformer for node-coordinate diffusion.

    Input  : x             [B, 2, 40]    noisy (x,y) coordinates
    Cond   : adj_matrix    [B, 40, 40]   adjacency (1=connected)
             node_mask     [B, 40]       1=valid node, 0=padding
             prompt_tokens [B, T]        BPE token IDs (optional)
             prompt_mask   [B, T]        1=valid token, 0=PAD (optional)
    Output : epsilon [B, 2, 40]  predicted noise
    """

    def __init__(self, model_channels=256, num_layers=6, num_heads=4,
                 dropout=0.1, bpe_vocab_size=12000):
        super().__init__()
        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_emb = nn.Linear(2, model_channels)

        # 文本token嵌入（BPE词表部分，ID 0~bpe_vocab_size-1）
        self.text_embed = nn.Embedding(bpe_vocab_size, model_channels, padding_idx=0)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, 2),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeDiffusionTransformer: {n_params:,} parameters")

    def _build_masks(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        adj_mask = torch.clamp(adj_mask + pad_keys, 0, 1)
        pad_mask = pad_keys.expand_as(adj_mask)
        return adj_mask, pad_mask

    def forward(self, x, timesteps, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        x = x.permute(0, 2, 1).float()

        t_emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        ).unsqueeze(1)

        out = self.input_emb(x) + t_emb
        adj_mask, pad_mask = self._build_masks(adj_matrix.float(), node_mask.float())

        # 文本嵌入
        text_kv       = None
        text_key_mask = None
        if prompt_tokens is not None:
            text_kv = self.text_embed(prompt_tokens)           # (B, T, d)
            if prompt_mask is not None:
                text_key_mask = (1 - prompt_mask.float())      # 1=PAD，0=有效
            else:
                text_key_mask = (prompt_tokens == 0).float()   # 用PAD ID推断

        for layer in self.layers:
            out = layer(out, adj_mask, pad_mask, text_kv, text_key_mask)

        out = self.output_head(out)
        return out.permute(0, 2, 1)
