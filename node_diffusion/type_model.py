import torch
import torch.nn as nn

from .model import MultiHeadAttention, FeedForward, EncoderLayer

N_TYPES = 33   # 0=padding, 1-32=valid types


class NodeTypeClassifier(nn.Module):
    """
    Node type classifier.

    Input : x [B, 2, N]  clean coordinates (T=0)
            adj_matrix   [B, N, N]
            node_mask    [B, N]
            prompt_tokens [B, T_tok]
            prompt_mask   [B, T_tok]
    Output: logits [B, N, 33]
    """

    def __init__(self, model_channels=384, num_layers=6, num_heads=6,
                 dropout=0.1, bpe_vocab_size=10000, n_types=N_TYPES):
        super().__init__()
        self.model_channels = model_channels

        self.input_emb  = nn.Linear(2, model_channels)
        self.text_embed = nn.Embedding(bpe_vocab_size, model_channels, padding_idx=0)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )

        self.type_head = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, n_types),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NodeTypeClassifier: {n_params:,} parameters")

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def _build_pad_mask(self, text_pad, node_pad):
        full_key = torch.cat([text_pad, node_pad], dim=1)
        return full_key.unsqueeze(1).expand(-1, full_key.shape[1], -1)

    def forward(self, x, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B = x.shape[0]
        x = x.permute(0, 2, 1).float()     # [B, N, 2]

        node_emb = self.input_emb(x)        # [B, N, d]

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

        node_out = seq[:, T:, :]            # [B, N, d]
        logits   = self.type_head(node_out) # [B, N, 33]
        return logits
