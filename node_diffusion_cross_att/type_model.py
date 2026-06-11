import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .model import FeedForward, MultiHeadAttention

N_TYPES = 33   # 0=padding, 1-32=valid types


class GATLayer(nn.Module):
    """
    Single GAT layer: adjacency-masked self-attention + FFN.
    Only nodes connected by an edge can attend to each other.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff    = FeedForward(d_model, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, adj_mask):
        # x        : [B, N, d]
        # adj_mask : [B, N, N]  1=no edge (blocked), 0=has edge
        x2 = self.norm1(x)
        x  = x + self.drop(self.attn(x2, x2, x2, adj_mask))
        x2 = self.norm2(x)
        x  = x + self.drop(self.ff(x2))
        return x


class TextCondGNN(nn.Module):
    """
    Node type classifier using GAT + BERT CLS conditioning.

    Text is encoded to a single CLS vector and added to every node's
    feature before graph message passing — simple global conditioning.

    Input : x             [B, 2, N]   clean coordinates (T=0)
            adj_matrix    [B, N, N]
            node_mask     [B, N]
            prompt_tokens [B, T]      BERT input_ids
            prompt_mask   [B, T]      BERT attention_mask
    Output: logits        [B, N, 33]
    """

    def __init__(self, d_model=384, num_layers=3, num_heads=6,
                 dropout=0.1, bert_name='models/bert-base-uncased',
                 unfreeze_layers=0, n_types=N_TYPES):
        super().__init__()
        self.d_model = d_model

        # coordinate embedding
        self.coord_emb = nn.Linear(2, d_model)

        # BERT text encoder (frozen by default)
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            n_bert = len(self.bert.encoder.layer)
            for layer in self.bert.encoder.layer[n_bert - unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # project CLS [768] → d_model
        self.text_proj = nn.Linear(self.bert.config.hidden_size, d_model)

        # GAT layers
        self.layers = nn.ModuleList(
            [GATLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        # classification head
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_types),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'TextCondGNN: {trainable:,} trainable / {total:,} total')

    def _build_adj_mask(self, adj_matrix, node_mask):
        # block non-edges and padding nodes
        adj_mask = 1 - adj_matrix                          # [B, N, N]
        pad_keys = (1 - node_mask).unsqueeze(1)            # [B, 1, N]
        return torch.clamp(adj_mask + pad_keys, 0, 1)      # [B, N, N]

    def forward(self, x, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        x = x.permute(0, 2, 1).float()        # [B, N, 2]

        node_feat = self.coord_emb(x)          # [B, N, d]

        # encode text → CLS → broadcast to all nodes
        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                cls = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state[:, 0, :]   # [B, 768]
            text_feat = self.text_proj(cls).unsqueeze(1)   # [B, 1, d]
            node_feat = node_feat + text_feat              # broadcast add

        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        for layer in self.layers:
            node_feat = layer(node_feat, adj_mask)

        return self.type_head(node_feat)       # [B, N, 33]


# keep old name as alias so existing import references don't break
NodeTypeClassifier = TextCondGNN
