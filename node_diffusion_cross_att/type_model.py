import torch
import torch.nn as nn
from transformers import BertModel

from .model import EncoderLayer

N_TYPES = 33   # 0=padding, 1-32=valid types


class NodeTypeClassifier(nn.Module):
    """
    Node type classifier: predict node type from clean coordinates + graph + text.

    Input : x             [B, 2, N]   clean coordinates (T=0)
            adj_matrix    [B, N, N]
            node_mask     [B, N]
            prompt_tokens [B, T]      BERT input_ids
            prompt_mask   [B, T]      BERT attention_mask
    Output: logits        [B, N, 33]
    """

    def __init__(self, model_channels=384, num_layers=6, num_heads=6,
                 dropout=0.1, bert_name='models/bert-base-uncased',
                 unfreeze_layers=0, n_types=N_TYPES):
        super().__init__()
        self.model_channels = model_channels

        self.input_emb = nn.Linear(2, model_channels)

        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            n_bert = len(self.bert.encoder.layer)
            for layer in self.bert.encoder.layer[n_bert - unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_proj = nn.Linear(self.bert.config.hidden_size, model_channels)

        self.layers = nn.ModuleList(
            [EncoderLayer(model_channels, num_heads, dropout) for _ in range(num_layers)]
        )

        self.type_head = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, n_types),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'NodeTypeClassifier: {trainable:,} trainable / {total:,} total')

    def _build_adj_mask(self, adj_matrix, node_mask):
        adj_mask = 1 - adj_matrix
        pad_keys = (1 - node_mask).unsqueeze(1)
        return torch.clamp(adj_mask + pad_keys, 0, 1)

    def forward(self, x, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B, _, N = x.shape
        x = x.permute(0, 2, 1).float()       # [B, N, 2]

        node_emb = self.input_emb(x)          # [B, N, d]
        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state           # [B, T, 768]
            text_feat = self.text_proj(text_hidden)          # [B, T, d]
            text_mask = (1 - bert_attn.float()).unsqueeze(1) # [B, 1, T]
        else:
            text_feat = torch.zeros(B, 1, self.model_channels,
                                    device=node_emb.device, dtype=node_emb.dtype)
            text_mask = None

        seq = node_emb
        for layer in self.layers:
            seq = layer(seq, adj_mask, text_feat, text_mask)

        return self.type_head(seq)            # [B, N, 33]
