import torch
import torch.nn as nn
from transformers import BertModel

from .model import EncoderLayer

N_TYPES = 33   # 0=padding, 1-32=valid types


class TextCondGNN(nn.Module):
    """
    Node type classifier mirroring NodeDiffusionTransformer's dual-stream attention:
      each layer = adj self-attn → cross-attn (Q=node, K/V=BERT tokens) → FFN

    Input : x             [B, 2, N]   clean coordinates (T=0)
            adj_matrix    [B, N, N]
            node_mask     [B, N]
            prompt_tokens [B, T]      BERT input_ids
            prompt_mask   [B, T]      BERT attention_mask (1=valid, 0=pad)
    Output: logits        [B, N, 33]
    """

    def __init__(self, d_model=384, num_layers=4, num_heads=6,
                 dropout=0.1, bert_name='models/bert-base-uncased',
                 unfreeze_layers=0, n_types=N_TYPES,
                 # accept legacy alias
                 model_channels=None):
        super().__init__()
        if model_channels is not None:
            d_model = model_channels
        self.d_model = d_model

        self.coord_emb = nn.Linear(2, d_model)

        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_layers > 0:
            n_bert = len(self.bert.encoder.layer)
            for layer in self.bert.encoder.layer[n_bert - unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.text_proj = nn.Linear(self.bert.config.hidden_size, d_model)

        # same EncoderLayer as NodeDiffusionTransformer
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

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
        adj_mask = 1 - adj_matrix                        # [B, N, N]
        pad_keys = (1 - node_mask).unsqueeze(1)          # [B, 1, N]
        return torch.clamp(adj_mask + pad_keys, 0, 1)    # [B, N, N]

    def forward(self, x, adj_matrix, node_mask,
                prompt_tokens=None, prompt_mask=None, **kwargs):
        del kwargs
        B = x.shape[0]
        x = x.permute(0, 2, 1).float()          # [B, N, 2]
        node_feat = self.coord_emb(x)            # [B, N, d]

        adj_mask = self._build_adj_mask(adj_matrix.float(), node_mask.float())

        if prompt_tokens is not None:
            bert_attn = prompt_mask if prompt_mask is not None \
                        else (prompt_tokens != 0).long()
            with torch.no_grad():
                text_hidden = self.bert(
                    input_ids=prompt_tokens,
                    attention_mask=bert_attn,
                ).last_hidden_state                      # [B, T, 768]
            text_feat = self.text_proj(text_hidden)      # [B, T, d]
            text_mask = (1 - bert_attn.float()).unsqueeze(1)  # [B, 1, T]
        else:
            text_feat = torch.zeros(B, 1, self.d_model,
                                    device=node_feat.device, dtype=node_feat.dtype)
            text_mask = None

        for layer in self.layers:
            node_feat = layer(node_feat, adj_mask, text_feat, text_mask)

        return self.type_head(node_feat)         # [B, N, 33]


# keep old name as alias so existing import references don't break
NodeTypeClassifier = TextCondGNN
