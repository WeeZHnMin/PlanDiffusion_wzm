"""
ContrastivePretrain: CLIP 式对比模型，对齐户型图嵌入与 BERT 文本嵌入。

训练完成后：
  model.text_enc.bert  即为对齐到户型布局空间的 BERT，
  可直接 save_pretrained() 后作为 train.py 的 --bert 参数使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .graph_encoder import GraphEncoder


class TextEncoder(nn.Module):
    def __init__(self, bert_name='models/bert-base-uncased',
                 embed_dim=512, unfreeze_layers=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        n = len(self.bert.encoder.layer)
        for layer in self.bert.encoder.layer[n - unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.proj = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]   # [CLS] token [B, 768]
        return self.proj(cls)               # [B, embed_dim]


class ContrastivePretrain(nn.Module):
    """
    text_emb  = encode_text(prompt_tokens, prompt_mask)     → [B, embed_dim] normalized
    graph_emb = encode_graph(coords, adj, node_mask, types) → [B, embed_dim] normalized
    InfoNCE 拉近配对，推开非配对。
    """

    def __init__(self, bert_name='models/bert-base-uncased',
                 embed_dim=512, d_model=384, num_graph_layers=4,
                 num_heads=6, unfreeze_layers=2, dropout=0.1):
        super().__init__()
        self.text_enc  = TextEncoder(bert_name, embed_dim, unfreeze_layers)
        self.graph_enc = GraphEncoder(
            d_model=d_model, num_layers=num_graph_layers,
            num_heads=num_heads, embed_dim=embed_dim, dropout=dropout,
        )
        # 可学习温度，初始值对应 CLIP 的 1/0.07 ≈ 14.3
        self.logit_scale = nn.Parameter(torch.tensor(2.659))

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.text_enc(input_ids, attention_mask), dim=-1)

    def encode_graph(self, coords, adj_matrix, node_mask, node_types):
        return F.normalize(
            self.graph_enc(coords, adj_matrix, node_mask, node_types), dim=-1
        )

    def forward(self, input_ids, attention_mask,
                coords, adj_matrix, node_mask, node_types):
        text_emb  = self.encode_text(input_ids, attention_mask)
        graph_emb = self.encode_graph(coords, adj_matrix, node_mask, node_types)
        return text_emb, graph_emb


def info_nce_loss(text_emb, graph_emb, logit_scale):
    """
    对称 InfoNCE（CLIP loss）。
    返回 (loss, top1_acc)，acc 为 text→graph 方向的对角命中率。
    """
    scale     = logit_scale.exp().clamp(max=100.0)
    logits_tg = scale * text_emb @ graph_emb.T          # [B, B]
    logits_gt = logits_tg.T
    labels    = torch.arange(len(text_emb), device=text_emb.device)
    loss      = (F.cross_entropy(logits_tg, labels)
                 + F.cross_entropy(logits_gt, labels)) / 2

    with torch.no_grad():
        acc = (logits_tg.argmax(dim=-1) == labels).float().mean().item()

    return loss, acc
