import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

MAX_NODES = 40


# ── Graph Tower ──────────────────────────────────────────────────────────────

class GraphSAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim,  out_dim)
        self.lin_neigh = nn.Linear(in_dim,  out_dim)
        self.norm      = nn.LayerNorm(out_dim)

    def forward(self, x, adj_norm):
        # adj_norm: (B, N, N) row-normalised
        agg = torch.bmm(adj_norm, x)
        out = self.lin_self(x) + self.lin_neigh(agg)
        return self.norm(F.gelu(out))


class GraphTower(nn.Module):
    def __init__(self, in_dim=2, hidden=256, out_dim=384, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GraphSAGEConv(hidden, hidden) for _ in range(n_layers)]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _row_norm(adj, mask):
        # zero-out padding rows/cols, then row-normalise
        mask2d = mask.unsqueeze(2) * mask.unsqueeze(1)      # (B, N, N)
        adj    = adj * mask2d
        deg    = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        return adj / deg

    def forward(self, coords, adj, mask):
        # coords: (B, N, 2), adj: (B, N, N), mask: (B, N)
        adj_n = self._row_norm(adj, mask)
        x     = F.gelu(self.input_proj(coords))
        for conv in self.convs:
            x = conv(x, adj_n)
        # masked mean-pool
        m      = mask.unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.out_proj(pooled)                         # (B, out_dim)


# ── Text Tower ────────────────────────────────────────────────────────────────

class TextTower(nn.Module):
    def __init__(self, bert_name='models/bert-base-uncased', out_dim=384,
                 unfreeze_last_n=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        # freeze all
        for p in self.bert.parameters():
            p.requires_grad_(False)
        # unfreeze last n encoder layers
        n_layers = len(self.bert.encoder.layer)
        for i in range(n_layers - unfreeze_last_n, n_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad_(True)
        # unfreeze pooler
        for p in self.bert.pooler.parameters():
            p.requires_grad_(True)

        bert_dim = self.bert.config.hidden_size          # 768
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output                        # (B, 768)
        return self.proj(pooled)                          # (B, out_dim)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce(text_emb, graph_emb, tau=0.07):
    """Symmetric InfoNCE over a batch."""
    t = F.normalize(text_emb,  dim=-1)
    g = F.normalize(graph_emb, dim=-1)
    logits = torch.matmul(t, g.T) / tau          # (B, B)
    labels = torch.arange(t.shape[0], device=t.device)
    loss_t = F.cross_entropy(logits,   labels)   # text  → graph
    loss_g = F.cross_entropy(logits.T, labels)   # graph → text
    return (loss_t + loss_g) / 2.0


# ── Full alignment model ──────────────────────────────────────────────────────

class AlignModel(nn.Module):
    def __init__(self, bert_name='models/bert-base-uncased', out_dim=384,
                 unfreeze_last_n=2, tau=0.07):
        super().__init__()
        self.text_tower  = TextTower(bert_name, out_dim, unfreeze_last_n)
        self.graph_tower = GraphTower(in_dim=2, hidden=256, out_dim=out_dim)
        self.tau         = tau

    def forward(self, input_ids, attention_mask, coords, adj, mask):
        t = self.text_tower(input_ids, attention_mask)
        g = self.graph_tower(coords, adj, mask)
        loss = info_nce(t, g, self.tau)
        return loss, t, g
