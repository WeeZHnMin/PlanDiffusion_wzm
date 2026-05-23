"""
Adjacency Matrix Predictor
- 输入：文本 prompt → 冻结 BERT [CLS]
- 输出：N_MAX × N_MAX 的 0/1 邻接矩阵
- 损失：binary cross-entropy，只计算有效节点区域（非 padding）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel

torch.manual_seed(42)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PATH = Path(__file__).parent.parent / "models" / "bert-base-uncased"
DATA_PATH = Path(__file__).parent / "type_data.jsonl"

# ── 加载数据 ──────────────────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

N_MAX = max(len(r["vertices"]) for r in records)
print(f"Records: {len(records)},  N_MAX={N_MAX},  device={DEVICE}")

# 构建邻接矩阵 [N, N_MAX, N_MAX]
adj_matrices = []
node_counts  = []
for r in records:
    n   = len(r["vertices"])
    adj = torch.zeros(N_MAX, N_MAX)
    for i, neighbors in enumerate(r["vertex_adj"]):
        for j in neighbors:
            adj[i, j] = 1.0
    # 对角线：自己和自己相连
    for i in range(n):
        adj[i, i] = 1.0
    adj_matrices.append(adj)
    node_counts.append(n)

adj_tensor   = torch.stack(adj_matrices).to(DEVICE)   # [N, N_MAX, N_MAX]
node_counts  = torch.tensor(node_counts, dtype=torch.long)

# valid_mask[i, a, b] = 1 表示样本 i 的 (a,b) 位置是有效节点区域
valid_masks = torch.zeros(len(records), N_MAX, N_MAX)
for i, n in enumerate(node_counts):
    valid_masks[i, :n, :n] = 1.0
valid_masks = valid_masks.to(DEVICE)

prompts = [r["prompt"] for r in records]

# ── BERT（冻结）──────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
bert      = BertModel.from_pretrained(str(BERT_PATH)).to(DEVICE)
for p in bert.parameters():
    p.requires_grad = False
bert.eval()

print("Encoding prompts...")
with torch.no_grad():
    cls_encs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out    = bert(**inputs)
        cls_encs.append(out.last_hidden_state[0, 0])
    cls_encs = torch.stack(cls_encs)   # [N, 768]

print(f"CLS encodings: {cls_encs.shape}")

# ── 模型 ──────────────────────────────────────────────────────────────────────
class AdjPredictor(nn.Module):
    def __init__(self, d_in=768, d_hidden=512, n_max=N_MAX):
        super().__init__()
        self.n_max = n_max
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_max * n_max),
        )

    def forward(self, cls_enc):
        out = self.mlp(cls_enc)                        # [B, N_MAX*N_MAX]
        return out.view(-1, self.n_max, self.n_max)    # [B, N_MAX, N_MAX]


model = AdjPredictor().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-5)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# ── 训练 ──────────────────────────────────────────────────────────────────────
EPOCHS = 3000

pad_masks = (1.0 - valid_masks).bool()   # padding 区域，目标全为 0

for epoch in range(EPOCHS):
    logits = model(cls_encs)                           # [N, N_MAX, N_MAX]

    # 有效节点区域：BCE(pred, GT)
    logits_valid = logits[valid_masks.bool()]          # [M_valid]
    labels_valid = adj_tensor[valid_masks.bool()]      # [M_valid]
    loss_valid = F.binary_cross_entropy_with_logits(logits_valid, labels_valid)

    # Padding 区域：强制预测为 0，避免 demo 中 n_nodes 判断偏大
    logits_pad = logits[pad_masks]                     # [M_pad]
    loss_pad   = F.binary_cross_entropy_with_logits(
        logits_pad, torch.zeros_like(logits_pad))

    loss = loss_valid + loss_pad

    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            preds  = (logits.sigmoid() > 0.5).float()
            correct = (preds[valid_masks.bool()] == labels_valid).float().mean().item()
            edge_mask = (adj_tensor * valid_masks).bool()
            recall = (preds[edge_mask] == 1.0).float().mean().item() if edge_mask.sum() > 0 else 0.0
            # padding 区域不应有任何预测为 1
            pad_fp = preds[pad_masks].mean().item()   # 越接近 0 越好
        print(f"Epoch {epoch+1:4d}  loss={loss.item():.4f}  "
              f"acc={correct*100:.1f}%  recall={recall*100:.1f}%  "
              f"pad_fp={pad_fp*100:.1f}%")

# ── 评估 ──────────────────────────────────────────────────────────────────────
print("\n─── Per-sample Results ───")
model.eval()
with torch.no_grad():
    logits = model(cls_encs)
    preds  = (logits.sigmoid() > 0.5).float()

for i, r in enumerate(records):
    n     = len(r["vertices"])
    gt    = adj_tensor[i, :n, :n]
    pred  = preds[i, :n, :n]
    acc   = (pred == gt).float().mean().item()
    edge_recall = (pred[gt == 1] == 1).float().mean().item() if gt.sum() > 0 else 0.0
    # 验证 n_nodes 检测：对角线预测的节点数应 == n
    pred_n = int(preds[i].diagonal().sum().item())
    n_ok   = "✓" if pred_n == n else f"✗(pred={pred_n})"
    print(f"  [{i+1:2d}] n={n:2d}{n_ok}  acc={acc*100:5.1f}%  recall={edge_recall*100:5.1f}%  {r['prompt'][:40]}")

# ── 保存权重 ──────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), WEIGHTS_DIR / "adj_predictor.pt")
print(f"\n权重已保存 → {WEIGHTS_DIR / 'adj_predictor.pt'}")
