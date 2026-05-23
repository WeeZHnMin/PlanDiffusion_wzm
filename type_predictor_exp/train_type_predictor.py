"""
Type Sequence Predictor
- 输入：文本 prompt → 冻结 BERT 编码
- 输出：每个顶点位置的房间类型 id（共 MAX_LEN 个位置）
- 模型：BERT [CLS] → MLP → [MAX_LEN, n_classes]
- 损失：交叉熵，只计算非 padding 位置
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

ROOM_TYPES = ["bathroom", "bedroom", "living_room", "kitchen", "corridor", "dining_room"]
N_CLASSES  = len(ROOM_TYPES) + 1   # +1 for PAD
PAD_TYPE   = N_CLASSES - 1

# ── 加载数据 ──────────────────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

MAX_LEN = max(r["n_vertices"] for r in records)
print(f"Records: {len(records)},  MAX_LEN={MAX_LEN},  N_CLASSES={N_CLASSES},  device={DEVICE}")

labels = torch.tensor([r["type_seq_padded"] for r in records],
                      dtype=torch.long).to(DEVICE)       # [N, MAX_LEN]
masks  = torch.tensor([r["type_mask"] for r in records],
                      dtype=torch.bool).to(DEVICE)        # [N, MAX_LEN]
prompts = [r["prompt"] for r in records]

# ── 冻结 BERT，预计算编码 ─────────────────────────────────────────────────────
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
        out = bert(**inputs)
        cls_encs.append(out.last_hidden_state[0, 0])   # [CLS] token: [768]
    cls_encs = torch.stack(cls_encs)                    # [N, 768]

print(f"CLS encodings: {cls_encs.shape}")

# ── 模型：CLS → MLP → [MAX_LEN, N_CLASSES] ───────────────────────────────────
class TypePredictor(nn.Module):
    def __init__(self, d_in=768, d_hidden=256, seq_len=MAX_LEN, n_classes=N_CLASSES):
        super().__init__()
        self.seq_len  = seq_len
        self.n_classes = n_classes
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, seq_len * n_classes),
        )

    def forward(self, cls_enc):
        # cls_enc: [B, 768]
        out = self.mlp(cls_enc)                          # [B, seq_len * n_classes]
        return out.view(-1, self.seq_len, self.n_classes) # [B, seq_len, n_classes]


model = TypePredictor().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# ── 训练 ──────────────────────────────────────────────────────────────────────
EPOCHS = 2000

for epoch in range(EPOCHS):
    logits = model(cls_encs)                              # [N, MAX_LEN, n_classes]
    # 只在真实顶点位置计算损失
    logits_flat = logits[masks]                           # [M, n_classes]
    labels_flat = labels[masks]                           # [M]
    loss = F.cross_entropy(logits_flat, labels_flat)

    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if (epoch + 1) % 200 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=-1)                 # [N, MAX_LEN]
            correct = (preds[masks] == labels[masks]).float().mean().item()
        print(f"Epoch {epoch+1:4d}  loss={loss.item():.4f}  acc={correct*100:.1f}%")

# ── 评估 ──────────────────────────────────────────────────────────────────────
print("\n─── Per-sample Results ───")
model.eval()
with torch.no_grad():
    logits = model(cls_encs)
    preds  = logits.argmax(dim=-1)   # [N, MAX_LEN]

for i, r in enumerate(records):
    n  = r["n_vertices"]
    gt = labels[i, :n].tolist()
    pr = preds[i, :n].tolist()
    acc = sum(g == p for g, p in zip(gt, pr)) / n
    gt_names = [ROOM_TYPES[t] if t < len(ROOM_TYPES) else "PAD" for t in gt]
    pr_names = [ROOM_TYPES[t] if t < len(ROOM_TYPES) else "PAD" for t in pr]
    print(f"  [{i+1:2d}] acc={acc*100:5.1f}%  prompt: {r['prompt'][:50]}")
    if acc < 1.0:
        print(f"       GT  : {gt_names}")
        print(f"       Pred: {pr_names}")
