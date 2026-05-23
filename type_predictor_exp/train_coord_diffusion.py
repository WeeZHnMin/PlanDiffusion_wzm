"""
Full Pipeline: Type Predictor + Coord Diffusion + SEP token
序列格式：[SEP, room1_v0, room1_v1, ..., SEP, room2_v0, ...]
- SEP(type=0) 位置坐标固定 (0,0)，不参与扩散
- 类别：0=SEP, 1=bathroom, 2=bedroom, 3=living_room, 4=kitchen, 5=corridor, 6=dining_room, 7=PAD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import BertTokenizer, BertModel

torch.manual_seed(42)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PATH = Path(__file__).parent.parent / "models" / "bert-base-uncased"
DATA_PATH = Path(__file__).parent / "type_data.jsonl"

SEP_TYPE   = 0
ROOM_TYPES = ["SEP", "bathroom", "bedroom", "living_room",
              "kitchen", "corridor", "dining_room"]
PAD_TYPE   = 7
N_CLASSES  = 8   # 0=SEP, 1-6=rooms, 7=PAD

ROOM_COLORS = {
    "bathroom":    "#AED6F1",
    "bedroom":     "#A9DFBF",
    "living_room": "#F9E79F",
    "kitchen":     "#F1948A",
    "corridor":    "#D7BDE2",
    "dining_room": "#FAD7A0",
}


# ── 加载数据 ──────────────────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

MAX_LEN = max(r["n_tokens"] for r in records)
print(f"Records: {len(records)},  MAX_LEN={MAX_LEN},  device={DEVICE}")

# 全局归一化（只对真实坐标，SEP 的 (0,0) 不计入）
all_x = [c[0] for r in records for t, c in zip(r["type_seq"], r["coord_seq"]) if t != SEP_TYPE]
all_y = [c[1] for r in records for t, c in zip(r["type_seq"], r["coord_seq"]) if t != SEP_TYPE]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)

def norm_x(v):   return 2 * (v - xmin) / (xmax - xmin) - 1
def norm_y(v):   return 2 * (v - ymin) / (ymax - ymin) - 1
def denorm_x(v): return round((v + 1) / 2 * (xmax - xmin) + xmin)
def denorm_y(v): return round((v + 1) / 2 * (ymax - ymin) + ymin)

# 构建 tensor：SEP 位置坐标归一化后仍为 (0,0)（因为 norm(0) ≠ 0，所以手动置 0）
coords_list, type_list = [], []
for r in records:
    normed = []
    for t, (x, y) in zip(r["type_seq_padded"], r["coord_seq_padded"]):
        if t == SEP_TYPE or t == PAD_TYPE:
            normed.append([0.0, 0.0])
        else:
            normed.append([norm_x(x), norm_y(y)])
    coords_list.append(normed)
    type_list.append(r["type_seq_padded"])

coords_tensor = torch.tensor(coords_list, dtype=torch.float32).to(DEVICE)  # [N, MAX_LEN, 2]
type_tensor   = torch.tensor(type_list,   dtype=torch.long).to(DEVICE)     # [N, MAX_LEN]
# coord_mask：只对非 SEP、非 PAD 位置计算扩散损失
coord_masks   = torch.tensor([r["coord_mask"] for r in records],
                              dtype=torch.float32).to(DEVICE)               # [N, MAX_LEN]
labels_tensor = type_tensor
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
    text_encs, cls_encs = [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out    = bert(**inputs)
        text_encs.append(out.last_hidden_state[0])
        cls_encs.append(out.last_hidden_state[0, 0])
    cls_encs = torch.stack(cls_encs)


# ══════════════════════════════════════════════════════════════════════════════
# 阶段1：Type Predictor
# ══════════════════════════════════════════════════════════════════════════════
class TypePredictor(nn.Module):
    def __init__(self, d_in=768, d_hidden=256, seq_len=MAX_LEN, n_classes=N_CLASSES):
        super().__init__()
        self.seq_len   = seq_len
        self.n_classes = n_classes
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, seq_len * n_classes),
        )
    def forward(self, cls_enc):
        return self.mlp(cls_enc).view(-1, self.seq_len, self.n_classes)

type_model = TypePredictor().to(DEVICE)
type_opt   = torch.optim.AdamW(type_model.parameters(), lr=1e-3, weight_decay=1e-2)
type_sched = torch.optim.lr_scheduler.CosineAnnealingLR(type_opt, T_max=2000, eta_min=1e-5)

print(f"\nType predictor params: {sum(p.numel() for p in type_model.parameters()):,}")
print("── Stage 1: Training Type Predictor ──")

for epoch in range(2000):
    logits = type_model(cls_encs)
    loss   = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels_tensor.reshape(-1))
    type_opt.zero_grad(); loss.backward(); type_opt.step(); type_sched.step()
    if (epoch + 1) % 400 == 0:
        with torch.no_grad():
            preds    = logits.argmax(-1)
            acc_all  = (preds == labels_tensor).float().mean().item()
        print(f"  Epoch {epoch+1:4d}  loss={loss.item():.4f}  acc={acc_all*100:.1f}%")

type_model.eval()
print("Type predictor training done.")

# ══════════════════════════════════════════════════════════════════════════════
# 阶段2：Coord Diffusion
# ══════════════════════════════════════════════════════════════════════════════
T          = 200
betas      = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, coord_mask):
    """只对 coord_mask=1 的位置加噪声，SEP/PAD 位置保持 0"""
    ab  = alpha_bars[t].view(-1, 1, 1)
    eps = torch.randn_like(x0)
    xt  = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    m   = coord_mask.unsqueeze(-1)   # [1, MAX_LEN, 1]
    return xt * m, eps * m           # SEP/PAD 位置置 0


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class CoordDiffusionModel(nn.Module):
    def __init__(self, coord_len=MAX_LEN, d_model=64, nhead=4, n_layers=4,
                 t_dim=32, n_classes=N_CLASSES):
        super().__init__()
        self.coord_len  = coord_len
        self.text_proj  = nn.Linear(768, d_model)
        self.coord_proj = nn.Linear(2, d_model)
        self.type_emb   = nn.Embedding(n_classes, d_model)
        self.pos_emb    = nn.Embedding(coord_len, d_model)
        self.time_emb   = SinusoidalEmbedding(t_dim)
        self.time_proj  = nn.Linear(t_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, text_enc, noisy_coords, type_ids, t):
        txt       = self.text_proj(text_enc).unsqueeze(0)
        coord     = self.coord_proj(noisy_coords)
        tpe       = self.type_emb(type_ids)
        pos       = self.pos_emb(
            torch.arange(self.coord_len, device=noisy_coords.device).unsqueeze(0)
        )
        t_emb     = self.time_proj(self.time_emb(t)).unsqueeze(1)
        coord_tok = coord + tpe + pos + t_emb
        seq       = torch.cat([txt, coord_tok], dim=1)
        h         = self.transformer(seq)
        return self.out(h[:, txt.shape[1]:, :])   # [1, MAX_LEN, 2]


coord_model = CoordDiffusionModel().to(DEVICE)
coord_opt   = torch.optim.AdamW(coord_model.parameters(), lr=1e-3, weight_decay=1e-2)
coord_sched = torch.optim.lr_scheduler.CosineAnnealingLR(coord_opt, T_max=4000, eta_min=1e-5)

print(f"\nCoord diffusion params: {sum(p.numel() for p in coord_model.parameters()):,}")
print("── Stage 2: Training Coord Diffusion ──")

for epoch in range(4000):
    epoch_loss = 0.0
    for i in range(len(records)):
        t_rand       = torch.randint(0, T, (1,), device=DEVICE)
        x0           = coords_tensor[i].unsqueeze(0)
        cmask        = coord_masks[i].unsqueeze(0)
        tids         = type_tensor[i].unsqueeze(0)
        xt, eps      = q_sample(x0, t_rand, cmask)
        pred         = coord_model(text_encs[i], xt, tids, t_rand)
        loss         = ((pred - eps) ** 2 * cmask.unsqueeze(-1)).sum() / (cmask.sum() * 2)
        coord_opt.zero_grad(); loss.backward(); coord_opt.step()
        epoch_loss  += loss.item()
    coord_sched.step()
    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:5d}  loss={epoch_loss/len(records):.4f}")

coord_model.eval()
print("Coord diffusion training done.")

# ── 推理 ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_type_ids(cls_enc):
    return type_model(cls_enc.unsqueeze(0)).argmax(-1)   # [1, MAX_LEN]

@torch.no_grad()
def sample_coords(text_enc, type_ids, coord_mask):
    x = torch.randn(1, MAX_LEN, 2, device=DEVICE) * coord_mask.unsqueeze(-1)
    t_batch = torch.zeros(1, dtype=torch.long, device=DEVICE)
    for step in reversed(range(T)):
        t_batch[0] = step
        pred_eps   = coord_model(text_enc, x, type_ids, t_batch)
        ab   = alpha_bars[step]
        beta = betas[step]
        ab_p = alpha_bars[step - 1] if step > 0 else torch.tensor(1.0, device=DEVICE)
        pred_eps = pred_eps * coord_mask.unsqueeze(-1)   # SEP 位置噪声归零
        mean = (1.0 / alphas[step].sqrt()) * (x - beta / (1 - ab).sqrt() * pred_eps)
        mean = mean * coord_mask.unsqueeze(-1)            # SEP 位置强制归零
        if step > 0:
            sigma = ((1 - ab_p) / (1 - ab) * beta).sqrt()
            x = mean + sigma * torch.randn_like(x) * coord_mask.unsqueeze(-1)
        else:
            x = mean
    return x[0]   # [MAX_LEN, 2]

# ── 可视化 ────────────────────────────────────────────────────────────────────
def extract_rooms_from_seq(type_seq, coords_xy):
    """按 SEP(=0) 切分房间，每段第一个 token 是 SEP，后面是该房间顶点"""
    rooms, i = [], 0
    while i < len(type_seq):
        t = int(type_seq[i])
        if t == PAD_TYPE:
            break
        if t == SEP_TYPE:
            # 收集直到下一个 SEP 或 PAD
            j = i + 1
            while j < len(type_seq) and int(type_seq[j]) not in (SEP_TYPE, PAD_TYPE):
                j += 1
            if j > i + 1:
                room_type_id = int(type_seq[i + 1])
                room_name    = ROOM_TYPES[room_type_id] if 1 <= room_type_id <= 6 else "unknown"
                rooms.append({
                    "type":   room_name,
                    "coords": coords_xy[i+1:j]
                })
            i = j
        else:
            i += 1
    return rooms

def draw_floorplan(ax, rooms, title):
    ax.set_aspect("equal")
    all_x, all_y = [], []
    for room in rooms:
        if len(room["coords"]) < 2:
            continue
        xs = [c[0] for c in room["coords"]]
        ys = [-c[1] for c in room["coords"]]
        all_x.extend(xs); all_y.extend(ys)
        color = ROOM_COLORS.get(room["type"], "#D5D8DC")
        poly = plt.Polygon(list(zip(xs, ys)), closed=True,
                           facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85)
        ax.add_patch(poly)
        ax.text(sum(xs)/len(xs), sum(ys)/len(ys),
                room["type"].replace("_", "\n"),
                ha="center", va="center", fontsize=5.5, fontweight="bold")
    if all_x:
        m = 10
        ax.set_xlim(min(all_x)-m, max(all_x)+m)
        ax.set_ylim(min(all_y)-m, max(all_y)+m)
    ax.set_title(title, fontsize=6, wrap=True, pad=3)
    ax.tick_params(labelsize=5)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)


print("\n─── Inference & Visualization ───")
n = len(records)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 9, nrows * 4.5))

all_mae = []
for i, r in enumerate(records):
    pred_tids  = predict_type_ids(cls_encs[i])              # [1, MAX_LEN]
    cmask      = coord_masks[i].unsqueeze(0)
    pred_xy    = sample_coords(text_encs[i], pred_tids, cmask)

    # 真实坐标（只取非 SEP 位置）
    gt_type_seq = r["type_seq"]
    gt_coord_seq = r["coord_seq"]
    gt_real_coords = [(c[0], c[1]) for t, c in zip(gt_type_seq, gt_coord_seq) if t != SEP_TYPE]

    # 预测坐标（只取非 SEP 位置）
    n_tok = r["n_tokens"]
    pred_type_seq = pred_tids[0, :n_tok].tolist()
    pred_real_coords = []
    for j, t in enumerate(pred_type_seq):
        if t != SEP_TYPE:
            pred_real_coords.append((denorm_x(pred_xy[j, 0].item()),
                                     denorm_y(pred_xy[j, 1].item())))

    # MAE（对齐长度）
    L = min(len(gt_real_coords), len(pred_real_coords))
    if L > 0:
        mae = sum(abs(p[0]-g[0]) + abs(p[1]-g[1])
                  for p, g in zip(pred_real_coords[:L], gt_real_coords[:L])) / (L * 2)
    else:
        mae = 999.0
    all_mae.append(mae)

    type_acc = sum(p == g for p, g in zip(pred_type_seq, gt_type_seq)) / len(gt_type_seq)
    print(f"  [{i+1:2d}] MAE={mae:5.1f}  type_acc={type_acc*100:.0f}%  {r['prompt'][:45]}")

    row = i // ncols
    col = (i % ncols) * 2

    # 用完整序列（含 SEP 位置信息）来提取房间
    gt_full_coords   = [(c[0], c[1]) for c in r["coord_seq"]]
    pred_full_coords = [(denorm_x(pred_xy[j, 0].item()), denorm_y(pred_xy[j, 1].item()))
                        for j in range(n_tok)]

    gt_rooms   = extract_rooms_from_seq(gt_type_seq,   gt_full_coords)
    pred_rooms = extract_rooms_from_seq(pred_type_seq, pred_full_coords)

    draw_floorplan(axes[row, col],   gt_rooms,   f"[{i+1}] GT\n{r['prompt'][:40]}")
    draw_floorplan(axes[row, col+1], pred_rooms, f"[{i+1}] Pred  MAE={mae:.1f}")

for idx in range(n, nrows * ncols):
    row = idx // ncols; col = (idx % ncols) * 2
    axes[row, col].set_visible(False)
    axes[row, col+1].set_visible(False)

handles = [mpatches.Patch(facecolor=c, edgecolor="black", label=r)
           for r, c in ROOM_COLORS.items()]
fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=3,
           title="Room type", title_fontsize=9)
plt.suptitle(f"GT vs Predicted Floor Plans  (mean MAE={np.mean(all_mae):.1f})",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

out = Path(__file__).parent / "coord_diffusion_result.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {out}")
print(f"Mean MAE: {np.mean(all_mae):.1f}")
