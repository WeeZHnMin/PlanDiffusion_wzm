"""
Gradio Demo: Text Prompt → AdjPredictor → GCN Diffusion (step-by-step)
每20步渲染一帧节点图，展示扩散推理过程。
运行前需先用训练脚本保存权重到 weights/ 目录。
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import math
import json
import io
import base64
from pathlib import Path
from transformers import BertTokenizer, BertModel
from PIL import Image

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PATH = Path(__file__).parent.parent / "models" / "bert-base-uncased"
DATA_PATH = Path(__file__).parent / "node_data.jsonl"
WEIGHTS   = Path(__file__).parent / "weights"

T        = 400
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4

# ── 加载数据 / 归一化参数 ──────────────────────────────────────────────────────
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

N_MAX = len(records[0]["adj_matrix"])
all_x = [c[0] for r in records for c in r["node_coords"][:r["n_nodes"]]]
all_y = [c[1] for r in records for c in r["node_coords"][:r["n_nodes"]]]
_norm = {
    "xmin": min(all_x), "xrange": max(all_x) - min(all_x),
    "ymin": min(all_y), "yrange": max(all_y) - min(all_y),
}
VIZ_MIN = min(all_x) - 5
VIZ_MAX = max(all_x) + 5

# 训练集 prompt → 真实 adj/n_nodes 的查找表
# 这样对训练集 prompt 推理时直接用真实条件，保证演示效果
_gt_lookup = {r["prompt"].strip(): r for r in records}

def denorm_x(v): return np.round((v + 1) / 2 * _norm["xrange"] + _norm["xmin"]).astype(int)
def denorm_y(v): return np.round((v + 1) / 2 * _norm["yrange"] + _norm["ymin"]).astype(int)

# ── DDPM 噪声表 ───────────────────────────────────────────────────────────────
betas      = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# ── BERT（冻结）──────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
bert      = BertModel.from_pretrained(str(BERT_PATH)).to(DEVICE)
for p in bert.parameters():
    p.requires_grad = False
bert.eval()
print("BERT ready.")

# ── 模型定义（与训练脚本完全一致）────────────────────────────────────────────
class AdjPredictor(nn.Module):
    def __init__(self, d_in=768, d_hidden=512, n_max=N_MAX):
        super().__init__()
        self.n_max = n_max
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, n_max * n_max),
        )
    def forward(self, cls_enc):
        return self.mlp(cls_enc).view(-1, self.n_max, self.n_max)


def sinusoidal_emb(t, dim):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / max(half - 1, 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TwoHopGCNProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, coords, a1, a2):
        agg = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)
        return self.proj(agg)


class NodeDiffusionGCN(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, n_max=N_MAX, use_adj=True):
        super().__init__()
        self.use_adj = use_adj
        self.n_max   = n_max
        self.n_heads = n_heads
        self.gcn_proj  = TwoHopGCNProj(d_model)
        self.pos_emb   = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_proj = nn.Linear(768, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.tf  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, cls_enc, a1, a2, adj=None, node_mask=None):
        B = noisy_coords.size(0)
        coord_feat = self.gcn_proj(noisy_coords, a1, a2)
        t_emb   = self.time_proj(sinusoidal_emb(t_idx, D_MODEL)).unsqueeze(1)
        txt_emb = self.text_proj(cls_enc).unsqueeze(1)
        pos_idx = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_idx)
        x = coord_feat + pos_emb + t_emb + txt_emb
        key_pad   = ((node_mask == 0).float() * -1e9) if node_mask is not None else None
        attn_mask = None
        if self.use_adj and adj is not None:
            real_row  = node_mask.unsqueeze(-1)
            attn_mask = (1.0 - adj) * (-1e9) * real_row
            attn_mask = (attn_mask.unsqueeze(1)
                                  .expand(B, self.n_heads, N_MAX, N_MAX)
                                  .reshape(B * self.n_heads, N_MAX, N_MAX))
        x = self.tf(x, mask=attn_mask, src_key_padding_mask=key_pad)
        return self.out(x)


# ── 权重加载（懒加载，首次生成时触发）────────────────────────────────────────
_adj_predictor  = None
_diffusion_model = None
_loaded_adj_mode = None   # 记录当前已加载的 use_adj 状态


def _ensure_models(use_adj: bool):
    global _adj_predictor, _diffusion_model, _loaded_adj_mode

    # 检查权重文件
    adj_ckpt  = WEIGHTS / "adj_predictor.pt"
    diff_name = "diffusion_gcn_with_adj_mask.pt" if use_adj else "diffusion_gcn_no_adj_mask.pt"
    diff_ckpt = WEIGHTS / diff_name

    missing = [p for p in [adj_ckpt, diff_ckpt] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "权重文件尚未生成，请先运行训练脚本：\n"
            "  python type_predictor_exp/train_adj_predictor.py\n"
            "  python type_predictor_exp/train_node_diffusion_gcn.py --adj both\n\n"
            "缺少：" + "\n  ".join(str(p) for p in missing)
        )

    # AdjPredictor 只需加载一次
    if _adj_predictor is None:
        m = AdjPredictor().to(DEVICE)
        m.load_state_dict(torch.load(adj_ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        _adj_predictor = m
        print(f"AdjPredictor loaded from {adj_ckpt.name}")

    # 扩散模型按需切换
    if _diffusion_model is None or _loaded_adj_mode != use_adj:
        m = NodeDiffusionGCN(use_adj=use_adj).to(DEVICE)
        m.load_state_dict(torch.load(diff_ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        _diffusion_model = m
        _loaded_adj_mode = use_adj
        print(f"DiffusionGCN loaded: {diff_ckpt.name}")


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def row_normalize(adj, mask):
    deg    = adj.sum(dim=-1, keepdim=True).clamp(min=1)
    a_norm = adj / deg
    return a_norm * mask.unsqueeze(-1)


def render_frame(coords_np, adj_np, n_nodes, title, step_label):
    """渲染一帧节点图，返回 PIL Image。"""
    xs = denorm_x(coords_np[:n_nodes, 0])
    ys = denorm_y(coords_np[:n_nodes, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    for a in range(n_nodes):
        for b in range(a + 1, n_nodes):
            if adj_np[a, b] > 0.5:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                        color="steelblue", lw=1.5, alpha=0.75, zorder=1)
    ax.scatter(xs, ys, c=np.arange(n_nodes), cmap="tab20",
               s=60, zorder=3, edgecolors="white", linewidths=0.8)
    for i in range(n_nodes):
        ax.annotate(str(i), (xs[i], ys[i]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_title(step_label, fontsize=9, fontweight="bold")
    ax.set_xlabel(title[:55], fontsize=7)
    ax.set_xlim(VIZ_MIN, VIZ_MAX)
    ax.set_ylim(VIZ_MAX, VIZ_MIN)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def frames_to_html_gif(pil_frames, duration_ms=200):
    """把 PIL Image 列表合成动态 GIF，用 base64 <img> 嵌入 HTML。
    浏览器原生播放 GIF 动画，不依赖 JavaScript。"""
    gif_buf = io.BytesIO()
    # 转 P 模式（调色板）以生成合法 GIF
    p_frames = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
                for f in pil_frames]
    p_frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=p_frames[1:],
        duration=duration_ms,
        loop=0,          # 0 = 无限循环
        optimize=False,
    )
    gif_buf.seek(0)
    data = base64.b64encode(gif_buf.read()).decode()
    n    = len(pil_frames)
    return (
        f'<div style="text-align:center">'
        f'<img src="data:image/gif;base64,{data}" '
        f'style="max-width:560px;width:100%;border:1px solid #ccc;border-radius:8px;">'
        f'<div style="margin-top:6px;font-size:12px;color:#666;">'
        f'{n} 帧 · 每帧 {duration_ms}ms · 循环播放</div>'
        f'</div>'
    )


# ── 推理主函数 ────────────────────────────────────────────────────────────────
def generate(prompt, use_adj_toggle):
    if not prompt.strip():
        return "<p style='color:gray'>请输入提示词</p>", None, "请输入提示词"

    use_adj = (use_adj_toggle == "有邻接掩码 (with_adj_mask)")

    try:
        _ensure_models(use_adj)
    except FileNotFoundError as e:
        return f"<pre style='color:red'>{e}</pre>", None, str(e)

    adj_pred_model = _adj_predictor
    diff_model     = _diffusion_model

    with torch.no_grad():
        # 1. BERT 编码
        inp = tokenizer(prompt, return_tensors="pt",
                        padding=True, truncation=True, max_length=32)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        cls_enc = bert(**inp).last_hidden_state[0, 0].unsqueeze(0)  # [1, 768]

        # 2. 邻接矩阵和节点数
        # 训练集 prompt → 直接用 GT 条件（保证演示正确性）
        # 新 prompt  → 用 AdjPredictor 预测
        gt_rec = _gt_lookup.get(prompt.strip())
        if gt_rec is not None:
            n_nodes = gt_rec["n_nodes"]
            adj_use = torch.tensor(gt_rec["adj_matrix"],
                                   dtype=torch.float32, device=DEVICE).unsqueeze(0)
            src = "GT"
        else:
            logits  = adj_pred_model(cls_enc)
            adj_sym = ((logits.sigmoid() > 0.5).float() +
                       (logits.sigmoid() > 0.5).float().transpose(1, 2) > 0).float()
            n_nodes = int(adj_sym[0].diagonal().sum().item())
            n_nodes = max(min(n_nodes, N_MAX), 3)
            adj_use = adj_sym
            src = "AdjPredictor"

        node_mask = torch.zeros(1, N_MAX, device=DEVICE)
        node_mask[0, :n_nodes] = 1.0
        adj_use = adj_use * node_mask.unsqueeze(-1) * node_mask.unsqueeze(1)

        # 3. GCN 归一化
        a1 = row_normalize(adj_use, node_mask)
        a2 = row_normalize(torch.bmm(a1, a1), node_mask)

        adj_np  = adj_use[0].cpu().numpy()
        n_edges = int((adj_np[:n_nodes, :n_nodes].sum() - n_nodes) / 2)

        # 4. 扩散逆推理，每 20 步收集一帧，共 T/20 = 20 帧
        x      = torch.randn(1, N_MAX, 2, device=DEVICE) * node_mask.unsqueeze(-1)
        frames = []   # base64 PNG 列表

        for step in reversed(range(T)):
            t_idx = torch.tensor([step], dtype=torch.long, device=DEVICE)
            pred  = diff_model(
                x, t_idx, cls_enc,
                a1=a1, a2=a2,
                adj=adj_use if use_adj else None,
                node_mask=node_mask,
            ) * node_mask.unsqueeze(-1)

            beta  = betas[step]
            alpha = alphas[step]
            ab    = alpha_bars[step]
            mean  = (x - beta / (1.0 - ab).sqrt() * pred) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x) * node_mask.unsqueeze(-1)
            else:
                x = mean

            elapsed = T - step          # 已完成去噪步数（1~T）
            if elapsed % 20 == 0:       # 每 20 步存一帧，step=0 时 elapsed=T，T%20=0
                label = ("✅ 最终结果 (t=0)" if step == 0
                         else f"去噪 {elapsed}/{T} 步  (t={step})")
                frames.append(render_frame(
                    x[0].cpu().numpy(), adj_np, n_nodes, prompt, label))

    anim_html  = frames_to_html_gif(frames, duration_ms=200)
    final_frame = frames[-1]   # 最后一帧 PIL Image，静态展示
    status     = (f"生成完成  节点: {n_nodes}  边: {n_edges}  帧: {len(frames)}  "
                  f"来源: {src}  模式: {'有adj掩码' if use_adj else '无adj掩码'}")
    return anim_html, final_frame, status


# ── Gradio 界面 ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Floor Plan Node Diffusion", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Floor Plan Node Diffusion Demo")
    gr.Markdown(
        "输入建筑平面描述 → GCN Diffusion 400步逆扩散 → 每20步渲染一帧，共20帧，每帧0.2s循环播放"
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="提示词 (Prompt)",
                placeholder="例如：the bedroom is adjacent to the living room",
                lines=2,
            )
            adj_mode = gr.Radio(
                choices=["有邻接掩码 (with_adj_mask)", "无邻接掩码 (no_adj_mask)"],
                value="有邻接掩码 (with_adj_mask)",
                label="Transformer 注意力模式",
            )
            gen_btn    = gr.Button("生成扩散动画", variant="primary", size="lg")
            status_out = gr.Textbox(label="状态", interactive=False)
            gr.Examples(
                examples=[[r["prompt"], "有邻接掩码 (with_adj_mask)"] for r in records[:6]],
                inputs=[prompt_input, adj_mode],
                label="训练集示例（点击填充）",
            )

        with gr.Column(scale=2):
            _placeholder = "<p style='color:#aaa;text-align:center;padding:40px 0'>点击左侧按钮生成动画</p>"
            anim_out   = gr.HTML(value=_placeholder, label="扩散推理过程（动画）")
            final_out  = gr.Image(label="最终结果 (t=0)", type="pil", show_label=True)

    gen_btn.click(
        fn=generate,
        inputs=[prompt_input, adj_mode],
        outputs=[anim_out, final_out, status_out],
    )

if __name__ == "__main__":
    print(f"N_MAX={N_MAX}, device={DEVICE}")
    print(f"x: {min(all_x)}~{max(all_x)}, y: {min(all_y)}~{max(all_y)}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
