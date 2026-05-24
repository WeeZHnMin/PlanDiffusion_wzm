"""
Large-scale training for node diffusion cross-attention model.

- Keeps architecture style of train_node_diffusion_cross_attn.py
- Supports ~50k jsonl records with batched training
- Features: DataLoader workers, AMP, token-id cache, train/val split, best checkpoint
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--save", type=Path, default=Path("type_predictor_exp/weights/diffusion_gcn_cross_attn_large.pt"))
    parser.add_argument("--token-cache", type=Path, default=Path("data/jsonl/train_nodes.diffusion_tokens.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bert-lr", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--adj", choices=["yes", "no"], default="no")
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--init-from-uncond", type=Path, default=None,
                        help="unconditional checkpoint to warm-start shared weights")
    parser.add_argument("--pred-type", choices=["epsilon", "x0"], default="x0")
    parser.add_argument("--snr-gamma", type=float, default=5.0,
                        help="Min-SNR-gamma cap for x0 loss weighting (ignored for epsilon)")
    return parser.parse_args()


def load_jsonl_records(path: Path):
    records = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] skip bad json line={ln}")
    if not records:
        raise SystemExit(f"No valid records in {path}")
    if bad:
        print(f"[WARN] skipped bad lines: {bad}")
    return records


def build_or_load_token_cache(tokenizer, prompts, max_length: int, cache_path: Path):
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        input_ids = cache["input_ids"]
        attention_mask = cache["attention_mask"]
        if input_ids.size(0) == len(prompts) and input_ids.size(1) == max_length:
            print(f"token cache hit: {cache_path}")
            return input_ids, attention_mask
        print("token cache shape mismatch, rebuilding...")

    print("tokenizing all prompts once...")
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].contiguous()
    attention_mask = enc["attention_mask"].contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, cache_path)
    print(f"token cache saved: {cache_path}")
    return input_ids, attention_mask


def row_normalize(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    a_norm = adj / deg
    return a_norm * mask.unsqueeze(-1)


def transfer_uncond_weights(cond_model: "NodeDiffusionCrossAttn", ckpt_path) -> int:
    """
    Load shared weights from an unconditional checkpoint into the conditional model.

    Mapped layers (must have same d_model):
      uncond input_proj[0,2]  -> cond gcn_proj.proj[0,2]  (first 2 input cols copied)
      uncond pos_emb          -> cond pos_emb
      uncond time_proj        -> cond time_proj
      uncond layers[i].norm1  -> cond layers[i].norm1
      uncond layers[i].norm2  -> cond layers[i].norm2
      uncond layers[i].attn   -> cond layers[i].self_attn
      uncond layers[i].ffn    -> cond layers[i].ffn
      uncond final_norm       -> cond final_norm
      uncond out              -> cond out

    cross_attn / norm3 / text_kv_proj stay randomly initialized.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    copied = 0

    def _copy(dst: nn.Parameter, src_key: str):
        nonlocal copied
        if src_key in sd and sd[src_key].shape == dst.data.shape:
            dst.data.copy_(sd[src_key])
            copied += 1

    # pos_emb, time_proj, final_norm, out — shapes identical
    for key in ("pos_emb.weight",
                "time_proj.0.weight", "time_proj.0.bias",
                "time_proj.2.weight", "time_proj.2.bias",
                "final_norm.weight", "final_norm.bias",
                "out.weight", "out.bias"):
        _copy(dict(cond_model.named_parameters())[key], key)

    # gcn_proj.proj[2] == input_proj[2]: Linear(d_model, d_model) — same shape
    _copy(cond_model.gcn_proj.proj[2].weight, "input_proj.2.weight")
    _copy(cond_model.gcn_proj.proj[2].bias,   "input_proj.2.bias")

    # gcn_proj.proj[0]: Linear(6, d_model) — copy first 2 input columns from input_proj[0]: Linear(2, d_model)
    src_w = sd.get("input_proj.0.weight")   # (d_model, 2)
    src_b = sd.get("input_proj.0.bias")     # (d_model,)
    if src_w is not None and src_w.shape == (cond_model.d_model, 2):
        with torch.no_grad():
            cond_model.gcn_proj.proj[0].weight[:, :2].copy_(src_w)
            cond_model.gcn_proj.proj[0].weight[:, 2:].zero_()
        if src_b is not None:
            cond_model.gcn_proj.proj[0].bias.data.copy_(src_b)
        copied += 2

    # per-layer: norm1, norm2, self_attn (=attn), ffn
    n_shared = min(len(cond_model.layers), len([k for k in sd if k.startswith("layers.")
                                                 and k.split(".")[1].isdigit()
                                                 and int(k.split(".")[1]) < len(cond_model.layers)]))
    uncond_layer_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    for ui, ci in zip(uncond_layer_ids[:len(cond_model.layers)], range(len(cond_model.layers))):
        prefix_u = f"layers.{ui}"
        prefix_c = f"layers.{ci}"
        mapping = {
            f"{prefix_c}.norm1.weight": f"{prefix_u}.norm1.weight",
            f"{prefix_c}.norm1.bias":   f"{prefix_u}.norm1.bias",
            f"{prefix_c}.norm2.weight": f"{prefix_u}.norm2.weight",
            f"{prefix_c}.norm2.bias":   f"{prefix_u}.norm2.bias",
            f"{prefix_c}.ffn.0.weight": f"{prefix_u}.ffn.0.weight",
            f"{prefix_c}.ffn.0.bias":   f"{prefix_u}.ffn.0.bias",
            f"{prefix_c}.ffn.2.weight": f"{prefix_u}.ffn.2.weight",
            f"{prefix_c}.ffn.2.bias":   f"{prefix_u}.ffn.2.bias",
        }
        # self_attn in cond = attn in uncond
        for suffix in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"):
            mapping[f"{prefix_c}.self_attn.{suffix}"] = f"{prefix_u}.attn.{suffix}"
        cond_params = dict(cond_model.named_parameters())
        for dst_key, src_key in mapping.items():
            if dst_key in cond_params:
                _copy(cond_params[dst_key], src_key)

    return copied


def cosine_alpha_bars(timesteps: int, s: float = 0.008):
    ts = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((ts / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def snr_weighted_loss(pred_x0, x0, t_idx, alpha_bars, snr_gamma, mask3):
    """Min-SNR-γ weighted x0-prediction MSE (Choi et al. 2022)."""
    ab = alpha_bars[t_idx]                        # (B,)
    snr = ab / (1.0 - ab)                         # (B,)  signal-to-noise ratio
    w = snr.clamp(max=snr_gamma)                  # (B,)  cap at γ
    per_elem = (pred_x0 - x0) ** 2 * mask3        # (B, N, 2)
    per_sample = per_elem.sum(dim=(1, 2)) / mask3.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
    return (w * per_sample).mean()


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor):
    ab = alpha_bars[t_idx].view(-1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TwoHopGCNProj(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, coords: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor):
        agg = torch.cat([coords, torch.bmm(a1, coords), torch.bmm(a2, coords)], dim=-1)
        return self.proj(agg)


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, text_kv, self_attn_mask=None, node_key_pad=None, text_key_pad=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=self_attn_mask, key_padding_mask=node_key_pad)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, text_kv, text_kv, key_padding_mask=text_key_pad)[0]
        x = x + self.ffn(self.norm3(x))
        return x


class NodeDiffusionCrossAttn(nn.Module):
    def __init__(self, n_max: int, d_model: int, n_heads: int, n_layers: int, use_adj: bool,
                 text_hidden: int = 768):
        super().__init__()
        self.use_adj = use_adj
        self.n_max = n_max
        self.n_heads = n_heads
        self.d_model = d_model
        self.gcn_proj = TwoHopGCNProj(d_model)
        self.pos_emb = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_kv_proj = nn.Linear(text_hidden, d_model)
        self.layers = nn.ModuleList([CrossAttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, noisy_coords, t_idx, text_enc, text_pad_mask, a1, a2, adj, node_mask):
        bsz = noisy_coords.size(0)
        pos_idx = torch.arange(self.n_max, device=noisy_coords.device).unsqueeze(0)
        time_emb = self.time_proj(sinusoidal_emb(t_idx, self.d_model)).unsqueeze(1)

        x = self.gcn_proj(noisy_coords, a1, a2) + self.pos_emb(pos_idx) + time_emb
        text_kv = self.text_kv_proj(text_enc)
        text_key_pad = (text_pad_mask == 0)
        node_key_pad = (node_mask == 0)

        self_attn_mask = None
        if self.use_adj:
            real_row = node_mask.unsqueeze(-1)
            self_attn_mask = (
                (1.0 - adj) * (-1e9) * real_row
            ).unsqueeze(1).expand(bsz, self.n_heads, self.n_max, self.n_max).reshape(
                bsz * self.n_heads, self.n_max, self.n_max
            )

        for layer in self.layers:
            x = layer(
                x,
                text_kv,
                self_attn_mask=self_attn_mask,
                node_key_pad=node_key_pad,
                text_key_pad=text_key_pad,
            )
        return self.out(self.final_norm(x))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    use_adj = args.adj == "yes"
    torch.manual_seed(42)

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.bert.exists():
        raise SystemExit(f"Missing bert path: {args.bert}")

    print("loading jsonl records...")
    records = load_jsonl_records(args.data)
    n_total = len(records)
    n_max = len(records[0]["adj_matrix"])

    val_size = max(1, int(n_total * args.val_ratio))
    if val_size >= n_total:
        val_size = max(1, n_total - 1)
    train_n = n_total - val_size
    print(f"records={n_total}, train={train_n}, val={val_size}, n_max={n_max}, device={device}, amp={use_amp}, adj={use_adj}")

    print("building tensors...")
    coords_raw = torch.zeros((n_total, n_max, 2), dtype=torch.float32)
    node_masks = torch.zeros((n_total, n_max), dtype=torch.float32)
    adj_tensor = torch.zeros((n_total, n_max, n_max), dtype=torch.float32)

    prompts = []
    for i, r in enumerate(records):
        n_nodes = int(r["n_nodes"])
        prompts.append(r["prompt"])
        node_masks[i, :n_nodes] = 1.0
        adj_tensor[i] = torch.tensor(r["adj_matrix"], dtype=torch.float32)
        raw = r["node_coords"][:n_nodes]
        xs = [c[0] for c in raw]
        ys = [c[1] for c in raw]
        xmin_r = min(xs); xrange_r = max(max(xs) - min(xs), 1)
        ymin_r = min(ys); yrange_r = max(max(ys) - min(ys), 1)
        for k, (x, y) in enumerate(raw):
            coords_raw[i, k, 0] = 2.0 * (x - xmin_r) / xrange_r - 1.0
            coords_raw[i, k, 1] = 2.0 * (y - ymin_r) / yrange_r - 1.0
        if (i + 1) % 5000 == 0:
            print(f"prepared {i + 1}/{n_total}")

    print("loading bert/tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert = BertModel.from_pretrained(str(args.bert)).to(device)
    for name, p in bert.named_parameters():
        layer_id = None
        for part in name.split("."):
            if part.isdigit():
                layer_id = int(part)
                break
        p.requires_grad = layer_id is not None and layer_id >= 10
    bert_trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print(f"bert trainable params (last 2 layers): {bert_trainable:,}")

    input_ids, attention_mask = build_or_load_token_cache(
        tokenizer, prompts, args.max_length, args.token_cache
    )

    dataset = TensorDataset(input_ids, attention_mask, coords_raw, node_masks, adj_tensor)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        lengths=[train_n, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    text_hidden = bert.config.hidden_size
    model = NodeDiffusionCrossAttn(
        n_max=n_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_adj=use_adj,
        text_hidden=text_hidden,
    ).to(device)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")

    if args.init_from_uncond is not None:
        n_copied = transfer_uncond_weights(model, args.init_from_uncond)
        print(f"transferred {n_copied} tensors from {args.init_from_uncond}")
    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": bert_params, "lr": args.bert_lr},
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        bert.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch in train_loader:
            b_input_ids, b_attn_mask, b_coords, b_mask, b_adj = batch
            b_input_ids = b_input_ids.to(device, non_blocking=True)
            b_attn_mask = b_attn_mask.to(device, non_blocking=True)
            b_coords = b_coords.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_adj = b_adj.to(device, non_blocking=True)

            if use_adj:
                a1 = row_normalize(b_adj, b_mask)
                a2 = row_normalize(torch.bmm(a1, a1), b_mask)
                adj_for_attn = b_adj
            else:
                a1 = torch.zeros_like(b_adj)
                a2 = torch.zeros_like(b_adj)
                adj_for_attn = b_adj

            t_idx = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
            noise = torch.randn_like(b_coords)
            mask3 = b_mask.unsqueeze(-1)
            x_t = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3

            with torch.amp.autocast("cuda", enabled=use_amp):
                text_enc = bert(input_ids=b_input_ids, attention_mask=b_attn_mask).last_hidden_state
                text_mask = b_attn_mask.float()
                pred = model(
                    x_t, t_idx, text_enc, text_mask, a1=a1, a2=a2, adj=adj_for_attn, node_mask=b_mask
                )
                if args.pred_type == "x0":
                    loss = ((pred - b_coords) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)
                else:
                    loss = ((pred - noise) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(b_coords.size(0))
            train_loss_sum += loss.item() * bs
            train_items += bs
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch={epoch+1:4d} step={global_step:7d} train_loss={train_loss_sum / max(train_items,1):.6f}")

        sched.step()

        model.eval()
        bert.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids, b_attn_mask, b_coords, b_mask, b_adj = batch
                b_input_ids = b_input_ids.to(device, non_blocking=True)
                b_attn_mask = b_attn_mask.to(device, non_blocking=True)
                b_coords = b_coords.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                b_adj = b_adj.to(device, non_blocking=True)

                if use_adj:
                    a1 = row_normalize(b_adj, b_mask)
                    a2 = row_normalize(torch.bmm(a1, a1), b_mask)
                    adj_for_attn = b_adj
                else:
                    a1 = torch.zeros_like(b_adj)
                    a2 = torch.zeros_like(b_adj)
                    adj_for_attn = b_adj

                t_idx = torch.randint(0, args.timesteps, (b_coords.size(0),), device=device)
                noise = torch.randn_like(b_coords)
                mask3 = b_mask.unsqueeze(-1)
                x_t = q_sample(b_coords, t_idx, noise, alpha_bars) * mask3

                with torch.amp.autocast("cuda", enabled=use_amp):
                    text_enc = bert(input_ids=b_input_ids, attention_mask=b_attn_mask).last_hidden_state
                    text_mask = b_attn_mask.float()
                    pred = model(
                        x_t, t_idx, text_enc, text_mask, a1=a1, a2=a2, adj=adj_for_attn, node_mask=b_mask
                    )
                    if args.pred_type == "x0":
                        # unweighted MSE for val: cleaner metric, comparable to baseline
                        loss = ((pred - b_coords) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)
                    else:
                        loss = ((pred - noise) ** 2 * mask3).sum() / mask3.sum().clamp(min=1.0)

                bs = int(b_coords.size(0))
                val_loss_sum += loss.item() * bs
                val_items += bs

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss = val_loss_sum / max(val_items, 1)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_max": n_max,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "timesteps": args.timesteps,
                    "use_adj": use_adj,
                    "pred_type": args.pred_type,
                    "snr_gamma": args.snr_gamma if args.pred_type == "x0" else None,
                    "norm_type": "per_record",
                    "bert_path": str(args.bert),
                    "data_path": str(args.data),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                },
                args.save,
            )
        print(
            f"epoch={epoch+1:4d} done train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_val_loss={best_val:.6f} {'(saved)' if improved else ''}"
        )

    print(f"best_saved={args.save}, best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
