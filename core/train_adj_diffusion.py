"""
Adjacency matrix diffusion trainer (continuous relaxation).
adj values {0,1} normalized to {-1,+1}, Gaussian noise added, epsilon-prediction.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--save", type=Path, default=Path("core/weights/adj_diffusion.pt"))
    parser.add_argument("--token-cache", type=Path, default=Path("data/jsonl/train_nodes.tokens.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bert-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=40)
    return parser.parse_args()


def load_jsonl_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    if not records:
        raise SystemExit(f"No valid records in {path}")
    return records


def build_or_load_token_cache(tokenizer, prompts, max_length, cache_path):
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        ids = cache["input_ids"]
        mask = cache["attention_mask"]
        if ids.size(0) == len(prompts) and ids.size(1) == max_length:
            print(f"token cache hit: {cache_path}")
            return ids, mask
        print("token cache shape mismatch, rebuilding...")
    print("tokenizing...")
    enc = tokenizer(prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_length)
    ids = enc["input_ids"].contiguous()
    mask = enc["attention_mask"].contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": ids, "attention_mask": mask}, cache_path)
    print(f"token cache saved: {cache_path}")
    return ids, mask


def cosine_alpha_bars(timesteps, s=0.008):
    ts = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((ts / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return ab[1:].float().clamp(min=1e-5)


def q_sample(x0, t_idx, noise, alpha_bars):
    ab = alpha_bars[t_idx].view(-1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def sinusoidal_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class AdjDiffusionModel(nn.Module):
    """
    Each node's noisy adj row (n_max values) is projected to d_model.
    Time embedding added. Cross-attend to text tokens.
    Bilinear head outputs predicted noise for each node pair.
    """
    def __init__(self, n_max, d_model, n_heads, n_layers):
        super().__init__()
        self.n_max = n_max
        self.d_model = d_model
        # project noisy adj row (n_max dims) to d_model
        self.row_proj = nn.Sequential(
            nn.Linear(n_max, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(n_max, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_proj = nn.Linear(768, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.edge_w = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def forward(self, noisy_adj, t_idx, text_enc, text_mask, node_mask):
        # noisy_adj: (B, N, N)  — each row is the noisy connections of one node
        bsz = noisy_adj.size(0)
        pos_idx = torch.arange(self.n_max, device=noisy_adj.device).unsqueeze(0)
        time_emb = self.time_proj(sinusoidal_emb(t_idx, self.d_model)).unsqueeze(1)

        x = self.row_proj(noisy_adj) + self.pos_emb(pos_idx) + time_emb
        memory = self.text_proj(text_enc)
        text_key_pad = (text_mask == 0)
        node_key_pad = (node_mask == 0)

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_key_padding_mask=node_key_pad,
            memory_key_padding_mask=text_key_pad,
        )
        x = self.final_norm(x)
        # bilinear pairwise: (B, N, N)
        logits = torch.einsum("bnd,df,bmf->bnm", x, self.edge_w, x)
        # enforce symmetry
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(42)

    print("loading records...")
    records = load_jsonl_records(args.data)
    n_total = len(records)
    n_max = args.n_max

    val_size = max(1, int(n_total * args.val_ratio))
    train_n = n_total - val_size
    print(f"records={n_total}, train={train_n}, val={val_size}, n_max={n_max}")

    print("building tensors...")
    adj_raw = torch.zeros((n_total, n_max, n_max), dtype=torch.float32)
    node_masks = torch.zeros((n_total, n_max), dtype=torch.float32)
    prompts = []

    for i, r in enumerate(records):
        n_nodes = int(r["n_nodes"])
        prompts.append(r["prompt"])
        node_masks[i, :n_nodes] = 1.0
        raw_adj = r["adj_matrix"]
        for ri in range(min(len(raw_adj), n_max)):
            for ci in range(min(len(raw_adj[ri]), n_max)):
                adj_raw[i, ri, ci] = float(raw_adj[ri][ci])
        if (i + 1) % 10000 == 0:
            print(f"  prepared {i+1}/{n_total}")

    # normalize adj {0,1} -> {-1,+1}
    adj_scaled = adj_raw * 2.0 - 1.0

    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert))
    bert = BertModel.from_pretrained(str(args.bert)).to(device)
    for name, p in bert.named_parameters():
        layer_id = None
        for part in name.split("."):
            if part.isdigit():
                layer_id = int(part)
                break
        p.requires_grad = layer_id is not None and layer_id >= 10
    print(f"bert trainable: {sum(p.numel() for p in bert.parameters() if p.requires_grad):,}")

    input_ids, attention_mask = build_or_load_token_cache(
        tokenizer, prompts, args.max_length, args.token_cache
    )

    dataset = TensorDataset(input_ids, attention_mask, adj_scaled, node_masks)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]
    train_ds = TensorDataset(
        input_ids[train_idx], attention_mask[train_idx],
        adj_scaled[train_idx], node_masks[train_idx],
    )
    val_ds = TensorDataset(
        input_ids[val_idx], attention_mask[val_idx],
        adj_scaled[val_idx], node_masks[val_idx],
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            persistent_workers=(args.num_workers > 0),
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    model = AdjDiffusionModel(n_max=n_max, d_model=args.d_model,
                              n_heads=args.n_heads, n_layers=args.n_layers).to(device)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")

    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr},
         {"params": bert_params, "lr": args.bert_lr}],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    alpha_bars = cosine_alpha_bars(args.timesteps).to(device)
    best_val = float("inf")
    global_step = 0

    # upper-triangle mask for loss (only unique node pairs, no diagonal)
    tri_mask = torch.triu(torch.ones(n_max, n_max, dtype=torch.bool, device=device), diagonal=1)

    for epoch in range(args.epochs):
        model.train()
        bert.train()
        train_loss_sum = 0.0
        train_items = 0

        for b_ids, b_mask, b_adj, b_nmask in train_loader:
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_adj = b_adj.to(device, non_blocking=True)
            b_nmask = b_nmask.to(device, non_blocking=True)

            t_idx = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
            noise = torch.randn_like(b_adj)
            x_t = q_sample(b_adj, t_idx, noise, alpha_bars)

            # only add noise to valid node pairs, zero out padding
            valid = b_nmask.unsqueeze(-1) * b_nmask.unsqueeze(-2)
            x_t = x_t * valid

            with torch.amp.autocast("cuda", enabled=use_amp):
                text_enc = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                text_mask_f = b_mask.float()
                pred_eps = model(x_t, t_idx, text_enc, text_mask_f, b_nmask)
                # loss only on upper triangle of valid node pairs
                valid_tri = valid.bool() & tri_mask.unsqueeze(0)
                loss = ((pred_eps - noise)[valid_tri] ** 2).mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss_sum += loss.item() * b_adj.size(0)
            train_items += b_adj.size(0)
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch={epoch+1:4d} step={global_step:7d} train_loss={train_loss_sum/max(train_items,1):.6f}")

        sched.step()

        model.eval()
        bert.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for b_ids, b_mask, b_adj, b_nmask in val_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                b_adj = b_adj.to(device, non_blocking=True)
                b_nmask = b_nmask.to(device, non_blocking=True)

                t_idx = torch.randint(0, args.timesteps, (b_adj.size(0),), device=device)
                noise = torch.randn_like(b_adj)
                valid = b_nmask.unsqueeze(-1) * b_nmask.unsqueeze(-2)
                x_t = q_sample(b_adj, t_idx, noise, alpha_bars) * valid
                valid_tri = valid.bool() & tri_mask.unsqueeze(0)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    text_enc = bert(input_ids=b_ids, attention_mask=b_mask).last_hidden_state
                    text_mask_f = b_mask.float()
                    pred_eps = model(x_t, t_idx, text_enc, text_mask_f, b_nmask)
                    loss = ((pred_eps - noise)[valid_tri] ** 2).mean()

                val_loss_sum += loss.item() * b_adj.size(0)
                val_items += b_adj.size(0)

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss = val_loss_sum / max(val_items, 1)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_max": n_max, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "timesteps": args.timesteps, "pred_type": "epsilon",
                "bert_path": str(args.bert), "epoch": epoch + 1,
                "best_val_loss": best_val,
            }, args.save)
        print(f"epoch={epoch+1:4d} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f} {'(saved)' if improved else ''}")

    print(f"best_saved={args.save}, best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
