"""
Train text -> adjacency(0/1) with a Transformer Decoder head.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/jsonl/train_nodes.jsonl"))
    parser.add_argument("--bert", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--n-max", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--prefetch-factor", type=int, default=6)
    parser.add_argument("--token-cache", type=Path, default=Path("data/jsonl/train_nodes.tokens.pt"))
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help=">0 use fixed pos_weight; <=0 auto-estimate from train set")
    parser.add_argument("--save", type=Path, default=Path("core/weights/adj_text_decoder.pt"))
    return parser.parse_args()


def scan_jsonl(path: Path):
    offsets = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] skip bad json near byte offset {pos}")
                continue
            offsets.append(pos)
    if not offsets:
        raise SystemExit(f"No valid records in {path}")
    if bad:
        print(f"[WARN] skipped bad lines: {bad}")
    return offsets


def read_batch(path: Path, offsets, s: int, e: int):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i in range(s, e):
            f.seek(offsets[i])
            line = f.readline().strip()
            records.append(json.loads(line))
    return records


class TextToAdjDecoder(nn.Module):
    def __init__(self, n_max: int, d_model: int, nhead: int, layers: int, ffn: int):
        super().__init__()
        self.n_max = n_max
        self.text_proj = nn.Linear(768, d_model)
        self.query_embed = nn.Parameter(torch.randn(n_max, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.edge_w = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.edge_b = nn.Parameter(torch.zeros(1))

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        memory = self.text_proj(text_tokens)
        bsz = memory.size(0)
        tgt = self.query_embed.unsqueeze(0).expand(bsz, self.n_max, -1)
        h = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=(text_mask == 0),
        )
        logits = torch.einsum("bnd,df,bmf->bnm", h, self.edge_w, h) + self.edge_b
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits


def build_masks(batch_n: torch.Tensor, n_max: int, device: torch.device):
    bsz = int(batch_n.size(0))
    valid = torch.zeros((bsz, n_max, n_max), dtype=torch.float32, device=device)
    for bi, cnt in enumerate(batch_n.tolist()):
        valid[bi, :cnt, :cnt] = 1.0
    pad = (1.0 - valid).bool()
    return valid, pad


def compute_binary_metrics(preds: torch.Tensor, labels: torch.Tensor):
    preds_f = preds.float()
    labels_f = labels.float()
    tp = float(((preds_f == 1) & (labels_f == 1)).sum().item())
    fp = float(((preds_f == 1) & (labels_f == 0)).sum().item())
    fn = float(((preds_f == 0) & (labels_f == 1)).sum().item())
    pred_pos = float((preds_f == 1).sum().item())
    gt_pos = float((labels_f == 1).sum().item())
    total = float(labels_f.numel())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    pred_pos_rate = pred_pos / (total + 1e-12)
    gt_pos_rate = gt_pos / (total + 1e-12)
    return pred_pos_rate, gt_pos_rate, precision, recall, f1


def estimate_pos_weight(adj_tensor: torch.Tensor, n_nodes_tensor: torch.Tensor, n_max: int) -> float:
    tri = torch.triu(torch.ones((n_max, n_max), dtype=torch.bool), diagonal=1)
    pos = 0.0
    total = 0.0
    for i in range(adj_tensor.size(0)):
        n = int(n_nodes_tensor[i].item())
        m = torch.zeros((n_max, n_max), dtype=torch.bool)
        m[:n, :n] = True
        mask = m & tri
        y = adj_tensor[i][mask]
        pos += float((y > 0.5).sum().item())
        total += float(y.numel())
    neg = max(total - pos, 1.0)
    pos = max(pos, 1.0)
    return neg / pos


def load_all_records(path: Path, offsets, n_max: int):
    prompts = []
    adjs = []
    n_nodes = []
    clipped = 0
    with path.open("r", encoding="utf-8") as f:
        for i, pos in enumerate(offsets):
            f.seek(pos)
            obj = json.loads(f.readline().strip())
            prompts.append(obj["prompt"])
            raw_adj = obj["adj_matrix"]
            raw_n = int(obj["n_nodes"])
            use_n = min(raw_n, n_max)
            if raw_n > n_max:
                clipped += 1

            adj = [[0.0] * n_max for _ in range(n_max)]
            for r in range(min(len(raw_adj), n_max)):
                row = raw_adj[r]
                for c in range(min(len(row), n_max)):
                    adj[r][c] = float(row[c])

            adjs.append(adj)
            n_nodes.append(use_n)
            if (i + 1) % 5000 == 0:
                print(f"loaded records: {i + 1}/{len(offsets)}")
    if clipped:
        print(f"[WARN] clipped n_nodes to n_max={n_max}: {clipped}")
    return prompts, adjs, n_nodes


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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.bert.exists():
        raise SystemExit(f"Missing bert path: {args.bert}")

    offsets = scan_jsonl(args.data)
    n_max = args.n_max
    n = len(offsets)
    val_size = max(1, int(n * args.val_ratio))
    if val_size >= n:
        val_size = max(1, n - 1)
    train_n = n - val_size
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng).tolist()
    train_offsets = [offsets[i] for i in perm[:train_n]]
    val_offsets = [offsets[i] for i in perm[train_n:]]
    use_amp = bool(args.amp and device.type == "cuda")

    print(
        f"records={n}, train={len(train_offsets)}, val={len(val_offsets)}, "
        f"n_max={n_max}, device={device}, amp={use_amp}"
    )

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
    bert_trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print(f"bert trainable params (last 2 layers): {bert_trainable:,}")

    model = TextToAdjDecoder(
        n_max=n_max,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ffn=args.ffn,
    ).to(device)
    bert_params = [p for p in bert.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": bert_params, "lr": 1e-5},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    global_step = 0

    print("loading dataset into memory...")
    prompts, adjs, n_nodes = load_all_records(args.data, offsets, n_max)
    input_ids, attention_mask = build_or_load_token_cache(tokenizer, prompts, args.max_length, args.token_cache)
    adj_tensor = torch.tensor(adjs, dtype=torch.float32)
    n_nodes_tensor = torch.tensor(n_nodes, dtype=torch.long)

    perm_t = torch.tensor(perm, dtype=torch.long)
    train_idx = perm_t[:train_n]
    val_idx = perm_t[train_n:]
    train_ds = TensorDataset(
        input_ids[train_idx],
        attention_mask[train_idx],
        adj_tensor[train_idx],
        n_nodes_tensor[train_idx],
    )
    val_ds = TensorDataset(
        input_ids[val_idx],
        attention_mask[val_idx],
        adj_tensor[val_idx],
        n_nodes_tensor[val_idx],
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

    if args.pos_weight > 0:
        pos_weight_value = float(args.pos_weight)
    else:
        pos_weight_value = estimate_pos_weight(adj_tensor[train_idx], n_nodes_tensor[train_idx], n_max)
    pos_weight_t = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    print(f"using pos_weight={pos_weight_value:.4f}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0

        bert.train()
        for batch in train_loader:
            b_input_ids, b_attn_mask, b_adj, b_n = batch
            b_input_ids = b_input_ids.to(device, non_blocking=True)
            b_attn_mask = b_attn_mask.to(device, non_blocking=True)
            batch_adj = b_adj.to(device, non_blocking=True)
            batch_n = b_n.to(device, non_blocking=True)
            batch_valid, _ = build_masks(batch_n, n_max, device)
            tri_mask = torch.triu(
                torch.ones((n_max, n_max), dtype=torch.bool, device=device), diagonal=1
            )
            loss_mask = batch_valid.bool() & tri_mask.unsqueeze(0)

            with torch.amp.autocast("cuda", enabled=use_amp):
                text_out = bert(input_ids=b_input_ids, attention_mask=b_attn_mask).last_hidden_state
                text_mask = b_attn_mask.float()
                logits = model(text_out, text_mask)
                logits_valid = logits[loss_mask]
                labels_valid = batch_adj[loss_mask]
                loss = F.binary_cross_entropy_with_logits(
                    logits_valid, labels_valid, pos_weight=pos_weight_t
                )

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(batch_adj.size(0))
            total_loss += loss.item() * bs
            total_items += bs
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch+1:4d} step={global_step:7d} "
                    f"train_loss={total_loss/max(total_items,1):.4f}"
                )

        model.eval()
        bert.eval()
        val_loss_sum = 0.0
        val_items = 0
        val_pred_pos = 0.0
        val_gt_pos = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids, b_attn_mask, b_adj, b_n = batch
                b_input_ids = b_input_ids.to(device, non_blocking=True)
                b_attn_mask = b_attn_mask.to(device, non_blocking=True)
                batch_adj = b_adj.to(device, non_blocking=True)
                batch_n = b_n.to(device, non_blocking=True)
                batch_valid, _ = build_masks(batch_n, n_max, device)
                tri_mask = torch.triu(
                    torch.ones((n_max, n_max), dtype=torch.bool, device=device), diagonal=1
                )
                loss_mask = batch_valid.bool() & tri_mask.unsqueeze(0)
                text_out = bert(input_ids=b_input_ids, attention_mask=b_attn_mask).last_hidden_state
                text_mask = b_attn_mask.float()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(text_out, text_mask)
                    logits_valid = logits[loss_mask]
                    labels_valid = batch_adj[loss_mask]
                    loss = F.binary_cross_entropy_with_logits(
                        logits_valid, labels_valid, pos_weight=pos_weight_t
                    )

                preds = (logits.sigmoid() > 0.5)
                preds_valid = preds[loss_mask]
                labels_valid_bin = (batch_adj[loss_mask] > 0.5)
                m_pred_pos, m_gt_pos, m_p, m_r, m_f1 = compute_binary_metrics(preds_valid, labels_valid_bin)
                val_pred_pos += m_pred_pos
                val_gt_pos += m_gt_pos
                val_precision += m_p
                val_recall += m_r
                val_f1 += m_f1
                val_batches += 1
                bs = int(batch_adj.size(0))
                val_loss_sum += loss.item() * bs
                val_items += bs

        train_loss_epoch = total_loss / max(total_items, 1)
        val_loss_epoch = val_loss_sum / max(val_items, 1)
        pred_pos_rate = val_pred_pos / max(val_batches, 1)
        gt_pos_rate = val_gt_pos / max(val_batches, 1)
        precision = val_precision / max(val_batches, 1)
        recall = val_recall / max(val_batches, 1)
        f1 = val_f1 / max(val_batches, 1)
        improved = val_loss_epoch < best_val_loss
        if improved:
            best_val_loss = val_loss_epoch
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_max": n_max,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "layers": args.layers,
                    "ffn": args.ffn,
                    "head": "transformer_decoder",
                    "bert_path": str(args.bert),
                    "data_path": str(args.data),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                },
                args.save,
            )

        print(
            f"epoch={epoch+1:4d} done "
            f"train_loss={train_loss_epoch:.4f} val_loss={val_loss_epoch:.4f} "
            f"pred_pos={pred_pos_rate*100:.2f}% gt_pos={gt_pos_rate*100:.2f}% "
            f"p={precision*100:.2f}% r={recall*100:.2f}% f1={f1*100:.2f}% "
            f"best_val_loss={best_val_loss:.4f} "
            f"{'(saved)' if improved else ''}"
        )

    print(f"best_saved={args.save}, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
