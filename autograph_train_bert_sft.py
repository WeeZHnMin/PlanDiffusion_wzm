"""
Prompt -> graph token sequence generation (prefix-concat, decoder-only).

Setup:
- Text encoder: lightweight random-init Transformer, using bert-base-chinese tokenizer vocab
- Decoder: GPT-2 decoder-only (no cross-attention), conditioned via prefix concat
- Training data: reproducibly sample 40k examples from the old prompt/token dataset

Default inputs:
- data/processed/graph_tokens_combo_from_final_old.npz
- data/processed/graph_prompts_combo_from_final_old.txt
- data/processed/type_combo_vocab_old.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel


@dataclass
class VocabConfig:
    pad_id: int
    bos_id: int
    eos_id: int
    vocab_size: int
    max_nodes: int
    node_offset: int


def load_vocab(path: Path) -> VocabConfig:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return VocabConfig(
        pad_id=0,
        bos_id=int(vocab["BOS_ID"]),
        eos_id=int(vocab["EOS_ID"]),
        vocab_size=int(vocab["VOCAB_SIZE"]),
        max_nodes=int(vocab["MAX_NODES"]),
        node_offset=int(vocab["NODE_OFFSET"]),
    )


class PromptTokenDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        prompt_txt_path: Path,
        bert_path: Path,
        subset_size: int,
        subset_seed: int,
        max_text_len: int,
        subset_index_out: Path | None = None,
    ):
        raw = np.load(npz_path, allow_pickle=True)
        tokens = raw["tokens"].astype(np.int32)
        lengths = raw["lengths"].astype(np.int32)

        prompts = prompt_txt_path.read_text(encoding="utf-8").splitlines()
        usable_n = min(len(tokens), len(prompts))
        if len(tokens) != len(prompts):
            print(
                f"warning: prompt/token count mismatch, using first {usable_n} pairs "
                f"(npz={len(tokens)}, txt={len(prompts)})"
            )

        tokens = tokens[:usable_n]
        lengths = lengths[:usable_n]
        prompts = prompts[:usable_n]

        rng = np.random.default_rng(subset_seed)
        subset_size = min(subset_size, usable_n)
        subset_indices = np.sort(rng.choice(usable_n, size=subset_size, replace=False))
        if subset_index_out is not None:
            subset_index_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(subset_index_out, subset_indices.astype(np.int32))

        self.tokens = tokens[subset_indices]
        self.lengths = lengths[subset_indices]
        self.prompts = [prompts[i] for i in subset_indices]
        self.indices = subset_indices
        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_path))
        print(
            f"PromptTokenDataset: total_pairs={usable_n}, subset={len(self.tokens)}, "
            f"max_seq={self.tokens.shape[1]}, max_text_len={self.max_text_len}"
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        seq_len = int(self.lengths[idx])
        seq = self.tokens[idx, :seq_len]
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target_tokens": torch.tensor(seq, dtype=torch.long),
            "sample_index": int(self.indices[idx]),
        }


def collate_batch(batch, pad_id: int):
    batch_size = len(batch)
    max_tgt_len = max(item["target_tokens"].shape[0] for item in batch)

    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    target_tokens = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    target_mask = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
    sample_indices = torch.tensor([item["sample_index"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        seq = item["target_tokens"]
        target_tokens[i, : seq.shape[0]] = seq
        target_mask[i, : seq.shape[0]] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
        "sample_indices": sample_indices,
    }


class LightTextEncoder(nn.Module):
    """随机初始化的轻量文本编码器，借用BERT中文词汇表（21128 tokens）。"""
    def __init__(self, bert_vocab_size=21128, d_model=384, nhead=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(bert_vocab_size, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        pad_mask = (attention_mask == 0)  # True表示该位置被忽略
        return self.transformer(x, src_key_padding_mask=pad_mask)


class PrefixGraphModel(nn.Module):
    def __init__(
        self,
        bert_path: Path,
        graph_vocab_size: int,
        max_target_len: int,
        max_text_len: int = 256,
        bert_vocab_size: int = 21128,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 8,
        dropout: float = 0.1,
        decoder_init_ckpt: Path | None = None,
        bos_id: int = 36,
        eos_id: int = 37,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = LightTextEncoder(bert_vocab_size, d_model, nhead, num_encoder_layers, dropout)
        self.max_text_len = max_text_len
        decoder_cfg = GPT2Config(
            vocab_size=graph_vocab_size,
            n_embd=d_model,
            n_layer=num_decoder_layers,
            n_head=nhead,
            n_positions=max_target_len + max_text_len,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            add_cross_attention=False,
        )
        self.decoder = GPT2LMHeadModel(decoder_cfg)

        if decoder_init_ckpt is not None:
            ckpt = torch.load(decoder_init_ckpt, map_location="cpu")
            missing, unexpected = self.decoder.load_state_dict(ckpt, strict=False)
            print(
                f"decoder init from {decoder_init_ckpt} | "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print("  missing sample:", missing[:8])
            if unexpected:
                print("  unexpected sample:", unexpected[:8])

    def forward(self, input_ids, attention_mask, target_tokens, target_mask):
        # 1. 文本 → prefix embeddings  (B, text_len, d_model)
        prefix = self.encoder(input_ids, attention_mask)

        # 2. 图token → embeddings  (B, seq_len, d_model)
        token_emb = self.decoder.transformer.wte(target_tokens)

        # 3. prefix concat graph embeddings
        inputs_embeds = torch.cat([prefix, token_emb], dim=1)

        # 4. attention mask拼接
        full_mask = torch.cat([attention_mask, target_mask], dim=1)

        # 5. prefix部分label设为-100，不计入loss
        prefix_labels = torch.full(
            (target_tokens.shape[0], prefix.shape[1]), -100,
            dtype=torch.long, device=target_tokens.device
        )
        full_labels = torch.cat([prefix_labels, target_tokens], dim=1)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            use_cache=False,
        )
        return out.logits, out.loss


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("data/processed/graph_tokens_combo_from_final_old.npz"))
    parser.add_argument("--prompts", type=Path, default=Path("data/processed/graph_prompts_combo_from_final_old.txt"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/type_combo_vocab_old.json"))
    parser.add_argument("--bert_path", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--decoder_init_ckpt", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/autograph_prefix_sft"))
    parser.add_argument("--subset_size", type=int, default=40000)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--subset_index_out", type=Path, default=Path("checkpoints/autograph_prefix_sft/subset_indices.npy"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision training")
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto")
    return parser


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_amp_dtype(device: torch.device, amp_enabled: bool, amp_dtype: str):
    if not amp_enabled or device.type != "cuda":
        return None
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_optimizer(model: PrefixGraphModel, lr: float, weight_decay: float):
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def shift_tokens_for_teacher_forcing(target_tokens: torch.Tensor, bos_id: int):
    decoder_in = target_tokens.clone()
    decoder_in[:, 1:] = target_tokens[:, :-1]
    decoder_in[:, 0] = bos_id
    return decoder_in


def masked_token_accuracy(logits: torch.Tensor, target_tokens: torch.Tensor, pad_id: int) -> float:
    pred = logits.argmax(dim=-1)
    valid = target_tokens.ne(pad_id)
    correct = pred.eq(target_tokens) & valid
    denom = valid.sum().item()
    if denom == 0:
        return 0.0
    return correct.sum().item() / denom


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vocab = load_vocab(args.vocab)
    dataset = PromptTokenDataset(
        npz_path=args.npz,
        prompt_txt_path=args.prompts,
        bert_path=args.bert_path,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        max_text_len=args.max_text_len,
        subset_index_out=args.subset_index_out,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id),
        drop_last=True,
    )

    bert_tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    bert_vocab_size = bert_tokenizer.vocab_size  # 21128

    model = PrefixGraphModel(
        bert_path=args.bert_path,
        graph_vocab_size=vocab.vocab_size,
        max_target_len=int(dataset.tokens.shape[1]),
        max_text_len=args.max_text_len,
        bert_vocab_size=bert_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        decoder_init_ckpt=args.decoder_init_ckpt,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        pad_id=vocab.pad_id,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"trainable params: total={trainable_params:,} encoder={encoder_params:,} decoder={decoder_params:,}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    amp_dtype = choose_amp_dtype(device, not args.no_amp, args.amp_dtype)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    print(
        f"amp: {'on' if amp_dtype is not None else 'off'}"
        + (f" ({str(amp_dtype).replace('torch.', '')})" if amp_dtype is not None else "")
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["vocab_size"] = vocab.vocab_size
    run_config["bos_id"] = vocab.bos_id
    run_config["eos_id"] = vocab.eos_id
    with open(args.save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.max_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.perf_counter()

        for batch in loader:
            step_start = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            decoder_input = shift_tokens_for_teacher_forcing(target_tokens, vocab.bos_id)
            decoder_attention_mask = batch["target_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, loss = model(input_ids, attention_mask, decoder_input, decoder_attention_mask)
            prefix_len = input_ids.shape[1]
            graph_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_tokens.shape[1], :]
            token_acc = masked_token_accuracy(graph_logits.detach(), target_tokens, vocab.pad_id)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                step_time = time.perf_counter() - step_start
                samples_per_sec = args.batch_size / max(step_time, 1e-6)
                print(
                    f"epoch {epoch + 1:2d} | step {global_step:6d} | "
                    f"loss {loss.item():.4f} | acc {token_acc:.4f} | {samples_per_sec:.1f} samples/s"
                )

            if global_step % args.save_every == 0:
                ckpt_path = args.save_dir / f"step_{global_step:07d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch + 1,
                        "args": run_config,
                    },
                    ckpt_path,
                )
                print(f"  saved -> {ckpt_path}")

        avg_loss = running_loss / len(loader)
        epoch_time = time.perf_counter() - epoch_start
        print(f"=== epoch {epoch + 1} done | avg_loss={avg_loss:.4f} | epoch_time={epoch_time:.1f}s ===")

        epoch_ckpt = args.save_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "args": run_config,
            },
            epoch_ckpt,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.save_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "epoch": epoch + 1,
                    "args": run_config,
                },
                best_path,
            )
            print(f"  best updated -> {best_path} | best_loss={best_loss:.4f}")

    print(f"training done | best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
