"""
Stage2：文本条件图序列微调。
序列：[BPE文本] BOS_G [N_tok] [parents] SEP [extra_edges] EOS_G
Loss：只计算 BOS_G 之后的图序列部分。

用法：
  python -m llm_graph.train_stage2
  python -m llm_graph.train_stage2 \
      --stage1-ckpt checkpoints/llm_graph/stage1/best.pt \
      --resume      checkpoints/llm_graph/stage2/best.pt
"""

import argparse
import gc
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import LlamaConfig, LlamaForCausalLM

from llm_graph.dataset import make_loader
from llm_graph.metrics import compute_metrics


VOCAB_SIZE = 10084
PAD_ID     = 10000
BOS_ID     = 10001
EOS_ID     = 10002


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          default="data/processed/graph_tree/text_graph_tree.npz")
    p.add_argument("--vocab",         default="llm_graph/vocab/vocab_config.json")
    p.add_argument("--save-dir",      default="checkpoints/llm_graph/stage2")
    p.add_argument("--stage1-ckpt",   default=None,
                   help="Stage1 checkpoint 初始化权重")
    p.add_argument("--resume",        default=None)

    p.add_argument("--batch-size",    type=int,   default=24)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--log-every",     type=int,   default=200)
    p.add_argument("--save-every",    type=int,   default=5_000)

    p.add_argument("--hidden-size",       type=int, default=512)
    p.add_argument("--num-layers",        type=int, default=8)
    p.add_argument("--num-heads",         type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--max-pos-emb",       type=int, default=384)
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


def make_labels(tokens, text_lens, pad_id):
    B, L = tokens.shape
    y = tokens[:, 1:].clone()
    for i in range(B):
        tl = max(0, int(text_lens[i]) - 1)
        if tl > 0:
            y[i, :tl] = -100
    y[y == pad_id] = -100
    return y


def main():
    warnings.filterwarnings('ignore', category=UserWarning,
                            module='torch.optim.lr_scheduler')
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    torch.backends.cudnn.benchmark = True

    nw = 0 if platform.system() == 'Windows' else 4
    loader = make_loader(args.data, args.batch_size, stage=2,
                         pad_id=PAD_ID, shuffle=True, num_workers=nw)
    steps_per_epoch = len(loader.dataset) // args.batch_size
    total_steps     = args.epochs * steps_per_epoch
    print(f'数据集: {len(loader.dataset)} 条  '
          f'batch={args.batch_size}  '
          f'steps/epoch={steps_per_epoch}  '
          f'epochs={args.epochs}  '
          f'total_steps={total_steps}')

    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_pos_emb,
        bos_token_id=BOS_ID, eos_token_id=EOS_ID,
        pad_token_id=PAD_ID, rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(cfg).to(device)

    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
        sd   = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'stage1 ckpt loaded | missing={len(missing)} unexpected={len(unexpected)}')
    elif not args.resume:
        print('警告: 未提供 stage1 checkpoint，从零初始化')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'使用 {torch.cuda.device_count()} 张 GPU')
    print(f'参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    opt = AdamW(model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / 'log.jsonl'
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
    print(f'日志: {log_path}')

    start_step = 0
    best_loss  = float('inf')
    running    = {'loss': 0.0, 'acc_all': 0.0, 'acc_N': 0.0,
                  'acc_parent': 0.0, 'acc_edge': 0.0}
    save_win_loss  = 0.0
    save_win_steps = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        sd   = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        raw  = model.module if hasattr(model, 'module') else model
        raw.load_state_dict(sd, strict=True)
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        best_loss  = ckpt.get('best_loss', float('inf'))
        print(f'resumed from step {start_step} / {total_steps}  '
              f'({start_step/steps_per_epoch:.1f} epochs done)')

    def infinite():
        while True:
            yield from loader

    data_iter = infinite()
    t0 = time.perf_counter()
    model.train()

    for step in range(start_step, total_steps):
        tokens, mask, text_lens = next(data_iter)
        tokens    = tokens.to(device)
        mask      = mask.to(device)
        text_lens = text_lens.to(device)

        x = tokens[:, :-1]
        y = make_labels(tokens, text_lens, PAD_ID).to(device)

        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out    = model(input_ids=x, attention_mask=mask[:, :-1])
            logits = out.logits
            loss   = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            m = compute_metrics(logits.float(), y, text_lens)

        lv = loss.item()
        running['loss']       += lv
        running['acc_all']    += m['acc_all']
        running['acc_N']      += m['acc_N'] if m['acc_N'] == m['acc_N'] else 0
        running['acc_parent'] += m['acc_parent'] if m['acc_parent'] == m['acc_parent'] else 0
        running['acc_edge']   += m['acc_edge'] if m['acc_edge'] == m['acc_edge'] else 0
        save_win_loss  += lv
        save_win_steps += 1

        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if step % args.log_every == 0 and step > 0:
            n   = args.log_every
            avg = {k: v / n for k, v in running.items()}
            running = {k: 0.0 for k in running}
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()
            lr_now = scheduler.get_last_lr()[0]

            print(f'step {step:7d} | loss {avg["loss"]:.4f} '
                  f'| acc_all {avg["acc_all"]:.3f} '
                  f'| acc_N {avg["acc_N"]:.3f} '
                  f'| acc_parent {avg["acc_parent"]:.3f} '
                  f'| acc_edge {avg["acc_edge"]:.3f} '
                  f'| lr {lr_now:.2e} | {elapsed:.1f}s')

            log_file.write(json.dumps({
                'step': step, **{k: round(v, 4) for k, v in avg.items()},
                'lr': round(lr_now, 8), 'elapsed': round(elapsed, 1),
            }, ensure_ascii=False) + '\n')

        if step % args.save_every == 0 and step > 0:
            win_avg = save_win_loss / max(save_win_steps, 1)
            save_win_loss = save_win_steps = 0
            if win_avg < best_loss:
                best_loss = win_avg
                raw = model.module if hasattr(model, 'module') else model
                torch.save({
                    'model': raw.state_dict(), 'opt': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(),
                    'step': step, 'best_loss': best_loss,
                }, save_dir / 'best.pt')
                print(f'  best saved → step={step} loss={best_loss:.4f}')

    raw = model.module if hasattr(model, 'module') else model
    torch.save({'model': raw.state_dict(), 'opt': opt.state_dict(),
                'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(),
                'step': total_steps, 'best_loss': best_loss},
               save_dir / 'final.pt')
    log_file.close()
    print('Stage2 训练完成')


if __name__ == '__main__':
    main()
