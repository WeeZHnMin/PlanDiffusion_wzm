"""
端到端推理耗时基准测试。

GPU 模式：对同一条输入重复推理 3 次，打印各阶段耗时及平均总耗时。
CPU 模式：推理 1 次，打印各阶段耗时及总耗时。
两种模式均打印设备型号。

用法（项目根目录）：
    python benchmark_e2e.py \
        --ckpt1 checkpoints/llm_graph/stage2/20260614_155601/latest.pt \
        --ckpt2 checkpoints/node_diffusion_cross_att/latest.pt \
        --ckpt3 checkpoints/node_type/20260616_223156/model_latest.pt
"""

import argparse
import platform
import time

import numpy as np
import torch
from pathlib import Path
from tokenizers import Tokenizer
from transformers import BertTokenizer

from llm_graph.infer_stage1 import (
    load_model as load_llm,
    parse_sequence,
    generate,
    encode_text,
    BOS_ID,
)
from llm_graph.infer_batch import MAX_BERT_LEN
from node_diffusion_cross_att.model import NodeDiffusionTransformer
from node_diffusion_cross_att.diffusion import GaussianDiffusion
from node_diffusion_cross_att.type_model import NodeTypeClassifier
from node_diffusion_cross_att.render import (
    load_vocab, find_faces, vote_room_type, _build_sorted_neighbors,
)
from node_diffusion_cross_att.visualize_e2e import (
    CUSTOM_PROMPTS, sample_coords_batch,
)

TEST_TEXT = CUSTOM_PROMPTS[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt1', default='checkpoints/llm_graph/stage2/20260614_155601/latest.pt')
    p.add_argument('--ckpt2', default='checkpoints/node_diffusion_cross_att/latest.pt')
    p.add_argument('--ckpt3', default='checkpoints/node_type/20260616_223156/model_latest.pt')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--bert',        default='models/bert-base-uncased')
    p.add_argument('--combo_vocab', default='data/processed/type_combo_vocab.json')
    p.add_argument('--runs', type=int, default=3,
                   help='GPU 模式下重复推理次数（CPU 模式固定为 1）')
    return p.parse_args()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def print_device(device):
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'设备: GPU  {name}  ({mem:.1f} GB)')
    else:
        cpu = platform.processor() or platform.machine()
        print(f'设备: CPU  {cpu}')


def run_once(model1, model2, diffusion, model3, id_to_combo,
             bert_tok, prefix, device, run_idx=0):
    """执行完整一次端到端推理，返回各阶段耗时（秒）。"""

    # ── θ₁ ───────────────────────────────────────────────────────────────────
    sync(device)
    t0 = time.perf_counter()

    torch.manual_seed(run_idx + 777777)
    if device.type == 'cuda':
        torch.cuda.manual_seed(run_idx + 777777)
    gen_seq = generate(model1, prefix, device, max_new_tokens=200)
    parsed  = parse_sequence(gen_seq)

    sync(device)
    t1_time = time.perf_counter() - t0

    if not parsed['valid']:
        raise RuntimeError('θ₁ 生成无效图，换一条输入或检查模型')

    N      = parsed['n_nodes']
    adj_np = np.zeros((40, 40), dtype=np.float32)
    adj_np[:N, :N] = np.array(parsed['adj'], dtype=np.float32)
    mask_np        = np.zeros(40, dtype=np.float32)
    mask_np[:N]    = 1.0

    enc     = bert_tok(TEST_TEXT, max_length=MAX_BERT_LEN,
                       padding='max_length', truncation=True)
    ptok_np = np.array(enc['input_ids'],      dtype=np.int64)
    pmsk_np = np.array(enc['attention_mask'], dtype=np.float32)

    # ── θ₂ ───────────────────────────────────────────────────────────────────
    sync(device)
    t1 = time.perf_counter()

    with torch.no_grad():
        pred_coords = sample_coords_batch(
            model2, diffusion,
            adj_np[None], mask_np[None],
            ptok_np[None], pmsk_np[None],
            device, sample_indices=[run_idx],
        )[0]   # [40, 2]

    sync(device)
    t2_time = time.perf_counter() - t1

    # ── θ₃ ───────────────────────────────────────────────────────────────────
    sync(device)
    t2 = time.perf_counter()

    with torch.no_grad():
        x_in   = torch.from_numpy(pred_coords.T[None]).to(device)
        adj_t  = torch.from_numpy(adj_np[None]).to(device)
        msk_t  = torch.from_numpy(mask_np[None]).to(device)
        ptk_t  = torch.from_numpy(ptok_np[None]).to(device)
        pmk_t  = torch.from_numpy(pmsk_np[None]).long().to(device)
        logits = model3(x_in, adj_matrix=adj_t, node_mask=msk_t,
                        prompt_tokens=ptk_t, prompt_mask=pmk_t)
        type_ids = logits[0].argmax(dim=-1).cpu().numpy()

    sync(device)
    t3_time = time.perf_counter() - t2

    # ── render（CPU） ──────────────────────────────────────────────────────────
    t3 = time.perf_counter()

    raw_coords = [(float(pred_coords[i, 0]), float(pred_coords[i, 1])) for i in range(N)]
    adj_list   = [[int(adj_np[i, j]) for j in range(N)] for i in range(N)]
    combo_ids  = [int(type_ids[i]) for i in range(N)]
    node_types = [id_to_combo.get(cid, ['other']) for cid in combo_ids]
    nbrs  = _build_sorted_neighbors(raw_coords, adj_list, N)
    faces = find_faces(raw_coords, adj_list)
    for f in faces:
        vote_room_type(f, node_types, nbrs)

    t4_time = time.perf_counter() - t3

    total = t1_time + t2_time + t3_time + t4_time
    return dict(theta1=t1_time, theta2=t2_time, theta3=t3_time,
                render=t4_time, total=total, n_nodes=N)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs   = args.runs if device.type == 'cuda' else 1

    print('=' * 60)
    print_device(device)
    print(f'推理次数: {runs}')
    print(f'测试文本: {TEST_TEXT[:60]}...')
    print('=' * 60)

    id_to_combo = load_vocab(Path(args.combo_vocab))
    bert_tok    = BertTokenizer.from_pretrained(args.bert)
    bpe_tok     = Tokenizer.from_file(args.vocab)
    prefix      = encode_text(TEST_TEXT, args.vocab) + [BOS_ID]

    # ── 加载三个模型（只加载一次） ──────────────────────────────────────────────
    print('\n[加载模型]')
    model1 = load_llm(args.ckpt1, device)
    model1.eval()

    model2 = NodeDiffusionTransformer(bert_name=args.bert).to(device)
    ckpt2  = torch.load(args.ckpt2, map_location=device)
    model2.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt2['model'].items()}, strict=False)
    model2.eval()
    del ckpt2
    diffusion = GaussianDiffusion(timesteps=1000)

    model3 = NodeTypeClassifier(bert_name=args.bert).to(device)
    ckpt3  = torch.load(args.ckpt3, map_location=device)
    model3.load_state_dict(
        {k.replace('module.', ''): v for k, v in ckpt3['model'].items()})
    model3.eval()
    del ckpt3
    print('模型加载完毕\n')

    # ── 推理计时 ───────────────────────────────────────────────────────────────
    results = []
    for i in range(runs):
        print(f'--- Run {i+1}/{runs} ---')
        r = run_once(model1, model2, diffusion, model3, id_to_combo,
                     bert_tok, prefix, device, run_idx=i)
        results.append(r)
        print(f'  θ₁ (LLM autoregressive): {r["theta1"]:.2f}s')
        print(f'  θ₂ (DDPM 1000 steps):    {r["theta2"]:.2f}s')
        print(f'  θ₃ (type classifier):    {r["theta3"]:.3f}s')
        print(f'  render (CPU):            {r["render"]:.3f}s')
        print(f'  total:                   {r["total"]:.2f}s  (n_nodes={r["n_nodes"]})')

    # ── 汇总 ───────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    if runs > 1:
        for key in ('theta1', 'theta2', 'theta3', 'render', 'total'):
            vals = [r[key] for r in results]
            print(f'  {key:8s}  avg={np.mean(vals):.2f}s  '
                  f'min={np.min(vals):.2f}s  max={np.max(vals):.2f}s')
    else:
        r = results[0]
        print(f'  θ₁: {r["theta1"]:.2f}s  θ₂: {r["theta2"]:.2f}s  '
              f'θ₃: {r["theta3"]:.3f}s  render: {r["render"]:.3f}s  '
              f'total: {r["total"]:.2f}s')
    print('=' * 60)


if __name__ == '__main__':
    main()
