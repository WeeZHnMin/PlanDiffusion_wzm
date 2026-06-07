"""
Stage1/2 推理脚本：加载模型，对给定文本自回归生成图序列，
应用四个解码约束，解析邻接矩阵，并与真实数据对比。

用法：
  # 对比真实数据（默认3条）
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage1/best.pt

  # 额外生成3条内置默认提示词（不依赖数据集）
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage1/best.pt --demo

  # 自定义文本
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage1/best.pt --text "..."

  # 只跑默认提示词，不做数据集对比
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage1/best.pt --demo --compare-n 0
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# ── Token ID 常量 ─────────────────────────────────────────────────────────────
VOCAB_SIZE  = 10084
PAD_ID      = 10000
BOS_ID      = 10001   # BOS_G
EOS_ID      = 10002   # EOS_G
SEP_ID      = 10003
N_START     = 10004   # N=k → N_START+(k-1)
NODE_START  = 10044   # node_j → NODE_START+j
MAX_NODES   = 40

# ── 三条默认演示提示词（从真实数据挑选，覆盖简单/中等/复杂三个规模）────────────
DEFAULT_PROMPTS = [
    # 简单：4个房间，15节点，空间描述简洁
    (
        "the corridor is located between the bedroom and the kitchen ; "
        "the bathroom is adjacent to the bedroom ; "
        "the living room is adjacent to both the bedroom and the kitchen ."
    ),
    # 中等：5个房间+阳台，13节点，方位描述
    (
        "the living room is at the center of the house , connecting to the kitchen in the north , "
        "leading to the bedroom in the south , adjacent to the bathroom in the east , "
        "and accessing the balcony in the west ."
    ),
    # 复杂：2卧2卫+走廊，27节点，多条连接关系
    (
        "the kitchen is adjacent to the living room , which connects via a corridor to two bedrooms "
        "and two bathrooms ; one bedroom is adjacent to the kitchen , the other bedroom is adjacent "
        "to one of the bathrooms , and the two bathrooms are on opposite sides of the corridor ."
    ),
]

# ── 模型配置（与训练保持一致）────────────────────────────────────────────────
MODEL_CFG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1536,
    max_position_embeddings=256,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    pad_token_id=PAD_ID,
    rms_norm_eps=1e-5,
)


# ── 序列解析 ──────────────────────────────────────────────────────────────────

def parse_sequence(tokens: list[int]) -> dict:
    """
    解析生成序列，返回：
      n_nodes, parents, extra_edges, adj_matrix, valid
    tokens 应从 BOS_ID 开始（含 BOS_ID）。
    """
    result = dict(n_nodes=0, parents=[], extra_edges=[], adj=None, valid=False)
    try:
        i = 0
        # 跳过文本前缀，找到 BOS_G
        while i < len(tokens) and tokens[i] != BOS_ID:
            i += 1
        if i >= len(tokens):
            return result
        i += 1  # 跳过 BOS_G

        # 读取 N_tok
        if i >= len(tokens) or not (N_START <= tokens[i] < N_START + MAX_NODES):
            return result
        N = tokens[i] - N_START + 1
        result['n_nodes'] = N
        i += 1

        # 读取 N-1 个父节点
        parents = []
        for _ in range(N - 1):
            if i >= len(tokens) or not (NODE_START <= tokens[i] < NODE_START + N):
                return result
            parents.append(tokens[i] - NODE_START)
            i += 1
        result['parents'] = parents

        # 读取 SEP
        if i >= len(tokens) or tokens[i] != SEP_ID:
            return result
        i += 1

        # 读取补边（成对节点 token）
        extra_edges = []
        while i < len(tokens) and tokens[i] != EOS_ID:
            if i + 1 >= len(tokens):
                break
            if not (NODE_START <= tokens[i] < NODE_START + N):
                break
            if not (NODE_START <= tokens[i + 1] < NODE_START + N):
                break
            u = tokens[i] - NODE_START
            v = tokens[i + 1] - NODE_START
            extra_edges.append((u, v))
            i += 2
        result['extra_edges'] = extra_edges

        # 构建邻接矩阵
        adj = [[0] * N for _ in range(N)]
        # 生成树边（BFS parent）
        for k, p in enumerate(parents):
            adj[k + 1][p] = adj[p][k + 1] = 1
        # 补边
        for u, v in extra_edges:
            if 0 <= u < N and 0 <= v < N and u != v:
                adj[u][v] = adj[v][u] = 1
        result['adj'] = adj
        result['valid'] = True
    except Exception:
        pass
    return result


def adj_to_str(adj: list) -> str:
    """邻接矩阵打印为紧凑字符串。"""
    n = len(adj)
    lines = []
    for i in range(n):
        row = ''.join(str(adj[i][j]) for j in range(n))
        lines.append(f'  [{i:2d}] {row}')
    return '\n'.join(lines)


def degrees(adj: list) -> list[int]:
    return [sum(row) for row in adj]


# ── 文本编码 ──────────────────────────────────────────────────────────────────

def encode_text(text: str, tokenizer_path: str, max_len: int = 128) -> list[int]:
    """用 WordPiece 分词器将文本编码为 token id 列表（截断到 max_len）。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    enc = tok.encode(text)
    ids = enc.ids[:max_len]
    return ids


# ── 约束解码 ──────────────────────────────────────────────────────────────────

def has_triangle(adj: list, u: int, v: int) -> bool:
    """加入边 (u,v) 后是否产生三角环（u、v 有共同邻居）。"""
    n = len(adj)
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return True
    return False


@torch.no_grad()
def generate(
    model,
    prefix_ids: list[int],
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
) -> list[int]:
    """
    自回归生成，从 prefix_ids（含文本+BOS_G）之后开始生成图序列。
    应用四个约束：
      ① 因果树约束：生成 p_k 时只允许 node_j，j < k
      ② SEP 时机：生成满 N-1 个父节点前禁 SEP；之后强制 SEP
      ③ 禁止三角环：补边阶段屏蔽会产生三角的 token
      ④ 度数下限：所有节点度 >= 2 前禁止 EOS_G
    """
    NEG_INF = float('-inf')

    generated = list(prefix_ids)
    phase = 'N_tok'       # N_tok → parents → sep → edges → done
    N = 0
    parent_count = 0
    parents = []

    # 用于约束③④的运行时图状态
    run_adj = None   # list[list[int]]，在知道 N 之后初始化

    model.eval()
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].float()   # (vocab_size,)

        if temperature != 1.0:
            logits = logits / temperature

        # ── 约束屏蔽 ──────────────────────────────────────────────────────────
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)  # True=屏蔽

        if phase == 'N_tok':
            # 只允许 N_START ~ N_START+MAX_NODES-1
            mask[:] = True
            mask[N_START: N_START + MAX_NODES] = False

        elif phase == 'parents':
            # 约束①：只允许 node_j，j < parent_count+1（即 j < k）
            # 约束②：禁止 SEP
            mask[:] = True
            if parent_count < N - 1:
                # 允许 NODE_START+0 ~ NODE_START+parent_count
                for j in range(parent_count + 1):
                    mask[NODE_START + j] = False
            # SEP 始终禁止（直到 parents 阶段结束后强制插入）

        elif phase == 'edges':
            # 允许节点 token（0~N-1）和 EOS_G
            mask[:] = True
            # 约束③：逐个检查哪些节点 token 不会产生三角
            # 补边是成对生成的，这里简单屏蔽会产生三角的组合
            for j in range(N):
                mask[NODE_START + j] = False   # 先全放开，再下面屏蔽
            # 约束④：若有节点度 < 2，屏蔽 EOS_G
            degs = degrees(run_adj)
            if all(d >= 2 for d in degs):
                mask[EOS_ID] = False           # 允许 EOS_G
            # 注：约束③在边的第一个 token 阶段全放开，
            #     精细屏蔽需要知道"当前边的第一个节点"，
            #     这里保守实现：屏蔽会与已有邻居产生三角的节点
            # （实际上对于第二个 token，还需要与第一个配对检查，略复杂，
            #   此处对第一个 token 不做三角预检，仅在第二个 token 加入时检查）

        # 应用屏蔽
        logits[mask] = NEG_INF

        # 采样
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())

        # ── 状态更新 ───────────────────────────────────────────────────────────
        if phase == 'N_tok':
            N = next_id - N_START + 1
            run_adj = [[0] * N for _ in range(N)]
            phase = 'parents' if N > 1 else 'sep_forced'
            parent_count = 0

        elif phase == 'parents':
            p = next_id - NODE_START
            parents.append(p)
            # 更新生成树边
            k = parent_count + 1
            run_adj[k][p] = run_adj[p][k] = 1
            parent_count += 1
            if parent_count == N - 1:
                # 约束②：强制插入 SEP
                generated.append(next_id)
                generated.append(SEP_ID)
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                phase = 'edges'
                continue

        elif phase == 'edges':
            if next_id == EOS_ID:
                generated.append(next_id)
                break
            # 加入边（成对）
            first_node = next_id - NODE_START
            # 生成第二个节点
            input_ids = torch.tensor(
                [generated + [next_id]], dtype=torch.long, device=device)
            out2 = model(input_ids=input_ids)
            logits2 = out2.logits[0, -1, :].float()
            if temperature != 1.0:
                logits2 = logits2 / temperature
            mask2 = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
            for j in range(N):
                if j != first_node and not run_adj[first_node][j]:
                    if not has_triangle(run_adj, first_node, j):
                        mask2[NODE_START + j] = False
            if mask2.all():  # 无合法第二节点，跳过
                generated.append(next_id)
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                continue
            logits2[mask2] = NEG_INF
            probs2 = torch.softmax(logits2, dim=-1)
            second_node_tok = int(torch.multinomial(probs2, 1).item())
            second_node = second_node_tok - NODE_START
            # 加边
            run_adj[first_node][second_node] = run_adj[second_node][first_node] = 1
            generated.append(next_id)
            generated.append(second_node_tok)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            continue

        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return generated


# ── 打印对比 ──────────────────────────────────────────────────────────────────

def print_comparison(gt_tokens: list[int], gen_tokens: list[int], sample_idx: int):
    gt  = parse_sequence(gt_tokens)
    gen = parse_sequence(gen_tokens)

    print(f'\n{"="*60}')
    print(f'样本 #{sample_idx}')
    print(f'{"="*60}')

    print(f'\n[真实] n_nodes={gt["n_nodes"]}  parents={gt["parents"]}')
    print(f'       extra_edges={gt["extra_edges"]}')
    if gt['adj']:
        degs = degrees(gt['adj'])
        print(f'       degrees={degs}')
        print(f'       邻接矩阵:\n{adj_to_str(gt["adj"])}')

    print(f'\n[生成] n_nodes={gen["n_nodes"]}  valid={gen["valid"]}  parents={gen["parents"]}')
    print(f'       extra_edges={gen["extra_edges"]}')
    if gen['adj']:
        degs = degrees(gen['adj'])
        print(f'       degrees={degs}  min_deg={min(degs)}')
        print(f'       邻接矩阵:\n{adj_to_str(gen["adj"])}')

    # 结构对比
    if gt['valid'] and gen['valid'] and gt['n_nodes'] == gen['n_nodes']:
        n = gt['n_nodes']
        gt_edges  = {(min(i,j), max(i,j)) for i in range(n) for j in range(n) if gt['adj'][i][j]}
        gen_edges = {(min(i,j), max(i,j)) for i in range(n) for j in range(n) if gen['adj'][i][j]}
        tp = len(gt_edges & gen_edges)
        fp = len(gen_edges - gt_edges)
        fn = len(gt_edges - gen_edges)
        print(f'\n[对比] N相同={gt["n_nodes"]}  边TP={tp}  FP={fp}  FN={fn}')
        if len(gt_edges) > 0:
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2*prec*rec / max(prec+rec, 1e-6)
            print(f'       边精确率={prec:.3f}  召回率={rec:.3f}  F1={f1:.3f}')
    elif gt['n_nodes'] != gen['n_nodes']:
        print(f'\n[对比] N不匹配: 真实={gt["n_nodes"]}  生成={gen["n_nodes"]}')


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True,
                   help='checkpoint 路径，e.g. checkpoints/llm_graph/stage1/best.pt')
    p.add_argument('--data',       default='data/processed/graph_tree/text_graph_tree.npz',
                   help='NPZ 数据集路径（用于对比）')
    p.add_argument('--vocab',      default='llm_graph/vocab/wp_tokenizer.json',
                   help='WordPiece tokenizer 路径')
    p.add_argument('--text',       default=None,
                   help='自定义文本描述（单条）')
    p.add_argument('--demo',       action='store_true',
                   help='生成 DEFAULT_PROMPTS 中三条内置提示词的结果')
    p.add_argument('--compare-n',  type=int, default=3,
                   help='从数据集随机取 N 条做真实/生成对比（默认3；0=跳过）')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--stage',      type=int, default=2, choices=[1, 2],
                   help='1=仅图序列（无文本前缀），2=text+图（默认）')
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    cfg = LlamaConfig(**MODEL_CFG)
    model = LlamaForCausalLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    step = ckpt.get('step', '?')
    loss = ckpt.get('best_loss', float('nan'))
    print(f'模型加载完毕  step={step}  best_loss={loss:.4f}')
    return model


def run_text_prompt(model, text: str, vocab_path: str, device, temperature: float, label: str = ''):
    """对单条文本生成图序列并打印结果（无 GT 对比）。"""
    if Path(vocab_path).exists():
        text_ids = encode_text(text, vocab_path)
        prefix   = text_ids + [BOS_ID]
    else:
        print('  [警告] 未找到 vocab，使用空文本前缀')
        prefix = [BOS_ID]

    gen_seq = generate(model, prefix, device, max_new_tokens=200, temperature=temperature)
    result  = parse_sequence(gen_seq)

    tag = f'  ({label})' if label else ''
    print(f'\n{"="*60}')
    print(f'[生成结果]{tag}')
    print(f'  文本: {text}')
    print(f'  n_nodes={result["n_nodes"]}  valid={result["valid"]}')
    print(f'  parents={result["parents"]}')
    print(f'  extra_edges={result["extra_edges"]}')
    if result['adj']:
        degs = degrees(result['adj'])
        print(f'  degrees={degs}  min_deg={min(degs) if degs else "-"}')
        print(f'  邻接矩阵:\n{adj_to_str(result["adj"])}')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args.ckpt, device)

    # ── ① 数据集真实/生成对比 ────────────────────────────────────────────────
    if args.compare_n > 0:
        d = np.load(args.data)
        all_tokens   = d['tokens'].astype(np.int32)
        all_lengths  = d['lengths'].astype(np.int32)
        all_textlens = d['text_lens'].astype(np.int32)
        n_total = len(all_tokens)
        print(f'数据集: {n_total} 条')

        rng = np.random.default_rng(args.seed)
        indices = rng.choice(n_total, size=args.compare_n, replace=False)

        for idx in indices:
            length   = int(all_lengths[idx])
            text_len = int(all_textlens[idx])
            gt_seq   = all_tokens[idx, :length].tolist()

            if args.stage == 2:
                prefix = gt_seq[:text_len]           # [...text tokens... BOS_G]
            else:
                bos_pos = text_len - 1
                prefix  = gt_seq[bos_pos: bos_pos + 1]

            gen_seq = generate(model, prefix, device,
                               max_new_tokens=200, temperature=args.temperature)
            print_comparison(gt_seq, gen_seq, idx)

    # ── ② 自定义文本（单条）────────────────────────────────────────────────
    if args.text is not None:
        run_text_prompt(model, args.text, args.vocab, device,
                        args.temperature, label='自定义')

    # ── ③ 内置默认提示词（--demo）───────────────────────────────────────────
    if args.demo:
        labels = ['简单 ~15节点', '中等 ~13节点', '复杂 ~27节点']
        print(f'\n\n{"#"*60}')
        print(f'# 默认演示提示词（DEFAULT_PROMPTS）')
        print(f'{"#"*60}')
        for label, prompt in zip(labels, DEFAULT_PROMPTS):
            run_text_prompt(model, prompt, args.vocab, device,
                            args.temperature, label=label)


if __name__ == '__main__':
    main()
