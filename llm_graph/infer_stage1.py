"""
Stage2 推理脚本：加载模型，对给定文本自回归生成图序列，
应用四个解码约束，解析邻接矩阵，并与真实数据对比。

用法：
  # 随机取3条真实样本，用其文本条件生成并对比
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage2/best.pt

  # 输入自定义文本，自动搜索数据集中最相近的真实样本做对比
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage2/best.pt \
      --text "the living room is adjacent to the kitchen and two bedrooms ."

  # 内置3条默认提示词（来自真实数据），直接与对应真实样本对比
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage2/best.pt --demo

  # 只跑默认演示，不做随机对比
  python -m llm_graph.infer_stage1 --ckpt checkpoints/llm_graph/stage2/best.pt --demo --compare-n 0
"""

import argparse
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

# ── 三条默认演示提示词（直接取自真实数据，存有对应 dataset idx）────────────────
# (dataset_idx, label, text)
DEFAULT_PROMPTS = [
    (
        256787,
        '简单 ~15节点 | 走廊居中，4空间',
        "the corridor is located between the bedroom and the kitchen ; "
        "the bathroom is adjacent to the bedroom ; "
        "the living room is adjacent to both the bedroom and the kitchen .",
    ),
    (
        146316,
        '中等 ~13节点 | 方位描述，5空间+阳台',
        "the living room is at the center of the house , connecting to the kitchen in the north , "
        "leading to the bedroom in the south , adjacent to the bathroom in the east , "
        "and accessing the balcony in the west .",
    ),
    (
        107473,
        '复杂 ~27节点 | 2卧2卫+走廊',
        "the kitchen is adjacent to the living room , which connects via a corridor to two bedrooms "
        "and two bathrooms ; one bedroom is adjacent to the kitchen , the other bedroom is adjacent "
        "to one of the bathrooms , and the two bathrooms are on opposite sides of the corridor .",
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

def parse_sequence(tokens: list) -> dict:
    """解析完整序列（含文本前缀），返回 n_nodes/parents/extra_edges/adj/valid。"""
    result = dict(n_nodes=0, parents=[], extra_edges=[], adj=None, valid=False)
    try:
        i = 0
        while i < len(tokens) and tokens[i] != BOS_ID:
            i += 1
        if i >= len(tokens):
            return result
        i += 1  # 跳过 BOS_G

        if i >= len(tokens) or not (N_START <= tokens[i] < N_START + MAX_NODES):
            return result
        N = tokens[i] - N_START + 1
        result['n_nodes'] = N
        i += 1

        parents = []
        for _ in range(N - 1):
            if i >= len(tokens) or not (NODE_START <= tokens[i] < NODE_START + N):
                return result
            parents.append(tokens[i] - NODE_START)
            i += 1
        result['parents'] = parents

        if i >= len(tokens) or tokens[i] != SEP_ID:
            return result
        i += 1

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

        adj = [[0] * N for _ in range(N)]
        for k, p in enumerate(parents):
            adj[k + 1][p] = adj[p][k + 1] = 1
        for u, v in extra_edges:
            if 0 <= u < N and 0 <= v < N and u != v:
                adj[u][v] = adj[v][u] = 1
        result['adj'] = adj
        result['valid'] = True
    except Exception:
        pass
    return result


def adj_to_str(adj: list) -> str:
    n = len(adj)
    lines = []
    for i in range(n):
        row = ''.join(str(adj[i][j]) for j in range(n))
        lines.append(f'  [{i:2d}] {row}')
    return '\n'.join(lines)


def node_degrees(adj: list) -> list:
    return [sum(row) for row in adj]


# ── 文本编码 / 最近邻搜索 ──────────────────────────────────────────────────────

def encode_text(text: str, tokenizer_path: str, max_len: int = 128) -> list:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    return tok.encode(text).ids[:max_len]


def find_nearest_sample(
    query_ids: list,
    all_tokens: np.ndarray,
    all_textlens: np.ndarray,
    search_n: int = 50000,
    seed: int = 0,
) -> int:
    """
    在数据集中找文本 token 集合 Jaccard 相似度最高的样本，
    对大数据集只搜 search_n 个随机子集以保证速度。
    """
    n_total = len(all_tokens)
    rng = np.random.default_rng(seed)
    cands = rng.choice(n_total, size=min(search_n, n_total), replace=False)

    query_set = set(query_ids)
    best_idx, best_score = int(cands[0]), -1.0

    for idx in cands:
        tl = int(all_textlens[idx]) - 1   # 不含 BOS_G
        if tl <= 0:
            continue
        sample_set = set(all_tokens[idx, :tl].tolist())
        inter = len(query_set & sample_set)
        union = len(query_set | sample_set)
        score = inter / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_idx = int(idx)

    print(f'  [最近邻] idx={best_idx}  Jaccard={best_score:.3f}')
    return best_idx


# ── 约束解码 ──────────────────────────────────────────────────────────────────

def has_triangle(adj: list, u: int, v: int) -> bool:
    n = len(adj)
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return True
    return False


@torch.no_grad()
def generate(
    model,
    prefix_ids: list,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    use_c1: bool = True,   # 约束①：因果树 p_k < k
    use_c2: bool = True,   # 约束②：SEP 时机控制
    use_c3: bool = True,   # 约束③：禁止三角环
    use_c4: bool = True,   # 约束④：度数下限 >= 2
) -> list:
    """
    从 prefix_ids（文本 tokens + BOS_G）后开始自回归生成。
    use_c1~c4 控制各约束是否启用，默认全开。
    """
    NEG_INF = float('-inf')
    generated = list(prefix_ids)
    phase = 'N_tok'
    N = 0
    parent_count = 0
    run_adj = None

    model.eval()
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids).logits[0, -1, :].float()
        if temperature != 1.0:
            logits = logits / temperature

        mask = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)  # True=禁止

        if phase == 'N_tok':
            mask[N_START: N_START + MAX_NODES] = False

        elif phase == 'parents':
            if use_c1:
                # 约束①：只允许 node_0 ~ node_{k-1}
                for j in range(parent_count + 1):
                    mask[NODE_START + j] = False
            else:
                # 无约束①：允许所有节点 token（仍排除 SEP/EOS）
                for j in range(N):
                    mask[NODE_START + j] = False

        elif phase == 'edges':
            for j in range(N):
                mask[NODE_START + j] = False
            # 约束④
            if use_c4 and all(d >= 2 for d in node_degrees(run_adj)):
                mask[EOS_ID] = False
            elif not use_c4:
                mask[EOS_ID] = False

        logits[mask] = NEG_INF
        next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

        if phase == 'N_tok':
            N = next_id - N_START + 1
            run_adj = [[0] * N for _ in range(N)]
            phase = 'parents' if N > 1 else 'edges'
            parent_count = 0
            generated.append(next_id)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        elif phase == 'parents':
            if NODE_START <= next_id < NODE_START + N:
                p = next_id - NODE_START
                k = parent_count + 1
                if 0 <= k < N:
                    run_adj[k][p] = run_adj[p][k] = 1
                parent_count += 1
            generated.append(next_id)
            if use_c2 and parent_count == N - 1:
                # 约束②：强制 SEP
                generated.append(SEP_ID)
                phase = 'edges'
            elif next_id == SEP_ID:
                phase = 'edges'
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        elif phase == 'edges':
            if next_id == EOS_ID:
                generated.append(next_id)
                break
            first = next_id - NODE_START
            tmp_ids = torch.tensor([generated + [next_id]], dtype=torch.long, device=device)
            logits2 = model(input_ids=tmp_ids).logits[0, -1, :].float()
            if temperature != 1.0:
                logits2 = logits2 / temperature
            mask2 = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
            for j in range(N):
                skip_triangle = use_c3 and has_triangle(run_adj, first, j)
                if j != first and not run_adj[first][j] and not skip_triangle:
                    mask2[NODE_START + j] = False
            if not mask2.all():
                logits2[mask2] = NEG_INF
                sec_tok = int(torch.multinomial(torch.softmax(logits2, dim=-1), 1).item())
                sec = sec_tok - NODE_START
                if 0 <= sec < N:
                    run_adj[first][sec] = run_adj[sec][first] = 1
                generated.extend([next_id, sec_tok])
            else:
                generated.append(next_id)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return generated


# ── 打印对比 ──────────────────────────────────────────────────────────────────

def print_comparison(gt_seq: list, gen_seq: list, sample_idx: int, text: str = ''):
    gt  = parse_sequence(gt_seq)
    gen = parse_sequence(gen_seq)

    print(f'\n{"="*64}')
    print(f'样本 #{sample_idx}' + (f'  |  {text[:80]}' if text else ''))
    print(f'{"="*64}')

    # 真实
    print(f'\n[真实]  n_nodes={gt["n_nodes"]}')
    print(f'  parents     = {gt["parents"]}')
    print(f'  extra_edges = {gt["extra_edges"]}')
    if gt['adj']:
        print(f'  degrees     = {node_degrees(gt["adj"])}')
        print(f'  邻接矩阵:\n{adj_to_str(gt["adj"])}')

    # 生成
    print(f'\n[生成]  n_nodes={gen["n_nodes"]}  valid={gen["valid"]}')
    print(f'  parents     = {gen["parents"]}')
    print(f'  extra_edges = {gen["extra_edges"]}')
    if gen['adj']:
        degs = node_degrees(gen['adj'])
        print(f'  degrees     = {degs}  min={min(degs) if degs else "-"}')
        print(f'  邻接矩阵:\n{adj_to_str(gen["adj"])}')

    # 边集对比
    if gt['valid'] and gen['valid']:
        n_gt  = gt['n_nodes']
        n_gen = gen['n_nodes']
        if n_gt == n_gen:
            n = n_gt
            gt_e  = {(min(i,j), max(i,j)) for i in range(n) for j in range(n) if gt['adj'][i][j]}
            gen_e = {(min(i,j), max(i,j)) for i in range(n) for j in range(n) if gen['adj'][i][j]}
            tp = len(gt_e & gen_e)
            fp = len(gen_e - gt_e)
            fn = len(gt_e - gen_e)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2*prec*rec / max(prec + rec, 1e-9)
            print(f'\n[对比]  N匹配={n}  TP={tp}  FP={fp}  FN={fn}'
                  f'  精确率={prec:.3f}  召回率={rec:.3f}  F1={f1:.3f}')
        else:
            print(f'\n[对比]  N不匹配: 真实={n_gt}  生成={n_gen}')
    else:
        print(f'\n[对比]  生成序列无效，跳过')


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        default='checkpoints/llm_graph/stage2/20260603_045254/best.pt',
                   help='checkpoint 路径（默认 stage2）')
    p.add_argument('--data',        default='data/processed/graph_tree/text_graph_tree.npz')
    p.add_argument('--vocab',       default='llm_graph/vocab/wp_tokenizer.json')
    p.add_argument('--text',        default=None,
                   help='自定义文本；自动在数据集里找最近邻真实样本做对比')
    p.add_argument('--demo',        action='store_true',
                   help='运行内置3条默认提示词并与对应真实样本对比')
    p.add_argument('--compare-n',   type=int, default=3,
                   help='随机取 N 条数据集样本做真实/生成对比（默认3；0=跳过）')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

    # 从 checkpoint 实际的 embedding 权重推断 vocab_size，避免硬编码不匹配
    actual_vocab = sd['model.embed_tokens.weight'].shape[0]
    cfg_dict = {**MODEL_CFG, 'vocab_size': actual_vocab}

    cfg = LlamaConfig(**cfg_dict)
    model = LlamaForCausalLM(cfg).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f'模型加载完毕  ckpt={ckpt_path}')
    print(f'  vocab_size={actual_vocab}  step={ckpt.get("step","?")}  best_loss={ckpt.get("best_loss", float("nan")):.4f}')
    return model


def load_dataset(data_path: str):
    d = np.load(data_path)
    tokens   = d['tokens'].astype(np.int32)
    lengths  = d['lengths'].astype(np.int32)
    textlens = d['text_lens'].astype(np.int32)
    print(f'数据集: {len(tokens)} 条  path={data_path}')
    return tokens, lengths, textlens


def get_prefix_and_gt(idx, all_tokens, all_lengths, all_textlens):
    """从数据集取第 idx 条样本，返回 (prefix, gt_seq)。"""
    length   = int(all_lengths[idx])
    text_len = int(all_textlens[idx])
    gt_seq   = all_tokens[idx, :length].tolist()
    prefix   = gt_seq[:text_len]   # [...text tokens... BOS_G]，stage2 直接用
    return prefix, gt_seq


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}\n')

    model = load_model(args.ckpt, device)

    all_tokens, all_lengths, all_textlens = load_dataset(args.data)

    # ── ① 随机数据集对比 ──────────────────────────────────────────────────────
    if args.compare_n > 0:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(all_tokens), size=args.compare_n, replace=False)
        print(f'\n{"#"*64}')
        print(f'# 随机数据集对比（{args.compare_n} 条）')
        print(f'{"#"*64}')
        for idx in indices:
            prefix, gt_seq = get_prefix_and_gt(idx, all_tokens, all_lengths, all_textlens)
            gen_seq = generate(model, prefix, device,
                               max_new_tokens=200, temperature=args.temperature)
            print_comparison(gt_seq, gen_seq, idx)

    # ── ② 自定义文本：搜最近邻真实样本对比 ──────────────────────────────────
    if args.text is not None:
        print(f'\n{"#"*64}')
        print(f'# 自定义文本推理')
        print(f'{"#"*64}')
        print(f'  文本: {args.text}')
        query_ids = encode_text(args.text, args.vocab)
        prefix    = query_ids + [BOS_ID]
        # 找最近邻真实样本
        near_idx  = find_nearest_sample(query_ids, all_tokens, all_textlens,
                                        seed=args.seed)
        _, gt_seq = get_prefix_and_gt(near_idx, all_tokens, all_lengths, all_textlens)
        gen_seq   = generate(model, prefix, device,
                             max_new_tokens=200, temperature=args.temperature)
        print_comparison(gt_seq, gen_seq, near_idx, text=args.text)

    # ── ③ 内置默认提示词，用已知 idx 直接拿真实样本对比 ─────────────────────
    if args.demo:
        print(f'\n{"#"*64}')
        print(f'# 默认演示提示词（DEFAULT_PROMPTS）')
        print(f'{"#"*64}')
        for gt_idx, label, prompt in DEFAULT_PROMPTS:
            print(f'\n  >> {label}')
            query_ids = encode_text(prompt, args.vocab)
            prefix    = query_ids + [BOS_ID]
            _, gt_seq = get_prefix_and_gt(gt_idx, all_tokens, all_lengths, all_textlens)
            gen_seq   = generate(model, prefix, device,
                                 max_new_tokens=200, temperature=args.temperature)
            print_comparison(gt_seq, gen_seq, gt_idx, text=prompt)


if __name__ == '__main__':
    main()
