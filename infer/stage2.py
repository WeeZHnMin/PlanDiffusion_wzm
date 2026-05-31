"""
Stage2 LLaMA 约束解码测试：文本提示词 → 图结构 token 序列。

用法（本地）:
    python -m infer.stage2

用法（Kaggle）:
    python -m infer.stage2 --ckpt /kaggle/input/... --prompts "客厅在中央" "两间卧室在左侧"
"""

import argparse
import ast
import json
import os
import sys

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/stage2_text_graph/best.pt")
    p.add_argument("--vocab",      default="node_diffusion/unified_vocab/vocab_config.json")
    p.add_argument("--bpe",        default="node_diffusion/unified_vocab/bpe_tokenizer.json")
    p.add_argument("--combo-vocab",default="data/processed/type_combo_vocab_old.json")
    p.add_argument("--prompts",    nargs="+", default=[
        "客厅位于中央，连接卧室、厨房和浴室。",
        "两间卧室在左侧，浴室居中，厨房在右侧与走廊相连。",
    ])
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def build_combo_maps(combo_vocab_path):
    cvocab = json.loads(open(combo_vocab_path, encoding="utf-8").read())
    BASE_SHORT = {1:"Ba", 2:"Bd", 3:"Lv", 4:"Ki", 5:"Co", 6:"Di", 7:"Ot"}
    combo_bases = {}
    combo_label = {}
    for key_str, cid in cvocab["combo_to_id"].items():
        bases = ast.literal_eval(key_str)
        combo_bases[cid] = bases
        combo_label[cid] = "+".join(BASE_SHORT[b] for b in bases)
    return combo_bases, combo_label


def load_stage2(ckpt_path, vocab_cfg, device):
    VOCAB_SIZE = vocab_cfg["total_vocab_size"]
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
        intermediate_size=1536, max_position_embeddings=384,
        bos_token_id=vocab_cfg["BOS_ID"],
        eos_token_id=vocab_cfg["EOS_ID"],
        pad_token_id=vocab_cfg["PAD_ID"],
        rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(cfg).to(device)
    raw = torch.load(ckpt_path, map_location=device)
    sd  = raw if "model" not in raw else raw["model"]
    sd  = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Stage2 loaded from:", ckpt_path)
    return model


def make_allowed_mask(vocab_size, state, seen_nodes, n_assigned, cur_node,
                      BOS_G, EOS_G, TOK_OPEN, TOK_CLOSE, TOK_BREAK,
                      TYPE_START, NODE_START, MAX_NODES):
    mask = torch.full((vocab_size,), float("-inf"))
    if state == "start":
        if n_assigned < MAX_NODES:
            mask[NODE_START + n_assigned] = 0.0
        mask[EOS_G] = 0.0
    elif state == "after_close":
        if n_assigned < MAX_NODES:
            mask[NODE_START + n_assigned] = 0.0
        mask[TOK_BREAK] = 0.0
        mask[EOS_G] = 0.0
    elif state == "after_node":
        mask[TYPE_START: TYPE_START + 32] = 0.0
    elif state == "after_type":
        mask[TOK_OPEN] = 0.0
    elif state == "in_nbrs":
        for nid in seen_nodes:
            if nid != cur_node:
                mask[NODE_START + nid] = 0.0
        mask[TOK_CLOSE] = 0.0
    return mask


def constrained_generate(prompt_text, model, bpe_tok, vocab_cfg, device,
                          max_new_tokens=256):
    BOS_G      = vocab_cfg["BOS_ID"]
    EOS_G      = vocab_cfg["EOS_ID"]
    TOK_OPEN   = vocab_cfg["TOK_OPEN"]
    TOK_CLOSE  = vocab_cfg["TOK_CLOSE"]
    TOK_BREAK  = vocab_cfg["TOK_BREAK"]
    TYPE_START = vocab_cfg["TYPE_START"]
    NODE_START = vocab_cfg["NODE_START"]
    MAX_NODES  = vocab_cfg["MAX_NODES"]
    VOCAB_SIZE = vocab_cfg["total_vocab_size"]

    text_ids  = bpe_tok.encode(prompt_text).ids[:128]
    input_ids = torch.tensor([text_ids + [BOS_G]], dtype=torch.long, device=device)

    state      = "start"
    seen_nodes = set()
    n_assigned = 0
    cur_node   = None
    adj        = {}
    node_types = {}
    generated  = []

    with torch.no_grad():
        past = None
        for _ in range(max_new_tokens):
            out    = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past   = out.past_key_values
            logits = out.logits[0, -1]
            logits = logits + make_allowed_mask(
                VOCAB_SIZE, state, seen_nodes, n_assigned, cur_node,
                BOS_G, EOS_G, TOK_OPEN, TOK_CLOSE, TOK_BREAK,
                TYPE_START, NODE_START, MAX_NODES,
            ).to(device)
            tok = int(logits.argmax())
            generated.append(tok)
            input_ids = torch.tensor([[tok]], dtype=torch.long, device=device)

            if tok == EOS_G:
                break

            if state in ("start", "after_close") and NODE_START <= tok < NODE_START + MAX_NODES:
                cur_node   = n_assigned
                seen_nodes.add(cur_node)
                n_assigned += 1
                adj[cur_node] = set()
                state = "after_node"
            elif state == "after_node" and TYPE_START <= tok < TYPE_START + 32:
                node_types[cur_node] = tok - TYPE_START + 1
                state = "after_type"
            elif state == "after_type" and tok == TOK_OPEN:
                state = "in_nbrs"
            elif state == "in_nbrs":
                if NODE_START <= tok < NODE_START + MAX_NODES:
                    nbr = tok - NODE_START
                    if nbr != cur_node:
                        adj[cur_node].add(nbr)
                        adj.setdefault(nbr, set()).add(cur_node)
                elif tok == TOK_CLOSE:
                    state = "after_close"
            elif state == "after_close" and tok == TOK_BREAK:
                state = "start"

    return n_assigned, adj, node_types, generated


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    vocab_cfg        = json.loads(open(args.vocab, encoding="utf-8").read())
    bpe_tok          = Tokenizer.from_file(args.bpe)
    combo_bases, combo_label = build_combo_maps(args.combo_vocab)
    model            = load_stage2(args.ckpt, vocab_cfg, device)

    for prompt in args.prompts:
        print("\n" + "=" * 60)
        print("提示词:", prompt)
        n, adj, ntypes, toks = constrained_generate(
            prompt, model, bpe_tok, vocab_cfg, device, args.max_new_tokens)
        print(f"节点数: {n}  token数: {len(toks)}")
        for i in range(n):
            cid   = ntypes.get(i, 7)
            label = combo_label.get(cid, str(cid))
            print(f"  节点{i:2d} [{label:12s}] 邻居: {sorted(adj.get(i, set()))}")
        isolated = [i for i in range(n) if not adj.get(i)]
        print("孤立节点:", isolated if isolated else "无")


if __name__ == "__main__":
    main()
