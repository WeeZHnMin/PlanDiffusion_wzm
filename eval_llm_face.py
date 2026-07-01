"""
eval_llm_face.py

给定文本描述 + 邻接图，提取环结构和环间邻接关系，让 MiMo 预测每个环的房间类型，
计算与 ground truth 的准确率。

用法:
    python eval_llm_face.py \
        --jsonl data/jsonl/val_graph_dataset_18k5.jsonl \
        --n_samples 200 \
        --output results/llm_face_eval.jsonl

API key 从环境变量 MIMO_API_KEY 读取，或通过 --api-key 传入。
"""

import argparse
import json
import os
import re
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from openai import OpenAI

# ── 常量 ──────────────────────────────────────────────────────────────────────

BASE_TYPES = {
    1: "bathroom",
    2: "bedroom",
    3: "living_room",
    4: "kitchen",
    5: "corridor",
    6: "dining_room",
    7: "other",
}
TYPE_NAMES   = set(BASE_TYPES.values())
COMBO_VOCAB  = Path("room_type_clf/type_combo_vocab_old.json")
MAX_ROOMS    = 20
MIMO_BASE    = "https://api.xiaomimimo.com/v1"
MIMO_MODEL   = "mimo-v2.5"
SYSTEM_MSG   = "You are MiMo, an AI assistant developed by Xiaomi."


# ── 环提取 ────────────────────────────────────────────────────────────────────

def find_rings(adj, n):
    """返回环列表，每个环是节点 frozenset。与 room_type_clf 逻辑一致。"""
    seen   = set()
    rings  = []

    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            prev  = {u: -1}
            q     = deque([u])
            found = False
            while q and not found:
                cur = q.popleft()
                for w in range(n):
                    if not adj[cur, w] or w in prev:
                        continue
                    if cur == u and w == v:
                        continue
                    prev[w] = cur
                    if w == v:
                        found = True
                        break
                    q.append(w)
            if not found:
                continue
            cycle = []
            cur   = v
            while cur != -1:
                cycle.append(cur)
                cur = prev[cur]
            key = frozenset(cycle)
            if key in seen:
                continue
            seen.add(key)
            rings.append(key)
            if len(rings) >= MAX_ROOMS:
                return rings

    return rings


def ring_adjacency(rings):
    """两个环共享 ≥2 个节点则视为相邻（共享一条边）。"""
    adj = {i: [] for i in range(len(rings))}
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                adj[i].append(j)
                adj[j].append(i)
    return adj


# ── Ground Truth ──────────────────────────────────────────────────────────────

def load_combo_vocab():
    with open(COMBO_VOCAB, encoding="utf-8") as f:
        v = json.load(f)
    # combo_id -> list of base type ints
    id_to_bases = {}
    for combo_str, cid in v["combo_to_id"].items():
        id_to_bases[cid] = json.loads(combo_str)
    return id_to_bases


def ring_gt_types(rings, node_combo_ids, id_to_bases):
    """每个环的 ground truth = 该环内节点最多票的 base type。"""
    gt = []
    for ring in rings:
        counts = Counter()
        for node in ring:
            if node < len(node_combo_ids):
                for bt in id_to_bases.get(int(node_combo_ids[node]), []):
                    counts[bt] += 1
        if counts:
            gt.append(BASE_TYPES.get(counts.most_common(1)[0][0], "other"))
        else:
            gt.append("other")
    return gt


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_prompt(text_desc, rings, ring_adj, n_nodes):
    lines = [
        "You are analyzing a floor plan represented as a planar graph.",
        "",
        f"Description: {text_desc}",
        "",
        f"The graph has {n_nodes} nodes (corners/junctions of walls).",
        "Each ring below is an enclosed face of the graph, i.e., one room.",
        "",
        "Rings (rooms):",
    ]
    for i, ring in enumerate(rings):
        nodes_str = ", ".join(str(nd) for nd in sorted(ring))
        lines.append(f"  Ring {i}: nodes [{nodes_str}]")

    lines += ["", "Ring adjacency (rooms sharing a wall):"]
    for i, adj_list in ring_adj.items():
        if adj_list:
            lines.append(f"  Ring {i} -> " + ", ".join(f"Ring {j}" for j in sorted(adj_list)))
        else:
            lines.append(f"  Ring {i} -> (no neighbors)")

    lines += [
        "",
        "Predict the room type for each ring.",
        "Allowed types: bathroom, bedroom, living_room, kitchen, corridor, dining_room, other",
        "",
        "Reply in exactly this format (one line per ring, no extra text):",
    ]
    for i in range(len(rings)):
        lines.append(f"Ring {i}: <room_type>")

    return "\n".join(lines)


# ── MiMo 调用 ─────────────────────────────────────────────────────────────────

def call_mimo(prompt, client, thinking=True, max_retries=3):
    extra = {} if thinking else {"extra_body": {"enable_thinking": False}}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                **extra,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# ── 解析响应 ──────────────────────────────────────────────────────────────────

def parse_response(response, n_rings):
    predicted = [None] * n_rings
    pattern   = re.compile(r'Ring\s+(\d+)\s*:\s*([a-z_]+)', re.IGNORECASE)
    for m in pattern.finditer(response):
        rid  = int(m.group(1))
        rtype = m.group(2).lower().strip()
        if rid < n_rings:
            predicted[rid] = rtype if rtype in TYPE_NAMES else "other"
    return predicted


# ── 主评测循环 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",      default="data/jsonl/val_graph_dataset_18k5.jsonl")
    p.add_argument("--n_samples",  type=int, default=200)
    p.add_argument("--output",     default="results/llm_face_eval.jsonl")
    p.add_argument("--api-key",    default="")
    p.add_argument("--sleep",       type=float, default=0.5, help="API 调用间隔（秒）")
    p.add_argument("--no_thinking", action="store_true",    help="关闭 MiMo 思考模式")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.getenv("MIMO_API_KEY", "")
    if not api_key:
        raise SystemExit("未提供 API key，请设置 MIMO_API_KEY 或使用 --api-key")

    client     = OpenAI(api_key=api_key, base_url=MIMO_BASE, timeout=60.0)
    id_to_bases = load_combo_vocab()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    total_rings = correct_rings = parsed_rings = 0
    n_done = n_skip = 0
    t0 = time.perf_counter()

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if n_done >= args.n_samples:
                break
            line = line.strip()
            if not line:
                continue

            rec   = json.loads(line)
            n     = min(int(rec["n_nodes"]), 40)
            adj   = np.array(rec["adj_matrix"], dtype=np.int32)[:n, :n]
            np.fill_diagonal(adj, 0)
            prompt_text   = rec.get("prompt", "").replace("\n", " ").strip()
            node_combo_ids = rec.get("node_combo_ids", [])

            rings = find_rings(adj, n)
            if not rings:
                n_skip += 1
                continue

            ring_adj = ring_adjacency(rings)
            gt       = ring_gt_types(rings, node_combo_ids, id_to_bases)
            prompt   = build_prompt(prompt_text, rings, ring_adj, n)

            try:
                response  = call_mimo(prompt, client, thinking=not args.no_thinking)
                if n_done == 0:
                    print("\n=== 第一条原始回答 ===")
                    print(response)
                    print("===================\n")
                predicted = parse_response(response, len(rings))
            except Exception as e:
                print(f"  [ERROR] sample {n_done}: {e}")
                n_skip += 1
                continue

            # 统计：漏答算错
            sample_correct = 0
            sample_total   = len(rings)
            total_rings   += sample_total
            for g, p in zip(gt, predicted):
                if p is not None:
                    parsed_rings += 1
                if g == p:           # p is None 时不等，自动算错
                    sample_correct += 1
                    correct_rings  += 1

            acc = sample_correct / max(sample_total, 1)
            n_done += 1

            rec_out = {
                "sample":    n_done,
                "n_rings":   len(rings),
                "gt":        gt,
                "predicted": predicted,
                "acc":       round(acc, 4),
                "response":  response[:500],
            }
            out_f.write(json.dumps(rec_out) + "\n")
            out_f.flush()

            if n_done % 10 == 0:
                overall = correct_rings / max(total_rings, 1)
                elapsed = time.perf_counter() - t0
                print(f"[{n_done:4d}/{args.n_samples}] "
                      f"acc={overall:.2%}  "
                      f"parsed={parsed_rings}/{total_rings}  "
                      f"{elapsed:.0f}s")

            time.sleep(args.sleep)

    out_f.close()
    overall = correct_rings / max(total_rings, 1)
    print(f"\n完成 {n_done} 条（跳过 {n_skip}）")
    print(f"总环数: {total_rings}  LLM 解析成功: {parsed_rings}  ({parsed_rings/max(total_rings,1):.1%})")
    print(f"准确率(漏答算错): {overall:.2%}  ({correct_rings}/{total_rings})")
    print(f"结果 -> {out_path}")


if __name__ == "__main__":
    main()
