"""
用 MiMo 多模态 LLM 从 K 次推理结果中选出最优 roll。

对每条测试数据：
  1. 读取 outputs/visualize_gacha/{idx:06d}.png（1行×K列并排渲染图）
  2. 连同文本描述发给 MiMo，让它选出最好的列（1~K）
  3. 输出完整 JSONL，包含 IoU 评估、可视化、FID 所需全部字段

输出 JSONL 字段（每条一行）：
  idx            : int       数据集索引
  n_nodes        : int       有效节点数
  text           : str       文本描述
  adj_matrix     : [[float]] n_nodes × n_nodes 邻接矩阵（已裁剪）
  node_mask      : [float]   长度 40（原始掩码，FID 可能用到）
  gt_coords      : [[float]] n_nodes × 2  GT 坐标
  gt_combo_ids   : [int]     n_nodes      GT 房间类型组合 ID
  chosen_k       : int       LLM 选中的 roll 索引（0-based）
  seed           : int       对应噪声种子 = idx + chosen_k * 1_000_000
  pred_coords    : [[float]] n_nodes × 2  选中 roll 的预测坐标
  pred_combo_ids : [int]     n_nodes      选中 roll 的预测类型 ID
  llm_response   : str       LLM 原始回复
  llm_model      : str       实际使用的模型名

Usage (from project root):
    python select_best_roll.py \\
        --npz     outputs/eval/infer_all_5roll.npz \\
        --data    data/jsonl/test_graph_dataset_10k.jsonl \\
        --img-dir outputs/visualize_gacha \\
        --out     outputs/eval/best_roll.jsonl \\
        --workers 30
"""

import argparse
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from openai import OpenAI

# ── 默认 API Keys（从 caption_floorplans_mimo.py 同步） ─────────────────────
DEFAULT_API_KEYS = [
    "sk-sncfd5uu7x3am5l1t5vusklxz0waktdsl45so1vf8s021jx4",
    "sk-shpbrd6utuqtjd1jjkmyalddacf8xveir4drm6gzqf8lamol",
    "sk-s3ef99lh5qxczbjqgzedhm0rblyl6oqtqj792jn68654hunh",
    "sk-sfhh1tzoejdslgxiw9cu1jghxzla2uhlm0wwdjewmf349xkf",
    "sk-s63csjy32yd8chzzxr9uh226oe21v91104bj400puj9d935n",
    "sk-sofnk8hxkrj7cfn2vyorztykkirwnwnlro7bj8pa0tan5tdb",
    "sk-s8kqiu5rd4smax8cwhn0rma315c3nm5xyzn8zzmm31diu29x",
]

BASE_URL      = "https://api.xiaomimimo.com/v1"
MODEL         = "mimo-v2.5"
SYSTEM_MSG    = "You are MiMo, an AI assistant developed by Xiaomi."

PROMPT_TMPL = """\
The image shows {K} AI-generated floor plans (columns 1 to {K} from left to right) \
for the same building layout. The target layout is described as:

"{text}"

Which column (integer from 1 to {K}) shows the floor plan that:
- best matches the room types and adjacency in the description
- looks most like a valid, coherent residential floor plan (all rooms connected, no obvious geometric errors)

Reply with a single integer only (e.g. 3). No explanation."""


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_done_indices(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "idx" in row:
                    done.add(int(row["idx"]))
            except Exception:
                continue
    return done


def parse_choice(text: str, K: int) -> Optional[int]:
    """从 LLM 回复中提取 1~K 的整数，返回 0-indexed，失败返回 None。"""
    # 先找第一个独立数字
    m = re.search(r'\b([1-9]\d*)\b', text.strip())
    if m:
        v = int(m.group(1))
        if 1 <= v <= K:
            return v - 1  # 转为 0-indexed
    return None


def call_mimo(api_key: str, img_b64: str, prompt: str,
              timeout: float, model: str) -> Dict:
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            temperature=0.0,
            max_tokens=32,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return {"ok": False, "text": "", "error": "empty_content", "model": model}
        return {"ok": True, "text": text, "error": "", "model": model}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e), "model": model}


def is_transient(error: str) -> bool:
    low = error.lower()
    return ("429" in error or "timed out" in low or "timeout" in low
            or "rate limit" in low or "connection" in low
            or error == "empty_content")


def is_quota(error: str) -> bool:
    low = error.lower()
    return ("quota" in low or "balance" in low or "insufficient" in low
            or "credit" in low or "billing" in low)


# ── 参数 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",      default="outputs/eval/infer_all_5roll.npz")
    p.add_argument("--data",     default="data/jsonl/test_graph_dataset_10k.jsonl")
    p.add_argument("--img-dir",  type=Path, default=Path("outputs/visualize_gacha"))
    p.add_argument("--out",      type=Path, default=Path("outputs/eval/best_roll.jsonl"))
    p.add_argument("--api-keys", nargs="+", default=DEFAULT_API_KEYS)
    p.add_argument("--model",    default=MODEL)
    p.add_argument("--workers",  type=int, default=30)
    p.add_argument("--timeout",  type=float, default=20.0)
    p.add_argument("--n",        type=int, default=0, help="处理前N条，0=全部")
    p.add_argument("--start",    type=int, default=0, help="从第几条样本开始")
    p.add_argument("--max-key-failures", type=int, default=5)
    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── 读 NPZ ───────────────────────────────────────────────────────────────
    print(f"读取 NPZ: {args.npz}")
    data           = np.load(args.npz, mmap_mode="r")
    pred_coords_all    = data["pred_coords"]     # [N, K, 40, 2]
    pred_combo_all     = data["pred_combo_ids"]  # [N, K, 40]
    gt_coords_all      = data["gt_coords"]        # [N, 40, 2]
    gt_combo_all       = data["gt_combo_ids"]     # [N, 40]
    gt_adj_all         = data["gt_adj"]           # [N, 40, 40]
    gt_mask_all        = data["gt_mask"]          # [N, 40]
    gt_n_nodes_all     = data["gt_n_nodes"]       # [N]
    K = int(data["rolls"]) if "rolls" in data else pred_coords_all.shape[1]
    N_total = len(gt_n_nodes_all)

    # ── 读文本描述 ────────────────────────────────────────────────────────────
    print(f"读取数据集: {args.data}")
    with open(args.data, encoding="utf-8") as f:
        all_lines = f.readlines()

    # ── 确定要处理的索引范围 ──────────────────────────────────────────────────
    end = min(args.start + args.n, N_total) if args.n > 0 else N_total
    all_indices = list(range(args.start, end))

    done = load_done_indices(args.out)
    todo = [i for i in all_indices if i not in done]
    print(f"共 {len(all_indices)} 条，已完成 {len(done)}，待处理 {len(todo)} 条，K={K}")
    if not todo:
        print("全部已完成。")
        return

    # ── 每条 sample 封装为任务 ────────────────────────────────────────────────
    tasks = []
    for i in todo:
        png = args.img_dir / f"{i:06d}.png"
        if not png.exists():
            print(f"  [WARN] 图片不存在，跳过: {png}")
            continue
        text = json.loads(all_lines[i]).get("prompt", "")
        tasks.append({"idx": i, "png": png, "text": text})

    print(f"有效任务: {len(tasks)}，workers={args.workers}")

    # ── 共享状态 ──────────────────────────────────────────────────────────────
    active_keys   = list(args.api_keys)
    dropped_keys: Dict[str, str] = {}
    key_fail_cnt: Dict[str, int] = {k: 0 for k in active_keys}
    key_idx       = {"i": 0}
    lock          = threading.Lock()
    write_lock    = threading.Lock()
    counters      = {"ok": 0, "err": 0, "fallback": 0, "finished": 0}

    def pick_key(exclude: Set[str]) -> Optional[str]:
        with lock:
            alive = [k for k in active_keys if k not in dropped_keys and k not in exclude]
            if not alive:
                return None
            k = alive[key_idx["i"] % len(alive)]
            key_idx["i"] += 1
            return k

    def process(task: Dict) -> Optional[Dict]:
        i       = task["idx"]
        png     = task["png"]
        text    = task["text"]
        n       = int(gt_n_nodes_all[i])
        img_b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        prompt  = PROMPT_TMPL.format(K=K, text=text)

        tried: Set[str] = set()
        llm_text    = ""
        llm_model   = ""
        chosen_k    = None
        parse_fails = 0
        MAX_PARSE_RETRIES = 3

        while chosen_k is None:
            key = pick_key(tried)
            if key is None:
                break
            tried.add(key)
            res = call_mimo(key, img_b64, prompt, args.timeout, args.model)
            llm_text  = res["text"]
            llm_model = res["model"]

            if res["ok"]:
                chosen_k = parse_choice(llm_text, K)
                if chosen_k is None:
                    parse_fails += 1
                    print(f"  [PARSE-FAIL {parse_fails}/{MAX_PARSE_RETRIES}] "
                          f"idx={i} resp={repr(llm_text[:80])}")
                    if parse_fails >= MAX_PARSE_RETRIES:
                        break  # 保留 llm_text，fallback k=0
                    tried.discard(key)  # 允许重试（可换 key 也可复用）
                    continue
                with lock:
                    key_fail_cnt[key] = 0
                break
            else:
                err = res["error"]
                if is_quota(err):
                    with lock:
                        key_fail_cnt[key] = key_fail_cnt.get(key, 0) + 1
                        if key_fail_cnt[key] >= args.max_key_failures and key not in dropped_keys:
                            dropped_keys[key] = err
                            print(f"  [DROP-KEY] ...{key[-6:]} quota exhausted")
                elif is_transient(err):
                    pass  # 换 key 重试
                else:
                    with lock:
                        if key not in dropped_keys:
                            dropped_keys[key] = err
                            print(f"  [DROP-KEY] ...{key[-6:]} fatal: {err[:80]}")

        if chosen_k is None:
            chosen_k = 0   # fallback: 选第一个
            with lock:
                counters["fallback"] += 1

        # ── 组装输出行 ────────────────────────────────────────────────────────
        adj_n = gt_adj_all[i, :n, :n]
        return {
            "idx":            i,
            "n_nodes":        n,
            "text":           text,
            "adj_matrix":     adj_n.tolist(),
            "node_mask":      gt_mask_all[i].tolist(),           # [40]
            "gt_coords":      gt_coords_all[i, :n].tolist(),    # [n, 2]
            "gt_combo_ids":   gt_combo_all[i, :n].tolist(),     # [n]
            "chosen_k":       int(chosen_k),
            "seed":           i + chosen_k * 1_000_000,
            "pred_coords":    pred_coords_all[i, chosen_k, :n].tolist(),  # [n, 2]
            "pred_combo_ids": pred_combo_all[i, chosen_k, :n].tolist(),   # [n]
            "llm_response":   llm_text,
            "llm_model":      llm_model,
        }

    # ── 并发执行 ──────────────────────────────────────────────────────────────
    in_flight: Dict = {}
    task_iter = iter(tasks)
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # 预填充
        for _ in range(min(args.workers, total)):
            try:
                t = next(task_iter)
                in_flight[pool.submit(process, t)] = t
            except StopIteration:
                break

        while in_flight:
            done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                t = in_flight.pop(fut)
                try:
                    row = fut.result()
                except Exception as e:
                    row = None
                    print(f"  [WORKER-ERR] idx={t['idx']} {e}")

                with lock:
                    counters["finished"] += 1
                    finished = counters["finished"]
                    if row is not None:
                        counters["ok"] += 1
                    else:
                        counters["err"] += 1

                if row is not None:
                    with write_lock:
                        with args.out.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if finished % 100 == 0 or finished == total:
                    with lock:
                        fb = counters["fallback"]
                    print(f"  [{finished}/{total}] ok={counters['ok']} "
                          f"err={counters['err']} fallback(k=0)={fb}",
                          flush=True)

                # 补充新任务
                try:
                    nt = next(task_iter)
                    in_flight[pool.submit(process, nt)] = nt
                except StopIteration:
                    pass

    print(f"\n完成。ok={counters['ok']} err={counters['err']} "
          f"fallback={counters['fallback']}")
    print(f"输出 → {args.out}")


if __name__ == "__main__":
    main()
