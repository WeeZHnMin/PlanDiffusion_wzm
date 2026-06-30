"""
label_rooms_mimo.py

对 node_diffusion 输出的 JSONL，找出每条预测图的所有环（房间面），
发给 MiMo 让其判断每个环的房间类型，解析结果写回 JSONL。

房间类型：
  1=bathroom  2=bedroom  3=living_room  4=kitchen  5=corridor

用法：
  python label_rooms_mimo.py \
      --jsonl  outputs/tri_ddim200_full.jsonl \
      --imgs   outputs/tri_ddim200_imgs \
      --out    outputs/tri_ddim200_labeled.jsonl
"""

import argparse
import base64
import json
import math
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

ROOM_TYPES = {
    "1": "bathroom",
    "2": "bedroom",
    "3": "living_room",
    "4": "kitchen",
    "5": "corridor",
}
VALID_TYPES = set(ROOM_TYPES.values())

try:
    from api_keys import MIMO_API_KEYS as _DEFAULT_KEYS
except ImportError:
    _DEFAULT_KEYS = []

BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL    = "mimo-v2.5"
SYSTEM_MESSAGE = "You are MiMo, an AI assistant developed by Xiaomi."

# ── 找环（面）────────────────────────────────────────────────────────────────

def _build_sorted_neighbors(coords, adj, n):
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j] == 1:
                nbrs[i].append(j)
    for i in range(n):
        nbrs[i] = sorted(
            nbrs[i],
            key=lambda w: math.atan2(coords[w][1] - coords[i][1],
                                     coords[w][0] - coords[i][0]),
        )
    return nbrs


def _next_half_edge(u, v, sorted_nbrs):
    nbrs = sorted_nbrs[v]
    if not nbrs:
        return None
    idx = nbrs.index(u)
    return nbrs[(idx - 1) % len(nbrs)]


def _signed_area(face, coords):
    pts = [coords[i] for i in face]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def find_faces(coords, adj):
    n = len(coords)
    sorted_nbrs = _build_sorted_neighbors(coords, adj, n)
    visited = set(); faces = []
    for u in range(n):
        for v in sorted_nbrs[u]:
            if (u, v) in visited:
                continue
            face = []; cu, cv = u, v; steps = 0
            while (cu, cv) not in visited and steps < n * n:
                visited.add((cu, cv)); face.append(cu)
                nw = _next_half_edge(cu, cv, sorted_nbrs)
                if nw is None:
                    break
                cu, cv = cv, nw; steps += 1
            if len(face) >= 3:
                faces.append(face)
    if not faces:
        return []
    abs_areas = [abs(_signed_area(f, coords)) for f in faces]
    outer_idx  = abs_areas.index(max(abs_areas))
    return [f for i, f in enumerate(faces) if i != outer_idx]


# ── MiMo 调用 ────────────────────────────────────────────────────────────────

def build_prompt(faces: List[List[int]], text_prompt: str) -> str:
    face_lines = "\n".join(
        f"  Room {i}: nodes {faces[i]}" for i in range(len(faces))
    )
    type_lines = "\n".join(f"  {k}: {v}" for k, v in ROOM_TYPES.items())
    return f"""This is a floor plan graph. Nodes are room corners; edges are walls.
The following enclosed regions are the rooms:
{face_lines}

Floor plan description: {text_prompt}

Classify each room into exactly one of these types:
{type_lines}

Reply with ONLY a JSON object, keys are room indices (as strings), values are type names.
Example: {{"0": "bedroom", "1": "kitchen", "2": "bathroom"}}"""


def call_mimo(api_key: str, img_b64: str, prompt: str, timeout: float,
              thinking: bool = False) -> Dict:
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            temperature=0.0,
            extra_body={"thinking": {"type": "enabled"} if thinking else {"type": "disabled"}},
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return {"ok": False, "error": "empty_content", "raw": ""}
        return {"ok": True, "raw": text, "error": ""}
    except Exception as e:
        return {"ok": False, "raw": "", "error": str(e)}


def parse_response(raw: str, n_faces: int) -> Optional[Dict[int, str]]:
    """从 MiMo 回答里提取 JSON，校验每个 face 都有合法类型。"""
    # 提取第一个 {...} 块
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except Exception:
        return None
    result = {}
    for i in range(n_faces):
        val = obj.get(str(i))
        if val is None:
            return None
        val = val.strip().lower().replace(" ", "_")
        if val not in VALID_TYPES:
            # 尝试数字键映射
            for k, v in ROOM_TYPES.items():
                if val == k:
                    val = v; break
            else:
                return None
        result[i] = val
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",    default="outputs/tri_ddim200_full.jsonl")
    p.add_argument("--imgs",     default="outputs/tri_ddim200_imgs")
    p.add_argument("--out",      default="outputs/tri_ddim200_labeled.jsonl")
    p.add_argument("--workers",  type=int, default=20)
    p.add_argument("--timeout",  type=float, default=30.0)
    p.add_argument("--max_retry", type=int, default=3)
    p.add_argument("--limit",    type=int, default=0, help="0=全量")
    p.add_argument("--api-keys", nargs="+", dest="api_keys", default=None,
                   help="覆盖默认 API key 列表")
    p.add_argument("--thinking", action="store_true", default=False,
                   help="开启 MiMo 思考模式（默认关闭）")
    return p.parse_args()


def load_done(out_path: Path) -> Set[int]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("ok"):
                    done.add(row["idx"])
            except Exception:
                pass
    return done


def main():
    args = parse_args()
    active_keys = args.api_keys if args.api_keys else _DEFAULT_KEYS
    if not active_keys:
        raise SystemExit("No API keys. Use --api-keys or create api_keys.py with MIMO_API_KEYS list.")
    print(f"使用 {len(active_keys)} 个 API key")
    jsonl_path = Path(args.jsonl)
    imgs_dir   = Path(args.imgs)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读入数据
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"共 {len(rows)} 条")

    done_idx = load_done(out_path)
    print(f"已完成 {len(done_idx)} 条，跳过")

    # 预处理：找环
    tasks = []
    for idx, row in enumerate(rows):
        if idx in done_idx:
            continue
        n = int(row["n_nodes"])
        adj = [row["adj_matrix"][i][:n] for i in range(n)]
        coords = [(float(c[0]), float(c[1])) for c in row["pred_node_coords"][:n]]
        faces = find_faces(coords, adj)
        if not faces:
            # 无法找到环，直接跳过并写空结果
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"idx": idx, "ok": True,
                                    "faces": [], "face_types": [],
                                    "note": "no_faces"}, ensure_ascii=False) + "\n")
            done_idx.add(idx)
            continue
        img_file = imgs_dir / f"{idx:05d}.png"
        if not img_file.exists():
            print(f"[WARN] 图片不存在: {img_file}")
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"idx": idx, "ok": False,
                                    "faces": [], "face_types": [],
                                    "error": "image_not_found"}, ensure_ascii=False) + "\n")
            continue
        tasks.append({
            "idx":    idx,
            "row":    row,
            "faces":  faces,
            "img":    img_file,
        })

    print(f"待处理 {len(tasks)} 条，workers={args.workers}")

    write_lock  = threading.Lock()
    key_index   = {"i": 0}
    key_lock    = threading.Lock()
    counters    = {"ok": 0, "fail": 0, "retry": 0}

    def next_key() -> str:
        with key_lock:
            k = active_keys[key_index["i"] % len(active_keys)]
            key_index["i"] += 1
            return k

    def process(task: Dict) -> Dict:
        idx   = task["idx"]
        row   = task["row"]
        faces = task["faces"]
        img_b64 = base64.b64encode(task["img"].read_bytes()).decode("ascii")
        prompt  = build_prompt(faces, row.get("prompt", ""))

        for attempt in range(args.max_retry):
            key = next_key()
            res = call_mimo(key, img_b64, prompt, args.timeout, args.thinking)
            if not res["ok"]:
                err = res["error"]
                if "balance" in err.lower() or "quota" in err.lower():
                    # 余额不足，换下一个 key 重试
                    continue
                print(f"[{idx}] FAIL attempt={attempt+1} err={err[:80]}")
                continue

            parsed = parse_response(res["raw"], len(faces))
            if parsed is not None:
                return {
                    "idx":        idx,
                    "ok":         True,
                    "faces":      faces,
                    "face_types": [parsed[i] for i in range(len(faces))],
                    "raw":        res["raw"],
                }
            else:
                print(f"[{idx}] parse fail attempt={attempt+1} raw={res['raw'][:120]}")

        # 全部重试失败
        return {
            "idx":   idx,
            "ok":    False,
            "faces": faces,
            "raw":   res.get("raw", ""),
            "error": "parse_failed_after_retries",
        }

    task_iter = iter(tasks)
    in_flight = {}
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # 初始填满
        while len(in_flight) < args.workers:
            try:
                task = next(task_iter)
                in_flight[pool.submit(process, task)] = task
            except StopIteration:
                break

        try:
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    task = in_flight.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {"idx": task["idx"], "ok": False,
                                  "error": f"worker_error: {e}"}

                    with write_lock:
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if result["ok"]:
                        counters["ok"] += 1
                    else:
                        counters["fail"] += 1

                    done_total = counters["ok"] + counters["fail"]
                    print(f"[{done_total}/{total}] idx={result['idx']} "
                          f"ok={result['ok']} | "
                          f"total ok={counters['ok']} fail={counters['fail']}")

                    # 补充新任务
                    try:
                        next_task = next(task_iter)
                        in_flight[pool.submit(process, next_task)] = next_task
                    except StopIteration:
                        pass

        except KeyboardInterrupt:
            print("\n中断，已保存进度，重跑可继续。")

    print(f"\n完成: ok={counters['ok']} fail={counters['fail']}")
    print(f"输出 -> {out_path}")


if __name__ == "__main__":
    main()
