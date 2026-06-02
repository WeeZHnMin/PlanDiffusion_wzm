"""
Multi-model caption translation with concurrency and resumable progress.

Current behavior:
1. Task unit is one JSONL row: one row is translated once.
2. Models are used as a shared worker pool (round-robin pick).
3. If a model fails 3 retries on one row, the model is dropped permanently.
4. If one model is dropped on one row, that same row is retried with another active model.
5. Progress is resumable by output jsonl line count + state json.
6. Output row order is kept identical to the source file.
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

DEFAULT_MODELS = [
    # Compatible with the current non-stream chat call pattern and suitable for plain translation.
    "qwen-plus-2025-07-28",
    "qwen-plus-2025-09-11",
    "qwen-plus-2025-12-01",
    "qwen-plus-latest",
    "qwen-plus",
    "qwen-plus-1220",
    "qwen-plus-0112",
    "qwen-plus-2025-04-28",
    "qwen-plus-2025-07-14",
    "qwen-plus-2025-01-25",
    "qwen-max",
    "qwen-turbo",
    "qwen-flash",
    "qwen-flash-2025-07-28",
    "qwen-long",
    "qwen-long-latest",
    "qwen-long-2025-01-25",
    "qwen3-8b",
    "qwen3-14b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1",
    "deepseek-r1-0528",
    "deepseek-v3.2-exp",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3.7-max",
    "qwen3.7-max-2026-05-20",
    "qwen3-next-80b-a3b-instruct",
    "qwen3-coder-30b-a3b-instruct",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-coder-flash",
    "qwen3-coder-flash-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-coder-plus-2025-07-22",
    "qwen3-coder-plus-2025-09-23",
    "qwen3-coder-next",
    "qwen-coder-plus",
    "qwen-coder-turbo",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5.1",
    "qwen-math-turbo",
    "qwen-math-plus",
    "qwen-math-plus-0816",
    "qwen-math-plus-0919",
    "qwen-math-plus-latest",
    "qvq-plus",
    "tongyi-xiaomi-analysis-flash",
]

DEFAULT_SRC_FILES = [
    Path("data/jsonl/viz_50000_captions_multi.jsonl"),
    Path("data/jsonl/viz_100000_captions_multi.jsonl"),
]
DEFAULT_OUT_FILES = [
    Path("data/jsonl/translated_en/viz_50000_captions_multi_en.jsonl"),
    Path("data/jsonl/translated_en/viz_100000_captions_multi_en.jsonl"),
]
DEFAULT_STATE_FILES = [
    Path("data/jsonl/translated_en/viz_50000_captions_multi_en.state.json"),
    Path("data/jsonl/translated_en/viz_100000_captions_multi_en.state.json"),
]
DEFAULT_PROMPT = """\
将以下中文翻译为英文，只回答正确且精炼的翻译结果，不要回答任何其他内容。

示例：
中文：厨房位于左侧，通过走廊与卧室相连，卧室包含两个部分，其中一个与客厅相邻，另一个与浴室相连，浴室位于底部中央。
英文：The kitchen is on the left, connected to the bedrooms via a corridor; one bedroom is adjacent to the living room and the other is connected to the bathroom, which is at the bottom center.

中文：Bed位于Bed下方，Bath位于Bed左侧，Bed右侧为Corridor，Bath右侧为Bed，Living位于Bed上方，Kitchen位于Living上方，Bath上方为Bath，Kitchen左侧为Kitchen。
英文：A bedroom is below another bedroom, a bathroom is to the left of the bedroom, a corridor is to the right of the bedroom, another bedroom is to the right of the bathroom, the living room is above the bedroom, the kitchen is above the living room, another bathroom is above the bathroom, and another kitchen is to the left of the kitchen.

中文：厨房位于左侧，与浴室相邻；走廊连接着卧室和浴室；客厅位于右侧，通过走廊与其他房间相连。
英文：The kitchen is on the left and adjacent to the bathroom; a corridor connects the bedrooms and the bathroom; the living room is on the right, connected to the other rooms via the corridor.

中文：{source}
英文：\
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--src-files", nargs="+", type=Path, default=DEFAULT_SRC_FILES)
    parser.add_argument("--out-files", nargs="+", type=Path, default=DEFAULT_OUT_FILES)
    parser.add_argument("--state-files", nargs="+", type=Path, default=DEFAULT_STATE_FILES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workers", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--retry-times", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=15)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--overwrite", action="store_true", help="delete existing outputs and states before running")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action="store_true", help="override disable-thinking")
    return parser.parse_args()


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"dropped_models": {}, "done_indices": [], "updated_at": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if "dropped_models" not in state or not isinstance(state["dropped_models"], dict):
            state["dropped_models"] = {}
        if "done_indices" not in state or not isinstance(state["done_indices"], list):
            state["done_indices"] = []
        return state
    except Exception:
        return {"dropped_models": {}, "done_indices": [], "updated_at": None}


def save_state(path: Path, state: Dict) -> None:
    state["updated_at"] = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source_rows(path: Path, limit: int) -> List[str]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(raw)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_done_indices_from_output(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            idx = row.get("source_index")
            if isinstance(idx, int):
                done.add(idx)
            elif isinstance(idx, str) and idx.isdigit():
                done.add(int(idx))
    return done


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def build_messages(prompt: str, source_text: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": prompt.format(source=source_text),
        },
    ]


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_one(
    client: OpenAI,
    model: str,
    source_text: str,
    prompt: str,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    retry_times: int,
    retry_delay: float,
) -> Dict:
    last_error = ""
    for attempt in range(1, retry_times + 1):
        try:
            kwargs = {}
            if disable_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(prompt, source_text),
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("empty_translation")
            if contains_cjk(source_text) and contains_cjk(text):
                raise ValueError(f"translation_still_contains_cjk: {text[:120]}")
            return {"ok": True, "caption": text, "attempts": attempt, "error": ""}
        except Exception as e:
            last_error = str(e)
            if attempt < retry_times:
                time.sleep(retry_delay * attempt)
    return {"ok": False, "caption": "", "attempts": retry_times, "error": last_error}


def process_dataset(
    src_file: Path,
    out_file: Path,
    state_file: Path,
    args: argparse.Namespace,
    disable_thinking: bool,
) -> Tuple[int, int, int]:
    print(f"Loading source: {src_file}")
    rows = load_source_rows(src_file, args.limit)
    if not rows:
        print(f"No rows found in {src_file}")
        return (0, 0, 0)

    if args.overwrite:
        if out_file.exists():
            out_file.unlink()
        if state_file.exists():
            state_file.unlink()

    print(f"Loading state: {state_file}")
    state = load_state(state_file)
    dropped_models: Dict[str, Dict] = dict(state.get("dropped_models", {}))
    done_indices: Set[int] = {int(x) for x in state.get("done_indices", []) if isinstance(x, int) or str(x).isdigit()}
    done_from_output = load_done_indices_from_output(out_file)
    if done_from_output - done_indices:
        done_indices.update(done_from_output)
        state["done_indices"] = sorted(done_indices)
        save_state(state_file, state)
    active_models: List[str] = [m for m in args.models if m not in dropped_models]
    if not active_models:
        print("No active models left. All configured models are dropped.")
        return (0, 0, 0)

    pending_indices = [idx for idx in range(len(rows)) if idx not in done_indices]
    total_tasks = len(pending_indices)
    print(f"Rows: {len(rows)} | Done: {len(done_indices)} | Pending: {total_tasks} | Active models: {len(active_models)}")
    print(f"Workers: {args.workers} | disable_thinking={disable_thinking}")
    if total_tasks == 0:
        print("Nothing to do. All rows are already translated.")
        return (0, 0, 0)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    model_lock = threading.Lock()
    rr_index = {"i": 0}
    counters = {"ok": 0, "err": 0, "dropped": 0, "finished": 0}
    model_done: Dict[str, int] = {m: 0 for m in active_models}
    client_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(client_local, "client", None)
        if client is None:
            client = make_client(args.api_key, args.base_url, args.timeout)
            client_local.client = client
        return client

    def pick_model(exclude: Set[str]) -> Optional[str]:
        with model_lock:
            alive = [m for m in active_models if m not in dropped_models and m not in exclude]
            if not alive:
                return None
            rr = rr_index["i"] % len(alive)
            model = alive[rr]
            rr_index["i"] += 1
            return model

    def drop_model(model: str, row_index: int, file_name: str, error: str) -> None:
        if model in dropped_models:
            return
        dropped_models[model] = {
            "reason": "3 retries failed on one row",
            "failed_row": row_index,
            "file": file_name,
            "error": error[:1000],
            "ts": int(time.time()),
        }
        state["dropped_models"] = dropped_models
        save_state(state_file, state)

    def process_row(index: int, raw: str) -> Dict:
        row = json.loads(raw)
        source_caption = row.get("caption", "")
        row["source_index"] = index
        if not row.get("ok") or not isinstance(source_caption, str) or not source_caption.strip():
            return {
                "index": index,
                "row_text": json.dumps(row, ensure_ascii=False),
                "ok": True,
                "model": "SKIP",
                "error": "",
            }

        tried: Set[str] = set()
        file_name = row.get("file", "")
        row_id = f"row_{index}"
        while True:
            model = pick_model(tried)
            if model is None:
                return {
                    "index": index,
                    "row_text": json.dumps(row, ensure_ascii=False),
                    "ok": False,
                    "model": "",
                    "error": "all_models_unavailable",
                }
            tried.add(model)
            client = get_client()
            res = call_one(
                client=client,
                model=model,
                source_text=source_caption,
                prompt=args.prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                disable_thinking=disable_thinking,
                retry_times=args.retry_times,
                retry_delay=args.retry_delay,
            )
            time.sleep(random.uniform(0.1, 0.3))
            if res["ok"]:
                row["caption"] = res["caption"]
                return {
                    "index": index,
                    "row_text": json.dumps(row, ensure_ascii=False),
                    "ok": True,
                    "model": model,
                    "error": "",
                }

            with model_lock:
                if model not in dropped_models:
                    drop_model(model, index, file_name, res["error"])
                    counters["dropped"] += 1
                    print(f"[DROP] {model} on {row_id} file={file_name} | {res['error'][:140]}")

    task_iter = iter(pending_indices)
    in_flight = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            while len(in_flight) < args.workers:
                idx = next(task_iter)
                in_flight[pool.submit(process_row, idx, rows[idx])] = idx
        except StopIteration:
            pass

        try:
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = in_flight.pop(fut)
                    row_obj = json.loads(rows[idx])
                    row_id = f"row_{idx}"
                    file_name = row_obj.get("file", "")
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {
                            "index": idx,
                            "row_text": rows[idx],
                            "ok": False,
                            "model": "",
                            "error": f"worker_error: {e}",
                        }

                    counters["finished"] += 1
                    if result["ok"]:
                        counters["ok"] += 1
                        if result["model"] not in ("", "SKIP"):
                            model_done[result["model"]] = model_done.get(result["model"], 0) + 1
                    else:
                        counters["err"] += 1

                    model_name = result["model"] if result["model"] else "NO_MODEL"
                    model_part = ""
                    if result["model"] and result["model"] != "SKIP":
                        model_part = f" | {result['model']} done={model_done.get(result['model'], 0)}"
                    print(
                        f"[{counters['finished']}/{total_tasks}] row={idx} id={row_id} file={file_name} -> {model_name}{model_part} "
                        f"| ok={counters['ok']} err={counters['err']} dropped={counters['dropped']}"
                    )

                    if result["ok"]:
                        with write_lock:
                            with out_file.open("a", encoding="utf-8") as f:
                                f.write(result["row_text"] + "\n")
                            done_indices.add(result["index"])
                            state["done_indices"] = sorted(done_indices)
                            save_state(state_file, state)

                    while len(in_flight) < args.workers:
                        try:
                            next_idx = next(task_iter)
                        except StopIteration:
                            break
                        in_flight[pool.submit(process_row, next_idx, rows[next_idx])] = next_idx
        except KeyboardInterrupt:
            print("\nInterrupted by user. Finished records are saved. Rerun to resume.")

    state["dropped_models"] = dropped_models
    save_state(state_file, state)
    alive = [m for m in args.models if m not in dropped_models]
    print("Run finished.")
    print(f"OK={counters['ok']} ERR={counters['err']} DROPPED={counters['dropped']}")
    print(f"Active models left: {len(alive)} / {len(args.models)}")
    print(f"Output: {out_file}")
    print(f"State: {state_file}")
    return (counters["ok"], counters["err"], counters["dropped"])


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Use --api-key or set DASHSCOPE_API_KEY.")
    if not (len(args.src_files) == len(args.out_files) == len(args.state_files)):
        raise SystemExit("--src-files, --out-files, and --state-files must have the same length.")

    disable_thinking = args.disable_thinking and (not args.enable_thinking)

    total_ok = 0
    total_err = 0
    total_dropped = 0
    for src_file, out_file, state_file in zip(args.src_files, args.out_files, args.state_files):
        ok_count, err_count, dropped_count = process_dataset(
            src_file=src_file,
            out_file=out_file,
            state_file=state_file,
            args=args,
            disable_thinking=disable_thinking,
        )
        total_ok += ok_count
        total_err += err_count
        total_dropped += dropped_count

    print("All datasets finished.")
    print(f"TOTAL_OK={total_ok} TOTAL_ERR={total_err} TOTAL_DROPPED={total_dropped}")


if __name__ == "__main__":
    main()
