"""
Multi-model floorplan captioning with concurrency and resumable progress.

Current behavior:
1. Task unit is image (not model-image pair): one image is processed once.
2. Models are used as a shared worker pool (round-robin pick).
3. If a model fails 3 retries on one image, the model is dropped permanently.
4. If one model is dropped on an image, that same image is retried with another active model.
5. Progress is resumable by output jsonl + state json.
"""

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

DEFAULT_MODELS = [
    "qwen3.5-122b-a10b",
    "qwen3.5-flash",
    "qwen3.5-35b-a3b",
    "qwen3.6-flash-2026-04-16",
    "qwen3.6-35b-a3b",
    "qwen3.5-27b",
    "qwen3.6-plus-2026-04-02",
    "qwen3.6-plus",
    "qwen3.5-plus-2026-04-20",
    "qwen3.5-flash-2026-02-23",
    "kimi-k2.6",
    "qwen3.6-flash",
    "qwen3.6-27b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--img-dir", type=Path, default=Path("data/viz_50000"))
    parser.add_argument("--out-file", type=Path, default=Path("data/jsonl/viz_50000_captions_multi.jsonl"))
    parser.add_argument("--state-file", type=Path, default=Path("data/jsonl/viz_50000_captions_multi.state.json"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    parser.add_argument(
        "--prompt",
        default="一句话精炼地描述图中的各个房间的位置关系、布局关系、连接关系。一句话简洁精练地形容描述即可。",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-times", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--restart-from",
        type=int,
        default=0,
        help="truncate output/state and restart from this image number (e.g. 31200)",
    )
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", action="store_true", help="override disable-thinking")
    return parser.parse_args()


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"dropped_models": {}, "updated_at": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"dropped_models": {}, "updated_at": None}


def save_state(path: Path, state: Dict) -> None:
    state["updated_at"] = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_done_files(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("ok") is True and row.get("file"):
                    done.add(row["file"])
            except Exception:
                continue
    return done


def find_duplicate_ok_files(path: Path) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("ok") is True and row.get("file"):
                    file_name = row["file"]
                    counts[file_name] = counts.get(file_name, 0) + 1
            except Exception:
                continue
    dups = [(k, v) for k, v in counts.items() if v > 1]
    dups.sort(key=lambda x: x[1], reverse=True)
    return dups


def file_index(file_name: str) -> Optional[int]:
    stem = Path(file_name).stem
    if stem.isdigit():
        return int(stem)
    return None


def truncate_output_from(path: Path, restart_from: int) -> int:
    if restart_from <= 0 or not path.exists():
        return 0
    kept_lines = []
    removed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                kept_lines.append(raw)
                continue

            idx = file_index(row.get("file", ""))
            if idx is not None and idx >= restart_from:
                removed += 1
                continue
            kept_lines.append(json.dumps(row, ensure_ascii=False))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    tmp.replace(path)
    return removed


def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_one(
    client: OpenAI,
    model: str,
    img_b64: str,
    prompt: str,
    temperature: float,
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=temperature,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {"ok": True, "caption": text, "attempts": attempt, "error": ""}
        except Exception as e:
            last_error = str(e)
            if attempt < retry_times:
                time.sleep(retry_delay * attempt)
    return {"ok": False, "caption": "", "attempts": retry_times, "error": last_error}


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Use --api-key or set DASHSCOPE_API_KEY.")

    disable_thinking = args.disable_thinking and (not args.enable_thinking)

    print(f"Scanning images from: {args.img_dir}")
    images = sorted(args.img_dir.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No PNG images found in {args.img_dir}")

    print(f"Loading state: {args.state_file}")
    state = load_state(args.state_file)

    if args.restart_from > 0:
        removed = truncate_output_from(args.out_file, args.restart_from)
        state["dropped_models"] = {}
        save_state(args.state_file, state)
        print(f"Restart from image {args.restart_from}: truncated {removed} old rows and reset dropped models.")

    dropped_models: Dict[str, Dict] = dict(state.get("dropped_models", {}))
    print(f"Loading done records: {args.out_file}")
    duplicate_ok = find_duplicate_ok_files(args.out_file)
    if duplicate_ok:
        preview = ", ".join([f"{name}x{cnt}" for name, cnt in duplicate_ok[:10]])
        raise SystemExit(
            f"Duplicate ok records found in output file ({len(duplicate_ok)} files). "
            f"Examples: {preview}. Please deduplicate or use a new --out-file."
        )
    done_files = load_done_files(args.out_file)

    active_models: List[str] = [m for m in args.models if m not in dropped_models]
    if not active_models:
        print("No active models left. All configured models are dropped.")
        return

    todo_images = [p for p in images if p.name not in done_files]
    total_tasks = len(todo_images)
    print(f"Images: {len(images)} | Pending images: {total_tasks} | Active models: {len(active_models)}")
    print(f"Workers: {args.workers} | disable_thinking={disable_thinking}")
    if total_tasks == 0:
        print("Nothing to do. All images are already done.")
        return

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    model_lock = threading.Lock()
    b64_lock = threading.Lock()
    rr_index = {"i": 0}

    b64_cache: Dict[str, str] = {}
    model_done: Dict[str, int] = {m: 0 for m in active_models}
    counters = {"ok": 0, "err": 0, "dropped": 0, "finished": 0}

    def get_img_b64(img_path: Path) -> str:
        key = img_path.name
        with b64_lock:
            cached = b64_cache.get(key)
            if cached is not None:
                return cached
        encoded = img_to_b64(img_path)
        with b64_lock:
            b64_cache[key] = encoded
        return encoded

    def pick_model(exclude: Set[str]) -> Optional[str]:
        with model_lock:
            alive = [m for m in active_models if m not in dropped_models and m not in exclude]
            if not alive:
                return None
            rr = rr_index["i"] % len(alive)
            model = alive[rr]
            rr_index["i"] += 1
            return model

    def drop_model(model: str, file_name: str, error: str) -> None:
        if model in dropped_models:
            return
        dropped_models[model] = {
            "reason": "3 retries failed on one image",
            "failed_file": file_name,
            "error": error[:1000],
            "ts": int(time.time()),
        }
        state["dropped_models"] = dropped_models
        save_state(args.state_file, state)

    def process_image(img_path: Path) -> Dict:
        tried: Set[str] = set()
        img_b64 = get_img_b64(img_path)
        while True:
            model = pick_model(tried)
            if model is None:
                return {
                    "ts": int(time.time()),
                    "model": "",
                    "file": img_path.name,
                    "ok": False,
                    "caption": "",
                    "error": "all_models_unavailable",
                    "attempts": 0,
                    "elapsed": 0.0,
                    "base_url": args.base_url,
                }
            tried.add(model)
            client = make_client(args.api_key, args.base_url, args.timeout)
            t0 = time.time()
            res = call_one(
                client=client,
                model=model,
                img_b64=img_b64,
                prompt=args.prompt,
                temperature=args.temperature,
                disable_thinking=disable_thinking,
                retry_times=args.retry_times,
                retry_delay=args.retry_delay,
            )
            elapsed = round(time.time() - t0, 3)
            if res["ok"]:
                return {
                    "ts": int(time.time()),
                    "model": model,
                    "file": img_path.name,
                    "ok": True,
                    "caption": res["caption"],
                    "error": "",
                    "attempts": res["attempts"],
                    "elapsed": elapsed,
                    "base_url": args.base_url,
                }

            with model_lock:
                if model not in dropped_models:
                    drop_model(model, img_path.name, res["error"])
                    counters["dropped"] += 1
                    print(f"[DROP] {model} on {img_path.name} | {res['error'][:140]}")

    task_iter = iter(todo_images)
    in_flight = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            while len(in_flight) < args.workers:
                img = next(task_iter)
                in_flight[pool.submit(process_image, img)] = img
        except StopIteration:
            pass

        try:
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    img = in_flight.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as e:
                        row = {
                            "ts": int(time.time()),
                            "model": "",
                            "file": img.name,
                            "ok": False,
                            "caption": "",
                            "error": f"worker_error: {e}",
                            "attempts": 0,
                            "elapsed": 0.0,
                            "base_url": args.base_url,
                        }

                    with write_lock:
                        with args.out_file.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    counters["finished"] += 1
                    if row["ok"]:
                        counters["ok"] += 1
                        done_files.add(row["file"])
                        model_done[row["model"]] = model_done.get(row["model"], 0) + 1
                    else:
                        counters["err"] += 1

                    model_name = row["model"] if row["model"] else "NO_MODEL"
                    model_part = ""
                    if row["model"]:
                        model_part = f" | {row['model']} done={model_done.get(row['model'], 0)}"
                    print(
                        f"[{counters['finished']}/{total_tasks}] {img.name} -> {model_name}{model_part} "
                        f"| ok={counters['ok']} err={counters['err']} dropped={counters['dropped']}"
                    )

                    while len(in_flight) < args.workers:
                        try:
                            next_img = next(task_iter)
                        except StopIteration:
                            break
                        in_flight[pool.submit(process_image, next_img)] = next_img
        except KeyboardInterrupt:
            print("\nInterrupted by user. Finished records are saved. Rerun to resume.")

    state["dropped_models"] = dropped_models
    save_state(args.state_file, state)
    alive = [m for m in args.models if m not in dropped_models]
    print("Run finished.")
    print(f"OK={counters['ok']} ERR={counters['err']} DROPPED={counters['dropped']}")
    print(f"Active models left: {len(alive)} / {len(args.models)}")
    print(f"Output: {args.out_file}")
    print(f"State: {args.state_file}")


if __name__ == "__main__":
    main()
