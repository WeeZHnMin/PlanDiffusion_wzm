"""
SiliconFlow-based floorplan captioning with concurrency and resumable progress.

Uses chat.completions API with base64-encoded images (no file upload needed).
Thinking mode is disabled via extra_body={"enable_thinking": False}.

Usage:
    python caption_floorplans_siliconflow.py --api-key YOUR_KEY
    python caption_floorplans_siliconflow.py --api-key YOUR_KEY --workers 20

Resume: rerun the same command — already-completed images in the output jsonl
are skipped automatically.

Migrate from doubao results:
    copy data\\jsonl\\viz_150000_captions_doubao.jsonl data\\jsonl\\viz_150000_captions_siliconflow.jsonl
    python caption_floorplans_siliconflow.py --api-key YOUR_KEY
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
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
]

BASE_URL = "https://api.siliconflow.cn/v1"

DEFAULT_PROMPT = """
Describe the positional and adjacency relationships of each room using directional words (left, right, above, below, center, corner, etc.). Do not use demonstrative words such as "in the picture", "layout" or "this". State spatial facts only — do not describe colors, styles, or visual appearance. Use one or a few concise sentences.

Examples of the expected style:
The kitchen is on the left, adjacent to the bathroom; the living room is to the right of the bathroom; the two bedrooms are on either side of the living room.
The corridor is centrally located, connecting five bedrooms, three bathrooms, the kitchen, and the living room. On the left is a bathroom and a bedroom; above are two bedrooms; on the right is a bedroom and a bathroom; below are the kitchen and living room, placed side by side beneath the corridor.
The bedroom is located at the lower left, the kitchen is above the bedroom with aligned right edges, the living room extends to the right and downward from the kitchen, and the bathroom is below the living room with aligned left edges.

Now describe the floorplan:"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("SILICONFLOW_API_KEY"))
    p.add_argument("--img-dir", type=Path, default=Path("data/viz_150000"))
    p.add_argument("--out-file", type=Path, default=Path("data/jsonl/viz_150000_captions_siliconflow.jsonl"))
    p.add_argument("--state-file", type=Path, default=Path("data/jsonl/viz_150000_captions_siliconflow.state.json"))
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--workers", type=int, default=90)
    p.add_argument("--limit", type=int, default=0, help="0 = all images")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--retry-times", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=5.0, help="Base delay between retries (seconds)")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--restart-from", type=int, default=0,
                   help="Truncate output/state and restart from this image index")
    return p.parse_args()


# ── state / output helpers ─────────────────────────────────────────────────────

def load_state(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
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
            try:
                row = json.loads(line.strip())
                if row.get("ok") is True and row.get("file"):
                    counts[row["file"]] = counts.get(row["file"], 0) + 1
            except Exception:
                continue
    return sorted([(k, v) for k, v in counts.items() if v > 1], key=lambda x: -x[1])


def file_index(name: str) -> Optional[int]:
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else None


def truncate_output_from(path: Path, restart_from: int) -> int:
    if restart_from <= 0 or not path.exists():
        return 0
    kept, removed = [], 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                kept.append(raw)
                continue
            idx = file_index(row.get("file", ""))
            if idx is not None and idx >= restart_from:
                removed += 1
                continue
            kept.append(json.dumps(row, ensure_ascii=False))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    tmp.replace(path)
    return removed


# ── API helpers ────────────────────────────────────────────────────────────────

def make_client(api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout)


def encode_image(img_path: Path) -> str:
    data = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


# Models that do not accept the enable_thinking parameter at all
NO_THINKING_PARAM_MODELS = {
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
}


def call_one(
    client: OpenAI,
    model: str,
    img_path: Path,
    prompt: str,
    retry_times: int,
    retry_delay: float,
    max_tokens: int,
) -> Dict:
    image_url = encode_image(img_path)
    extra = {} if model in NO_THINKING_PARAM_MODELS else {"enable_thinking": False}
    last_error = ""
    for attempt in range(1, retry_times + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                stream=False,
                extra_body=extra,
            )
            text = resp.choices[0].message.content.strip()
            return {"ok": True, "caption": text, "attempts": attempt, "error": ""}
        except Exception as e:
            last_error = str(e)
            if attempt < retry_times:
                time.sleep(retry_delay * attempt)
    return {"ok": False, "caption": "", "attempts": retry_times, "error": last_error}


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key required: --api-key or SILICONFLOW_API_KEY env var")

    client = make_client(args.api_key, args.timeout)

    print(f"Scanning images from: {args.img_dir}")
    images = sorted(args.img_dir.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No PNG images found in {args.img_dir}")

    state = load_state(args.state_file)

    if args.restart_from > 0:
        removed = truncate_output_from(args.out_file, args.restart_from)
        state["dropped_models"] = {}
        save_state(args.state_file, state)
        print(f"Restart from {args.restart_from}: truncated {removed} rows, reset dropped models.")

    dups = find_duplicate_ok_files(args.out_file)
    if dups:
        preview = ", ".join(f"{n}x{c}" for n, c in dups[:10])
        raise SystemExit(f"Duplicate ok records in output ({len(dups)} files): {preview}")

    done_files = load_done_files(args.out_file)
    dropped_models: Dict[str, Dict] = dict(state.get("dropped_models", {}))
    active_models: List[str] = [m for m in args.models if m not in dropped_models]
    if not active_models:
        raise SystemExit("No active models. All configured models are dropped.")

    todo_images = [p for p in images if p.name not in done_files]
    total_tasks = len(todo_images)
    print(f"Images: {len(images)} | Done: {len(done_files)} | Pending: {total_tasks}")
    print(f"Active models: {active_models}")
    print(f"Workers: {args.workers} | retry-times: {args.retry_times} | retry-delay: {args.retry_delay}s")
    if total_tasks == 0:
        print("Nothing to do.")
        return

    write_lock = threading.Lock()
    model_lock = threading.Lock()
    rr_index = {"i": 0}
    model_done: Dict[str, int] = {m: 0 for m in active_models}
    counters = {"ok": 0, "err": 0, "dropped": 0, "finished": 0}

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    def pick_model(exclude: Set[str]) -> Optional[str]:
        with model_lock:
            alive = [m for m in active_models if m not in dropped_models and m not in exclude]
            if not alive:
                return None
            idx = rr_index["i"] % len(alive)
            model = alive[idx]
            rr_index["i"] += 1
            return model

    def drop_model(model: str, file_name: str, error: str) -> None:
        with model_lock:
            if model in dropped_models:
                return
            dropped_models[model] = {
                "reason": f"{args.retry_times} retries failed",
                "failed_file": file_name,
                "error": error[:1000],
                "ts": int(time.time()),
            }
            state["dropped_models"] = dropped_models
            save_state(args.state_file, state)

    def is_transient_error(error: str) -> bool:
        return "429" in error or "timed out" in error.lower() or "timeout" in error.lower()

    def process_image(img_path: Path) -> Dict:
        tried: Set[str] = set()
        while True:
            model = pick_model(tried)
            if model is None:
                return {
                    "ts": int(time.time()), "model": "", "file": img_path.name,
                    "ok": False, "caption": "", "error": "all_models_unavailable",
                    "attempts": 0, "elapsed": 0.0,
                }
            tried.add(model)
            t0 = time.time()
            res = call_one(
                client=client,
                model=model,
                img_path=img_path,
                prompt=args.prompt,
                retry_times=args.retry_times,
                retry_delay=args.retry_delay,
                max_tokens=args.max_tokens,
            )
            elapsed = round(time.time() - t0, 3)
            if res["ok"]:
                return {
                    "ts": int(time.time()), "model": model, "file": img_path.name,
                    "ok": True, "caption": res["caption"], "error": "",
                    "attempts": res["attempts"], "elapsed": elapsed,
                }
            if is_transient_error(res["error"]):
                # 429 / timeout: skip this model for this image only, do NOT drop globally
                print(f"[SKIP] {model} | {img_path.name} | {res['error'][:120]}")
                continue
            drop_model(model, img_path.name, res["error"])
            counters["dropped"] += 1
            print(f"[DROP] {model} | {img_path.name} | {res['error'][:120]}")

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
                            "ts": int(time.time()), "model": "", "file": img.name,
                            "ok": False, "caption": "", "error": f"worker_error: {e}",
                            "attempts": 0, "elapsed": 0.0,
                        }

                    counters["finished"] += 1
                    if row["ok"]:
                        with write_lock:
                            with args.out_file.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counters["ok"] += 1
                        done_files.add(row["file"])
                        model_done[row["model"]] = model_done.get(row["model"], 0) + 1
                    else:
                        counters["err"] += 1

                    model_name = row["model"] or "NO_MODEL"
                    done_count = model_done.get(row["model"], 0) if row["model"] else 0
                    print(
                        f"[{counters['finished']}/{total_tasks}] {img.name} -> {model_name}"
                        f" done={done_count}"
                        f" | ok={counters['ok']} err={counters['err']} dropped={counters['dropped']}"
                    )

                    while len(in_flight) < args.workers:
                        try:
                            next_img = next(task_iter)
                            in_flight[pool.submit(process_image, next_img)] = next_img
                        except StopIteration:
                            break

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved. Rerun to resume.")

    state["dropped_models"] = dropped_models
    save_state(args.state_file, state)
    alive = [m for m in args.models if m not in dropped_models]
    print("\nRun finished.")
    print(f"OK={counters['ok']}  ERR={counters['err']}  DROPPED={counters['dropped']}")
    print(f"Active models left: {len(alive)} / {len(args.models)}")
    print(f"Output: {args.out_file}")
    print(f"State:  {args.state_file}")


if __name__ == "__main__":
    main()
