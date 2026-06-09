"""
caption_floorplans.py 的 MiMo 版本，仅更换模型为小米 MiMo。

变更：
  - API key 环境变量: MIMO_API_KEY
  - base_url: https://api.xiaomimimo.com/v1
  - 默认模型: mimo-v2.5
  - 加入 system message（MiMo 要求）
  - 去掉 enable_thinking extra_body（MiMo 不支持该参数）
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
    "mimo-v2.5",
    "mimo-v2-flash",
    "mimo-v2-omni",
    "mimo-v2.5-pro",
    "mimo-v2-pro"
]

SYSTEM_MESSAGE = "You are MiMo, an AI assistant developed by Xiaomi."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("MIMO_API_KEY", ""))
    parser.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    parser.add_argument("--img-dir", type=Path, default=Path("data/viz_150000"))
    parser.add_argument("--out-file", type=Path, default=Path("data/jsonl/viz_150000_captions_mimo.jsonl"))
    parser.add_argument("--state-file", type=Path, default=Path("data/jsonl/viz_150000_captions_mimo.state.json"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    parser.add_argument(
        "--prompt",
        default="""Describe the positional and adjacency relationships of each room using directional words (left, right, above, below, center, corner, etc.). Do not use demonstrative words such as "in the picture", "layout" or "this". State spatial facts only — do not describe colors, styles, or visual appearance. Use one or a few concise sentences.

Examples of the expected style:
The kitchen is on the left, adjacent to the bathroom; the living room is to the right of the bathroom; the two bedrooms are on either side of the living room.
The corridor is centrally located, connecting five bedrooms, three bathrooms, the kitchen, and the living room. On the left is a bathroom and a bedroom; above are two bedrooms; on the right is a bedroom and a bathroom; below are the kitchen and living room, placed side by side beneath the corridor.
The bedroom is located at the lower left, the kitchen is above the bedroom with aligned right edges, the living room extends to the right and downward from the kitchen, and the bathroom is below the living room with aligned left edges.

Now describe the floorplan:""",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument(
        "--restart-from",
        type=int,
        default=0,
        help="truncate output/state and restart from this image number (e.g. 31200)",
    )
    parser.add_argument("--max-model-failures", type=int, default=500,
                        help="permanently drop a model after this many cumulative failures")
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


def is_transient_error(error: str) -> bool:
    low = error.lower()
    return "429" in error or "timed out" in low or "timeout" in low or "rate limit" in low


def call_one(
    client: OpenAI,
    model: str,
    img_b64: str,
    prompt: str,
    temperature: float,
) -> Dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "caption": text, "error": ""}
    except Exception as e:
        return {"ok": False, "caption": "", "error": str(e)}


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Use --api-key or set MIMO_API_KEY.")

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
    print(f"Workers: {args.workers}")
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
    model_fail_count: Dict[str, int] = {m: 0 for m in active_models}
    model_generation: Dict[str, int] = {m: 0 for m in active_models}
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
        attempts = 0
        t0 = time.time()
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
                    "attempts": attempts,
                    "elapsed": round(time.time() - t0, 3),
                    "base_url": args.base_url,
                }
            tried.add(model)
            with model_lock:
                gen = model_generation.get(model, 0)
            client = make_client(args.api_key, args.base_url, args.timeout)
            res = call_one(
                client=client,
                model=model,
                img_b64=img_b64,
                prompt=args.prompt,
                temperature=args.temperature,
            )
            attempts += 1
            if res["ok"]:
                with model_lock:
                    model_fail_count[model] = 0
                    model_generation[model] = model_generation.get(model, 0) + 1
                return {
                    "ts": int(time.time()),
                    "model": model,
                    "file": img_path.name,
                    "ok": True,
                    "caption": res["caption"],
                    "error": "",
                    "attempts": attempts,
                    "elapsed": round(time.time() - t0, 3),
                    "base_url": args.base_url,
                }

            with model_lock:
                if model_generation.get(model, 0) == gen:
                    model_fail_count[model] = model_fail_count.get(model, 0) + 1
                fail_count = model_fail_count.get(model, 0)

            if is_transient_error(res["error"]):
                if fail_count >= args.max_model_failures:
                    with model_lock:
                        if model not in dropped_models:
                            drop_model(model, img_path.name, res["error"])
                            counters["dropped"] += 1
                            print(f"[DROP] {model} | failures={fail_count} >= {args.max_model_failures} | {res['error'][:100]}")
                else:
                    print(f"[SKIP] {model} | failures={fail_count} | {img_path.name} | {res['error'][:100]}")
            else:
                with model_lock:
                    if model not in dropped_models:
                        drop_model(model, img_path.name, res["error"])
                        counters["dropped"] += 1
                        print(f"[DROP] {model} | {img_path.name} | {res['error'][:120]}")
            time.sleep(2)

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

                    model_name = row["model"] if row["model"] else "NO_MODEL"
                    model_part = f" | {row['model']} done={model_done.get(row['model'], 0)}" if row["model"] else ""
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
