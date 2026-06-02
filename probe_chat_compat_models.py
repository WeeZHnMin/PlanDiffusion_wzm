"""
Probe which models support a specific DashScope compatible-mode chat call pattern.

The probed call pattern matches the current translation script style:
1. client.chat.completions.create(...)
2. non-streaming
3. messages = [system, user]
4. optional extra_body={"enable_thinking": False}

Output is written as JSONL, one result per model.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from translate_captions_jsonl import DEFAULT_MODELS


DEFAULT_OUT_FILE = Path("data/jsonl/model_probe_chat_compat.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retry-times", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    parser.add_argument(
        "--user-prompt",
        default="Reply with exactly: OK",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="send extra_body={enable_thinking: false}",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def load_done_models(path: Path) -> Dict[str, Dict]:
    done: Dict[str, Dict] = {}
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
            model = row.get("model")
            if model:
                done[model] = row
    return done


def classify_error(error_text: str) -> str:
    text = error_text.lower()
    if "only support stream mode" in text or "please enable the stream parameter" in text:
        return "stream_required"
    if "enable_thinking parameter is restricted to true" in text:
        return "thinking_must_be_enabled"
    if "role must be in [user, assistant]" in text:
        return "message_format_restricted"
    if "model_not_found" in text or "does not exist" in text:
        return "model_not_found"
    if "access denied" in text or "access_denied" in text:
        return "access_denied"
    if "free tier" in text or "allocationquota.freetieronly" in text:
        return "free_tier_exhausted"
    if "limit_requests" in text or "rate-limit" in text or "exceeded your current request limit" in text:
        return "rate_limited"
    if "current user api does not support http call" in text:
        return "http_call_not_supported"
    if "timeout" in text:
        return "timeout"
    return "other_error"


def probe_one(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool,
    retry_times: int,
    retry_delay: float,
) -> Dict:
    last_error = ""
    for attempt in range(1, retry_times + 1):
        t0 = time.time()
        try:
            kwargs = {}
            if disable_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {
                "ts": int(time.time()),
                "model": model,
                "ok": True,
                "mode": "chat_non_stream_system_user",
                "disable_thinking": disable_thinking,
                "attempts": attempt,
                "elapsed": round(time.time() - t0, 3),
                "response_preview": text[:300],
                "error_type": "",
                "error": "",
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retry_times:
                time.sleep(retry_delay * attempt)
    return {
        "ts": int(time.time()),
        "model": model,
        "ok": False,
        "mode": "chat_non_stream_system_user",
        "disable_thinking": disable_thinking,
        "attempts": retry_times,
        "elapsed": None,
        "response_preview": "",
        "error_type": classify_error(last_error),
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Use --api-key or set DASHSCOPE_API_KEY.")

    disable_thinking = args.disable_thinking and (not args.enable_thinking)

    if args.overwrite and args.out_file.exists():
        args.out_file.unlink()

    done = load_done_models(args.out_file)
    todo_models = [model for model in args.models if model not in done]
    print(f"Models total: {len(args.models)} | Already done: {len(done)} | Pending: {len(todo_models)}")
    print(
        f"Probe mode: chat.completions + non-stream + system/user"
        f" | disable_thinking={disable_thinking} | workers={args.workers}"
    )
    if not todo_models:
        print("Nothing to do.")
        return

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counters = {"ok": 0, "err": 0, "finished": 0}
    client_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(client_local, "client", None)
        if client is None:
            client = make_client(args.api_key, args.base_url, args.timeout)
            client_local.client = client
        return client

    def worker(model: str) -> Dict:
        client = get_client()
        return probe_one(
            client=client,
            model=model,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
            disable_thinking=disable_thinking,
            retry_times=args.retry_times,
            retry_delay=args.retry_delay,
        )

    task_iter = iter(todo_models)
    in_flight = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            while len(in_flight) < args.workers:
                model = next(task_iter)
                in_flight[pool.submit(worker, model)] = model
        except StopIteration:
            pass

        while in_flight:
            done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done_futs:
                model = in_flight.pop(fut)
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "ts": int(time.time()),
                        "model": model,
                        "ok": False,
                        "mode": "chat_non_stream_system_user",
                        "disable_thinking": disable_thinking,
                        "attempts": 0,
                        "elapsed": None,
                        "response_preview": "",
                        "error_type": "worker_error",
                        "error": str(e),
                    }

                with write_lock:
                    with args.out_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                counters["finished"] += 1
                if row["ok"]:
                    counters["ok"] += 1
                    print(
                        f"[{counters['finished']}/{len(todo_models)}] {model} -> OK"
                        f" | preview={row['response_preview'][:80]}"
                    )
                else:
                    counters["err"] += 1
                    print(
                        f"[{counters['finished']}/{len(todo_models)}] {model} -> {row['error_type']}"
                        f" | {row['error'][:160]}"
                    )

                while len(in_flight) < args.workers:
                    try:
                        next_model = next(task_iter)
                    except StopIteration:
                        break
                    in_flight[pool.submit(worker, next_model)] = next_model

    print("Probe finished.")
    print(f"OK={counters['ok']} ERR={counters['err']}")
    print(f"Results: {args.out_file}")


if __name__ == "__main__":
    main()
