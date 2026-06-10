"""
LLM-based quality check for text prompts in test_graph_dataset_8k.jsonl.

For each row the LLM is asked to judge whether the `prompt` field is a
well-formed English floor-plan description (mentions room types and spatial
relationships, is coherent, is not corrupted / gibberish).

Output JSONL columns (all original columns preserved):
    ok           bool   True if the prompt passes quality check
    check_reason str    Short reason from the LLM

Retry logic (mirrors caption_floorplans.py):
  - Each model gets ONE attempt per row (no per-model inner retry loop).
  - Transient error (429 / timeout / rate-limit): skip this model for this row,
    try the next model. Only permanently drop after cumulative failures
    across ALL rows >= --max-model-failures.
  - Fatal error (bad params, model not found, auth): drop that model globally
    immediately.
  - On success the model's failure counter resets to 0.
  - Resumable via output jsonl source_index + state json.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import OpenAI

DEFAULT_MODELS = [
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
]

DEFAULT_SRC   = Path("data/jsonl/test_graph_dataset_8k.jsonl")
DEFAULT_OUT   = Path("data/jsonl/test_graph_dataset_8k_checked.jsonl")
DEFAULT_STATE = Path("data/jsonl/test_graph_dataset_8k_checked.state.json")

SYSTEM_PROMPT = """\
You are a strict quality-control assistant for architectural floor-plan datasets.
Given an English text description of a residential floor plan, you must decide
whether it is a valid, well-formed description.

A VALID description must:
1. Be written in English (not garbled, not mixed with other languages in a broken way).
2. Describe at least one room type (e.g. bedroom, bathroom, kitchen, living room,
   dining room, corridor, hallway, balcony, etc.).
3. Contain at least one spatial relationship (e.g. adjacent to, connected to,
   above, below, next to, to the left/right of, etc.).
4. Be coherent and readable — not repetitive nonsense or a raw token dump.

An INVALID description fails any of the above criteria.

--- Examples ---

Description: 'The living room is in the center of the house, with the kitchen to the north, the balcony to the south, the master bedroom to the east, and the secondary bedroom and bathroom to the west.'
{"ok": true, "reason": "Clear English description with multiple room types and explicit directional relationships."}

Description: 'The master bedroom is adjacent to the main bathroom and separated from the living room by an entrance foyer; the secondary bedroom is near the guest bathroom, adjacent to the kitchen, which connects to the dining room, leading to the living room, which opens to a balcony.'
{"ok": true, "reason": "Well-formed description with rich room types and detailed spatial connections."}

Description: 'The kitchen is on the left, connected to the bedroom; a corridor runs through it, linking the bedroom, bathroom, and living room; the bathroom is below the corridor, near the bedroom; the living room is on the right, connected to the bedroom via the corridor.'
{"ok": true, "reason": "Coherent layout description with clear topology and spatial relationships."}

Description: 'Bed Bed Bath Corridor Living Kitchen Bath Kitchen Bed Living Corridor Bath'
{"ok": false, "reason": "Raw token sequence with no spatial relationships or sentence structure."}

Description: 'The 卧室 is next to the 厨房, 走廊 connects them.'
{"ok": false, "reason": "Mixed Chinese and English in a broken way; not a valid English description."}

Description: 'This is a house. It has rooms. The rooms are nice and well designed for the family.'
{"ok": false, "reason": "No specific room types or spatial relationships mentioned."}

--- End of Examples ---

Reply with a JSON object only — no explanation outside the JSON:
{"ok": true, "reason": "<one short sentence>"}
or
{"ok": false, "reason": "<one short sentence explaining the problem>"}
"""

USER_TEMPLATE = "Description: '{prompt}'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",           default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--base-url",          default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--src",               type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out",               type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state",             type=Path, default=DEFAULT_STATE)
    parser.add_argument("--models",            nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workers",           type=int, default=100)
    parser.add_argument("--limit",             type=int, default=0, help="0 = all rows")
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top-p",             type=float, default=0.1)
    parser.add_argument("--timeout",           type=float, default=30.0)
    parser.add_argument("--max-model-failures", type=int, default=500,
                        help="Permanently drop a model after this many cumulative transient failures.")
    parser.add_argument("--overwrite",         action="store_true")
    parser.add_argument("--disable-thinking",  action="store_true", default=True)
    parser.add_argument("--enable-thinking",   action="store_true", help="Override disable-thinking.")
    return parser.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"dropped_models": {}, "done_indices": [], "updated_at": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        state.setdefault("dropped_models", {})
        state.setdefault("done_indices", [])
        return state
    except Exception:
        return {"dropped_models": {}, "done_indices": [], "updated_at": None}


def save_state(path: Path, state: Dict) -> None:
    state["updated_at"] = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source_rows(path: Path, limit: int) -> List[str]:
    rows: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(raw)
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def load_done_from_output(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                idx = row.get("source_index")
                if isinstance(idx, int):
                    done.add(idx)
            except Exception:
                pass
    return done


def is_transient_error(error: str) -> bool:
    """429 / timeout / rate-limit: skip this model for this row, but don't drop globally yet."""
    low = error.lower()
    return "429" in error or "timed out" in low or "timeout" in low or "rate limit" in low


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_one(
    client: OpenAI,
    model: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
) -> Dict:
    """Single attempt — no retry loop. Returns {ok, verdict_ok, reason, error}."""
    try:
        kwargs: Dict = {}
        if disable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(prompt=prompt_text)},
            ],
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        verdict = json.loads(raw)
        if not isinstance(verdict.get("ok"), bool):
            raise ValueError(f"missing bool 'ok': {raw[:200]}")
        return {
            "ok":         True,
            "verdict_ok": verdict["ok"],
            "reason":     str(verdict.get("reason", "")),
            "error":      "",
        }
    except Exception as e:
        return {"ok": False, "verdict_ok": False, "reason": "", "error": str(e)}


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    disable_thinking = args.disable_thinking and (not args.enable_thinking)

    if args.overwrite:
        for p in (args.out, args.state):
            if p.exists():
                p.unlink()

    print(f"Loading source: {args.src}")
    rows = load_source_rows(args.src, args.limit)
    if not rows:
        raise SystemExit("No rows found.")

    state = load_state(args.state)
    dropped_models: Dict[str, Dict] = dict(state.get("dropped_models", {}))
    done_indices: Set[int] = {
        int(x) for x in state.get("done_indices", [])
        if isinstance(x, int) or str(x).isdigit()
    }
    done_from_out = load_done_from_output(args.out)
    if done_from_out - done_indices:
        done_indices.update(done_from_out)
        state["done_indices"] = sorted(done_indices)
        save_state(args.state, state)

    active_models: List[str] = [m for m in args.models if m not in dropped_models]
    if not active_models:
        raise SystemExit("All models are dropped.")

    pending = [i for i in range(len(rows)) if i not in done_indices]
    total = len(pending)
    print(f"Rows: {len(rows)} | Done: {len(done_indices)} | Pending: {total} | Models: {len(active_models)}")
    print(f"Workers: {args.workers} | disable_thinking={disable_thinking}")
    if total == 0:
        print("All rows already checked.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    model_lock = threading.Lock()
    rr_index   = {"i": 0}
    counters   = {"ok": 0, "fail": 0, "dropped": 0, "finished": 0}

    # Cumulative transient failure count per model (resets to 0 on success).
    model_fail_count:  Dict[str, int] = {m: 0 for m in active_models}
    # Bumped on each success so in-flight stale failure increments are discarded.
    model_generation:  Dict[str, int] = {m: 0 for m in active_models}
    model_done:        Dict[str, int] = {m: 0 for m in active_models}

    def pick_model(exclude: Set[str]) -> Optional[str]:
        with model_lock:
            alive = [m for m in active_models if m not in dropped_models and m not in exclude]
            if not alive:
                return None
            m = alive[rr_index["i"] % len(alive)]
            rr_index["i"] += 1
            return m

    def drop_model(model: str, row_idx: int, error: str) -> None:
        if model in dropped_models:
            return
        dropped_models[model] = {
            "failed_row": row_idx,
            "error": error[:800],
            "ts": int(time.time()),
        }
        state["dropped_models"] = dropped_models
        save_state(args.state, state)

    def process_row(index: int, raw: str) -> Dict:
        row = json.loads(raw)
        prompt_text = row.get("prompt", "")
        row["source_index"] = index

        if not prompt_text.strip():
            row["ok"] = False
            row["check_reason"] = "empty_prompt"
            return {"index": index, "row_text": json.dumps(row, ensure_ascii=False), "model": "SKIP"}

        tried: Set[str] = set()
        client = make_client(args.api_key, args.base_url, args.timeout)

        while True:
            model = pick_model(tried)
            if model is None:
                row["ok"] = False
                row["check_reason"] = "all_models_unavailable"
                return {"index": index, "row_text": json.dumps(row, ensure_ascii=False), "model": ""}

            tried.add(model)

            # Snapshot generation before the call so stale failure increments can be detected.
            with model_lock:
                gen = model_generation.get(model, 0)

            res = call_one(
                client=client,
                model=model,
                prompt_text=prompt_text,
                temperature=args.temperature,
                top_p=args.top_p,
                disable_thinking=disable_thinking,
            )

            if res["ok"]:
                # Success: reset failure counter and bump generation.
                with model_lock:
                    model_fail_count[model] = 0
                    model_generation[model] = model_generation.get(model, 0) + 1
                row["ok"] = res["verdict_ok"]
                row["check_reason"] = res["reason"]
                return {"index": index, "row_text": json.dumps(row, ensure_ascii=False), "model": model}

            # Failure — classify and decide whether to drop.
            with model_lock:
                if model_generation.get(model, 0) == gen:
                    model_fail_count[model] = model_fail_count.get(model, 0) + 1
                fail_count = model_fail_count.get(model, 0)

            if is_transient_error(res["error"]):
                if fail_count >= args.max_model_failures:
                    with model_lock:
                        if model not in dropped_models:
                            drop_model(model, index, res["error"])
                            counters["dropped"] += 1
                            print(f"[DROP] {model} | failures={fail_count} >= {args.max_model_failures} | {res['error'][:100]}")
                else:
                    print(f"[SKIP] {model} | failures={fail_count} | row={index} | {res['error'][:100]}")
            else:
                # Fatal error: drop immediately.
                with model_lock:
                    if model not in dropped_models:
                        drop_model(model, index, res["error"])
                        counters["dropped"] += 1
                        print(f"[DROP] {model} | row={index} | {res['error'][:120]}")

            time.sleep(1)

    task_iter = iter(pending)
    in_flight: Dict = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            while len(in_flight) < args.workers:
                idx = next(task_iter)
                in_flight[pool.submit(process_row, idx, rows[idx])] = idx
        except StopIteration:
            pass

        try:
            while in_flight:
                done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    idx = in_flight.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {
                            "index": idx,
                            "row_text": rows[idx],
                            "model": "",
                            "error": str(e),
                        }

                    counters["finished"] += 1
                    model_name = result.get("model", "")
                    row_obj = json.loads(result["row_text"]) if isinstance(result.get("row_text"), str) else {}
                    if row_obj.get("ok") is not False or model_name == "SKIP":
                        counters["ok"] += 1
                        if model_name and model_name != "SKIP":
                            model_done[model_name] = model_done.get(model_name, 0) + 1
                    else:
                        counters["fail"] += 1

                    model_part = f" | {model_name} done={model_done.get(model_name, 0)}" if model_name and model_name != "SKIP" else ""
                    print(
                        f"[{counters['finished']}/{total}] row={idx} -> {model_name or 'NO_MODEL'}{model_part} "
                        f"| ok={counters['ok']} fail={counters['fail']} dropped={counters['dropped']}"
                    )

                    with write_lock:
                        with args.out.open("a", encoding="utf-8") as f:
                            f.write(result["row_text"] + "\n")
                        done_indices.add(result["index"])
                        state["done_indices"] = sorted(done_indices)
                        save_state(args.state, state)

                    while len(in_flight) < args.workers:
                        try:
                            ni = next(task_iter)
                        except StopIteration:
                            break
                        in_flight[pool.submit(process_row, ni, rows[ni])] = ni

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved. Rerun to resume.")

    state["dropped_models"] = dropped_models
    save_state(args.state, state)
    alive = [m for m in args.models if m not in dropped_models]
    print(f"\nDone. OK={counters['ok']} FAIL={counters['fail']} DROPPED={counters['dropped']}")
    print(f"Active models left: {len(alive)} / {len(args.models)}")
    print(f"Output → {args.out}")
    print(f"State  → {args.state}")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or pass --api-key.")
    run(args)


if __name__ == "__main__":
    main()
