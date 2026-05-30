"""
Probe which models support vision (image input) by sending a tiny test image.

Usage:
    python probe_vision_models.py --api-key YOUR_KEY
    python probe_vision_models.py --api-key YOUR_KEY --workers 20
    python probe_vision_models.py --api-key YOUR_KEY --img data/viz_50000/00001.png

Output:
    probe_vision_results.jsonl  — one line per model
    probe_vision_summary.txt    — sorted pass/fail list
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

MODELS = [
    "qwen-math-turbo",
    "qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-32b-thinking",
    "qwen-plus-2025-07-28",
    "deepseek-r1-distill-qwen-7b",
    "qwen-vl-plus-latest",
    "qwen-max",
    "glm-5",
    "qwen-mt-flash",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3.6-plus",
    "qwen-vl-ocr-latest",
    "qwen3-32b",
    "deepseek-r1-distill-qwen-32b",
    "qwen3.6-flash",
    "qwen-vl-plus",
    "qwen-long",
    "qwen3.5-35b-a3b",
    "glm-4.5-air",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-vl-8b-thinking",
    "qwen3-coder-plus",
    "qwen3-vl-flash-2025-10-15",
    "qwen3.5-flash-2026-02-23",
    "qwen3-max-preview",
    "qwen-vl-ocr-1028",
    "qwen3-8b",
    "qwen-plus-0112",
    "qwen-plus",
    "gui-plus",
    "qwen-math-plus",
    "qwen-turbo",
    "qvq-max",
    "qwen3-coder-flash",
    "qwen3-next-80b-a3b-thinking",
    "qwen3.5-27b",
    "tongyi-xiaomi-analysis-flash",
    "deepseek-r1",
    "qwen3-vl-flash",
    "qwen-math-plus-0919",
    "qwen3-14b",
    "MiniMax-M2.5",
    "qwen-plus-2025-12-01",
    "qwen3-max-2025-09-23",
    "qwen-plus-character",
    "deepseek-v4-pro",
    "qwen-flash-character",
    "MiniMax-M2.1",
    "deepseek-r1-distill-qwen-14b",
    "qwen3-30b-a3b-instruct-2507",
    "qwen-flash",
    "qwen-flash-2025-07-28",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-coder-plus-2025-07-22",
    "kimi-k2.5",
    "qwen3.5-plus-2026-04-20",
    "qwen3.7-max",
    "qwen-vl-ocr",
    "kimi-k2.6",
    "qwen-long-latest",
    "qwen-vl-ocr-2025-04-13",
    "qwen-plus-1220",
    "qwen-vl-ocr-2025-11-20",
    "qwen3.5-122b-a10b",
    "tongyi-intent-detect-v3",
    "qwen3-max",
    "qwen3.5-plus-2026-02-15",
    "qwen3-235b-a22b-thinking-2507",
    "glm-5.1",
    "qwen3.7-max-preview",
    "kimi-k2-thinking",
    "qwen3.6-max-preview",
    "deepseek-v3.1",
    "qwen3.5-397b-a17b",
    "qwen3-vl-plus-2025-09-23",
    "deepseek-v3.2",
    "qwen3-coder-next",
    "qwen-math-plus-0816",
    "tongyi-xiaomi-analysis-pro",
    "qwen3.5-flash",
    "qwen3-vl-32b-instruct",
    "deepseek-v4-flash",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-coder-plus-2025-09-23",
    "qwen-plus-latest",
    "Moonshot-Kimi-K2-Instruct",
    "qwen3-max-2026-01-23",
    "qwen-plus-2025-09-11",
    "qwen3-vl-flash-2026-01-22",
    "qwen3.7-max-2026-05-20",
    "qwen-vl-max",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-235b-a22b-instruct",
    "qwen3-coder-30b-a3b-instruct",
    "qwen-flash-character-2026-02-26",
    "qwen3.6-27b",
    "qwen3-235b-a22b",
    "qwen-coder-plus",
    "qwen-mt-lite",
    "qwen-plus-2025-01-25",
    "qwen3.6-flash-2026-04-16",
    "qwen3-vl-plus",
    "qwen3.7-max-2026-05-17",
    "glm-4.5",
    "qwen3-30b-a3b",
    "glm-4.6",
    "qwen-coder-turbo",
    "qwen-mt-plus",
    "glm-4.7",
    "qwen3-vl-8b-instruct",
    "qwen-vl-ocr-2025-08-28",
    "qwen3-coder-flash-2025-07-28",
    "qvq-plus",
    "deepseek-v3",
    "gui-plus-2026-02-26",
    "qwen3-vl-plus-2025-12-19",
    "qwen-plus-2025-04-28",
    "qwen-mt-turbo",
    "qwen3.5-plus",
    "qwen-long-2025-01-25",
    "qwen3.6-35b-a3b",
    "qwen-math-plus-latest",
    "qwen-plus-2025-07-14",
    "qwq-plus",
    "qwen3.6-plus-2026-04-02",
    "deepseek-r1-0528",
    "qwen3-next-80b-a3b-instruct",
    "deepseek-v3.2-exp",
    "llama-4-scout-17b-16e-instruct",
    "llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
]

PROBE_PROMPT = "图中有几个房间？只回答数字。"

# Keywords that indicate the model rejected the image (not a vision model)
REJECT_KEYWORDS = [
    "不支持", "无法处理", "无法识别", "不能处理", "不支持图片",
    "unsupported", "does not support", "cannot process", "not support",
    "image input", "no image", "text only", "文字模型",
    "模型不支持", "该模型", "功能不支持",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    p.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--img", type=Path, default=Path("data/viz_50000/00001.png"))
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-elapsed", type=float, default=15.0, help="exclude models slower than this (seconds)")
    p.add_argument("--out", type=Path, default=Path("probe_vision_results.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("probe_vision_summary.txt"))
    return p.parse_args()


def is_rejection(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in REJECT_KEYWORDS)


def probe(model: str, img_b64: str, client: OpenAI, prompt: str) -> dict:
    t0 = time.time()
    result = {"model": model, "vision": False, "error": "", "response": "", "elapsed": 0.0}
    try:
        kwargs = {}
        if "qwen3" in model.lower() or model.lower().startswith("qwq") or model.lower().startswith("qvq"):
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
            **kwargs,
        )
        text = (resp.choices[0].message.content or "").strip()
        result["response"] = text[:200]
        result["vision"] = bool(text) and not is_rejection(text)
    except Exception as e:
        err = str(e)
        result["error"] = err[:300]
        # Some errors explicitly say multimodal not supported
        result["vision"] = False
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key required: --api-key or DASHSCOPE_API_KEY env var")
    if not args.img.exists():
        raise SystemExit(f"Test image not found: {args.img}")

    # Hard-cut HTTP requests slightly above max_elapsed so slow models don't block workers
    effective_timeout = min(args.timeout, args.max_elapsed + 2)
    img_b64 = base64.b64encode(args.img.read_bytes()).decode("ascii")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=effective_timeout)
    print(f"HTTP timeout: {effective_timeout}s (max_elapsed={args.max_elapsed}s)\n")

    print(f"Probing {len(MODELS)} models with {args.workers} workers...")
    print(f"Test image: {args.img}\n")

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, m, img_b64, client, PROBE_PROMPT): m for m in MODELS}
        done_count = 0
        with args.out.open("w", encoding="utf-8") as f:
            for fut in as_completed(futures):
                r = fut.result()
                done_count += 1
                tag = "PASS" if r["vision"] else "FAIL"
                print(f"[{done_count:>3}/{len(MODELS)}] {tag} {r['model']:<50} {r['elapsed']}s"
                      + (f"  -> {r['response'][:60]}" if r["vision"] else f"  ERR: {r['error'][:60]}"))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                results.append(r)

    passed_fast = sorted([r for r in results if r["vision"] and r["elapsed"] <= args.max_elapsed], key=lambda r: r["elapsed"])
    passed_slow = sorted([r for r in results if r["vision"] and r["elapsed"] > args.max_elapsed], key=lambda r: r["elapsed"])
    failed = sorted([r for r in results if not r["vision"]], key=lambda r: r["model"])

    def fmt(r):
        return f"{r['model']:<52} {r['elapsed']:>6.2f}s"

    summary_lines = [
        f"=== Vision-capable & fast (<={args.max_elapsed}s): {len(passed_fast)} models ===",
        *[fmt(r) for r in passed_fast],
        "",
        f"=== Vision-capable but SLOW (>{args.max_elapsed}s): {len(passed_slow)} models ===",
        *[fmt(r) for r in passed_slow],
        "",
        f"=== Not vision-capable: {len(failed)} models ===",
        *[r["model"] for r in failed],
    ]
    args.summary.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"PASS fast: {len(passed_fast)}  PASS slow: {len(passed_slow)}  FAIL: {len(failed)}")
    print(f"Results: {args.out}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
