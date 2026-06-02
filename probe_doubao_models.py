"""
Probe Doubao models: test multimodal support + thinking-disabled mode.

Usage:
    python probe_doubao_models.py
    python probe_doubao_models.py --api-key YOUR_KEY
    python probe_doubao_models.py --img data/viz_50000/00001.png

Output:
    probe_doubao_results.jsonl   — one line per model
    probe_doubao_summary.txt     — sorted pass/fail list
"""

import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

MODELS = [
    "doubao-seed-2-0-code-preview-260215",
    "doubao-seed-1-8-251228",
    "doubao-seed-1-6-251015",
    "doubao-seed-2-0-mini-260428",
    "doubao-seed-2-0-lite-260428",
    "doubao-seed-2-0-pro-260215",
    "doubao-1.5-vision-pro-250328",
    "doubao-1-5-lite-32k-250115",
    "doubao-seed-1-6-flash-250828",
    "doubao-seed-1-6-thinking-250715",
    "doubao-seed-1-6-vision-250815",
    "doubao-seed-1-6-lite-251015",

    "doubao-1-5-pro-32k-250115",
    "doubao-1-5-pro-256k-250115",
]

PROBE_PROMPT = "图中有几个房间？只回答数字。"

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

REJECT_KEYWORDS = [
    "不支持", "无法处理", "无法识别", "不能处理", "不支持图片",
    "unsupported", "does not support", "cannot process", "not support",
    "image input", "no image", "text only", "文字模型",
    "模型不支持", "该模型", "功能不支持",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("ARK_API_KEY", "ark-5167bc34-fcf1-4500-b036-210e54bebbe2-e56b8"))
    p.add_argument("--img", type=Path, default=Path("data/viz_50000/00001.png"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=Path("probe_doubao_results.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("probe_doubao_summary.txt"))
    return p.parse_args()


def is_rejection(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in REJECT_KEYWORDS)


def probe(model: str, file_id: str, client: OpenAI) -> dict:
    t0 = time.time()
    result = {
        "model": model,
        "vision": False,
        "thinking_disabled": False,
        "error": "",
        "response": "",
        "elapsed": 0.0,
    }
    try:
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": file_id},
                    {"type": "input_text", "text": PROBE_PROMPT},
                ],
            }],
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )
        # Extract text from response
        text = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        text += c.text
        result["response"] = text.strip()[:200]
        result["vision"] = bool(text.strip()) and not is_rejection(text)
        result["thinking_disabled"] = True  # no error means param was accepted
    except Exception as e:
        err = str(e)
        result["error"] = err[:400]
        # Check if it failed because thinking param is unsupported vs vision unsupported
        if "thinking" in err.lower() or "extra_body" in err.lower():
            result["thinking_disabled"] = False
        result["vision"] = False
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def upload_image(img_path: Path, client: OpenAI) -> str:
    print(f"Uploading {img_path} ...")
    with img_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="user_data")
    # Poll until processed
    for _ in range(30):
        if file_obj.status != "processing":
            break
        time.sleep(2)
        file_obj = client.files.retrieve(file_obj.id)
    print(f"File ready: {file_obj.id} (status={file_obj.status})\n")
    return file_obj.id


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key required: --api-key or ARK_API_KEY env var")
    if not args.img.exists():
        raise SystemExit(f"Test image not found: {args.img}")

    client = OpenAI(api_key=args.api_key, base_url=BASE_URL, timeout=args.timeout)

    file_id = upload_image(args.img, client)

    print(f"Probing {len(MODELS)} models with {args.workers} workers...")
    print(f"Test image: {args.img}\n")

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, m, file_id, client): m for m in MODELS}
        done = 0
        with args.out.open("w", encoding="utf-8") as f:
            for fut in as_completed(futures):
                r = fut.result()
                done += 1
                vision_tag = "VIS+" if r["vision"] else "VIS-"
                think_tag  = "THK-OFF" if r["thinking_disabled"] else "THK-NA "
                print(
                    f"[{done:>2}/{len(MODELS)}] {vision_tag} {think_tag}  "
                    f"{r['model']:<45} {r['elapsed']}s"
                    + (f"  -> {r['response'][:60]}" if r["vision"] else f"  ERR: {r['error'][:80]}")
                )
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                results.append(r)

    vision_ok  = sorted([r for r in results if r["vision"]], key=lambda r: r["elapsed"])
    vision_no  = sorted([r for r in results if not r["vision"]], key=lambda r: r["model"])
    think_ok   = sorted([r for r in results if r["thinking_disabled"]], key=lambda r: r["model"])

    def fmt(r):
        return (
            f"  {r['model']:<45} {r['elapsed']:>6.2f}s"
            f"  vision={'Y' if r['vision'] else 'N'}  thinking_disabled={'Y' if r['thinking_disabled'] else 'N'}"
        )

    summary_lines = [
        f"=== Vision OK: {len(vision_ok)} models ===",
        *[fmt(r) for r in vision_ok],
        "",
        f"=== Vision FAIL: {len(vision_no)} models ===",
        *[fmt(r) for r in vision_no],
        "",
        f"=== Thinking-disabled accepted: {len(think_ok)} models ===",
        *[r["model"] for r in think_ok],
    ]
    args.summary.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Vision OK: {len(vision_ok)}  Vision FAIL: {len(vision_no)}  Thinking-disabled OK: {len(think_ok)}")
    print(f"Results:  {args.out}")
    print(f"Summary:  {args.summary}")


if __name__ == "__main__":
    main()
