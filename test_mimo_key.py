"""
Quick test: verify a MiMo API key works with image input and thinking disabled.

Usage:
    python test_mimo_key.py --api-key YOUR_KEY
    python test_mimo_key.py --api-key KEY1 --api-key KEY2
    python test_mimo_key.py --api-keys KEY1 KEY2
"""

import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI

DEFAULT_IMG = "data/viz_150000/00001.png"
SYSTEM_MESSAGE = "You are MiMo, an AI assistant developed by Xiaomi."
PROMPT = "How many rooms are in this floor plan? Answer with a number only."


def test_key(api_key: str, model: str, img_path: Path, base_url: str) -> None:
    print(f"\n{'='*55}")
    print(f"Key : ...{api_key[-6:]}")
    print(f"Model: {model}")
    print(f"Image: {img_path}")

    img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        thinking = (getattr(msg, "reasoning_content", None) or "").strip()
        usage = resp.usage
        if text:
            print(f"PASS — response: {repr(text[:200])}")
        elif thinking:
            print(f"PASS — (content empty, thinking only): {repr(thinking[:200])}")
            print(f"      *** max_tokens may be too small or model returns thinking-only ***")
        else:
            print(f"PASS — (both content and thinking empty)")
        if usage:
            print(f"      tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    except Exception as e:
        print(f"FAIL — {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", action="append", dest="api_keys", default=[],
                   help="API key to test (repeat for multiple keys)")
    p.add_argument("--api-keys", nargs="+", dest="api_keys_bulk", default=[],
                   help="Multiple API keys at once")
    p.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    p.add_argument("--model", default="mimo-v2.5")
    p.add_argument("--img", type=Path, default=Path(DEFAULT_IMG))
    return p.parse_args()


def main():
    args = parse_args()
    keys = args.api_keys + args.api_keys_bulk
    if not keys:
        env = os.getenv("MIMO_API_KEY", "")
        if env:
            keys = [env]
        else:
            raise SystemExit("No API key provided. Use --api-key KEY or --api-keys KEY1 KEY2")

    if not args.img.exists():
        raise SystemExit(f"Test image not found: {args.img}")

    print(f"Testing {len(keys)} key(s) with model={args.model}")
    for key in keys:
        test_key(key, args.model, args.img, args.base_url)

    print(f"\n{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()
