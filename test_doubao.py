"""
快速测试豆包 API 是否正常响应。
用法:
    python test_doubao.py --api-key YOUR_ARK_KEY
"""
import argparse
import os
import time
from openai import OpenAI

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL    = "doubao-seed-2-0-pro-260215"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("ARK_API_KEY", ""))
    p.add_argument("--model",   default=MODEL)
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("请提供 --api-key 或设置 ARK_API_KEY 环境变量")

    client = OpenAI(api_key=args.api_key, base_url=BASE_URL, timeout=30.0)

    prompt = "请用一句话介绍自己。"
    print(f"model : {args.model}")
    print(f"prompt: {prompt}")
    print("发送中...\n")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    elapsed = time.perf_counter() - t0

    content = resp.choices[0].message.content or ""
    finish  = resp.choices[0].finish_reason
    print(f"回答 ({elapsed:.2f}s, finish={finish}):")
    print(content)

if __name__ == "__main__":
    main()
