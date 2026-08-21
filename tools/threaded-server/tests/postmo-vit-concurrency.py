#!/usr/bin/env python3
"""Run the acceptance mix: two VIT requests plus eight text requests."""

import argparse
import base64
import concurrent.futures
import json
import pathlib
import time
import urllib.request


def post(base_url, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
            result = json.loads(raw)
            return {
                "http": response.status,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                "model": result.get("model"),
                "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
                "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            }
    except Exception as error:
        return {
            "http": 0,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "error": str(error),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--vit-model", default="qwen3.5-35b-64k")
    parser.add_argument("--text-model", default="qwen3.6-35b-16k")
    parser.add_argument("--output")
    args = parser.parse_args()

    image = base64.b64encode(pathlib.Path(args.image).read_bytes()).decode()
    image_url = "data:image/png;base64," + image
    requests = []
    for index in range(2):
        requests.append((f"vit-{index}", {
            "model": args.vit_model,
            "priority": index,
            "max_tokens": 256,
            "temperature": 0,
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the image. List the colors and shapes, and count the black corner dots."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
        }))
    for index in range(8):
        requests.append((f"text-{index}", {
            "model": args.text_model,
            "priority": index % 4,
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
            "messages": [{"role": "user", "content": f"Reply with exactly TEXT-{index}-OK."}],
        }))

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(post, args.base_url, payload): name
            for name, payload in requests
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["request"] = futures[future]
            results.append(result)
    report = {
        "total_elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "requests": sorted(results, key=lambda item: item["request"]),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        pathlib.Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
