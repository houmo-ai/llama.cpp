#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def request_json(base_url, method, path, data=None, timeout=600):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8")
        try:
            detail = json.loads(payload)
        except json.JSONDecodeError:
            detail = payload
        raise RuntimeError(f"{method} {path} failed: HTTP {error.code}: {detail}") from error


def checkpoint(base_url, model, slot, session_id, name):
    _, result = request_json(
        base_url,
        "POST",
        f"/slots/{slot}?action=checkpoint",
        {
            "model": model,
            "session_id": session_id,
            "name": name,
        },
    )
    return result


def list_checkpoints(base_url, model, slot, session_id):
    query = urllib.parse.urlencode({"model": model, "session_id": session_id})
    _, result = request_json(
        base_url,
        "GET",
        f"/slots/{slot}/checkpoints?{query}",
    )
    return result


def restore(base_url, model, slot, session_id, name):
    _, result = request_json(
        base_url,
        "POST",
        f"/slots/{slot}?action=checkpoint_restore",
        {
            "model": model,
            "session_id": session_id,
            "name": name,
        },
    )
    return result


def prepare_demo(args):
    _, rendered = request_json(
        args.base_url,
        "POST",
        "/apply-template",
        {
            "model": args.model,
            "messages": [{"role": "system", "content": args.system_prompt}],
            "add_generation_prompt": False,
        },
    )
    prompt = rendered["prompt"]

    request_json(
        args.base_url,
        "POST",
        "/completion",
        {
            "model": args.model,
            "id_slot": args.slot,
            "prompt": prompt,
            "n_predict": 0,
            "cache_prompt": True,
            "stream": False,
        },
    )
    system_record = checkpoint(
        args.base_url,
        args.model,
        args.slot,
        args.session_id,
        "system_prompt_end",
    )

    for user_prompt in (args.user_prompt, args.user_prompt + " Continue."):
        request_json(
            args.base_url,
            "POST",
            "/v1/chat/completions",
            {
                "model": args.model,
                "id_slot": args.slot,
                "priority": args.priority,
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
            },
        )
        checkpoint(
            args.base_url,
            args.model,
            args.slot,
            args.session_id,
            "session_latest",
        )

    records = list_checkpoints(
        args.base_url, args.model, args.slot, args.session_id)
    names = sorted(item["name"] for item in records["data"] if item["valid"])
    if names != ["session_latest", "system_prompt_end"]:
        raise RuntimeError(f"expected exactly two checkpoints, got: {names}")

    print(json.dumps({
        "system_checkpoint": system_record,
        "checkpoints": records,
    }, indent=2, ensure_ascii=False))
    print("Prepare phase passed. Restart the server with the preload options, then run --mode verify.")


def verify_demo(args):
    records = list_checkpoints(
        args.base_url, args.model, args.slot, args.session_id)
    names = sorted(item["name"] for item in records["data"] if item["valid"])
    if names != ["session_latest", "system_prompt_end"]:
        raise RuntimeError(f"checkpoint registry is incomplete: {names}")

    restored = restore(
        args.base_url,
        args.model,
        args.restore_slot,
        args.session_id,
        "session_latest",
    )
    print(json.dumps({
        "checkpoints": records,
        "restored": restored,
    }, indent=2, ensure_ascii=False))
    print("Verify phase passed.")


def priority_request(args, index):
    priority = index % 4
    models = args.concurrency_models or [args.model]
    model = models[index % len(models)]
    started = time.monotonic()
    _, result = request_json(
        args.base_url,
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "priority": priority,
            "messages": [{"role": "user", "content": f"Request {index}: {args.user_prompt}"}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
        },
    )
    return {
        "index": index,
        "model": model,
        "priority": priority,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
    }


def concurrency_demo(args):
    _, props_before = request_json(args.base_url, "GET", "/props")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = [executor.submit(priority_request, args, index) for index in range(args.requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item["elapsed_ms"])
    _, props_after = request_json(args.base_url, "GET", "/props")
    print(json.dumps({
        "results": results,
        "device_arbiter_before": props_before.get("device_arbiter"),
        "device_arbiter_after": props_after.get("device_arbiter"),
    }, indent=2))
    print("Concurrency phase completed. Inspect device_arbiter counters and server permit logs.")


def main():
    parser = argparse.ArgumentParser(description="PostMo threaded server phase-one demo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--concurrency-models",
        nargs="+",
        help="models used round-robin by concurrency mode (default: --model only)",
    )
    parser.add_argument("--session-id", default="agent-42")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--restore-slot", type=int, default=1)
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--system-prompt", default="You are a reliable assistant.")
    parser.add_argument("--user-prompt", default="Give a one-sentence status update.")
    parser.add_argument("--mode", choices=("prepare", "verify", "concurrency"), required=True)
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_demo(args)
    elif args.mode == "verify":
        verify_demo(args)
    else:
        concurrency_demo(args)


if __name__ == "__main__":
    main()
