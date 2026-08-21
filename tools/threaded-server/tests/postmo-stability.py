#!/usr/bin/env python3
"""Long-running PostMo mixed VIT/text/priority/checkpoint stability test."""

import argparse
import base64
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request_json(base_url, method, path, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw), round((time.monotonic() - started) * 1000, 2)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw": raw.decode(errors="replace")}
        return error.code, result, round((time.monotonic() - started) * 1000, 2)


def post_chat(base_url, name, payload):
    try:
        status, result, elapsed_ms = request_json(
            base_url, "POST", "/v1/chat/completions", payload
        )
        choice = result.get("choices", [{}])[0]
        return {
            "request": name,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "model": result.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "content_length": len(choice.get("message", {}).get("content", "")),
        }
    except Exception as error:
        return {"request": name, "status": 0, "error": str(error)}


def mixed_cycle(args, image_url, cycle):
    payloads = []
    for index in range(2):
        payloads.append((f"vit-{index}", {
            "model": args.vit_model,
            "priority": index,
            "max_tokens": args.vit_tokens,
            "temperature": 0,
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the colors and shapes, and count the black corner dots."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
        }))
    for index in range(8):
        payloads.append((f"text-{index}", {
            "model": args.text_model,
            "priority": (cycle + index) % 4,
            "max_tokens": args.text_tokens,
            "temperature": 0,
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
            "messages": [{"role": "user", "content": f"Reply with exactly STABILITY-{cycle}-{index}-OK."}],
        }))

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(post_chat, args.base_url, name, payload): name
            for name, payload in payloads
        }
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return {
        "cycle": cycle,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "ok": sum(item["status"] == 200 for item in results),
        "failed": sum(item["status"] != 200 for item in results),
        "max_request_ms": max(item.get("elapsed_ms", 0) for item in results),
        "results": sorted(results, key=lambda item: item["request"]),
    }


def checkpoint_cycle(args, cycle):
    prefill = {
        "model": args.text_model,
        "id_slot": 7,
        "prompt": f"Stability checkpoint cycle {cycle}.",
        "n_predict": 0,
        "cache_prompt": True,
        "stream": False,
        "priority": 0,
    }
    status, _, prefill_ms = request_json(args.base_url, "POST", "/completion", prefill)
    if status != 200:
        return {"cycle": cycle, "ok": False, "stage": "prefill", "status": status}

    checkpoint = {
        "model": args.text_model,
        "session_id": args.session_id,
        "name": "session_latest",
    }
    status, saved, save_ms = request_json(
        args.base_url, "POST", "/slots/7?action=checkpoint", checkpoint
    )
    if status != 200:
        return {"cycle": cycle, "ok": False, "stage": "save", "status": status, "body": saved}

    status, restored, restore_ms = request_json(
        args.base_url, "POST", "/slots/6?action=checkpoint_restore", checkpoint
    )
    return {
        "cycle": cycle,
        "ok": status == 200,
        "stage": "restore",
        "status": status,
        "prefill_ms": prefill_ms,
        "save_ms": save_ms,
        "restore_ms": restore_ms,
        "token_count": saved.get("checkpoint", {}).get("token_count"),
        "state_size": saved.get("checkpoint", {}).get("state_size"),
        "body": None if status == 200 else restored,
    }


def read_rss_kb(pid):
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def resource_sampler(stop, pid, output, interval):
    memory_pattern = re.compile(r"DDR_Memory_Free\s+:\s+([0-9.]+)MB")
    while not stop.wait(0 if not output.exists() else interval):
        sample = {"timestamp": time.time(), "rss_kb": read_rss_kb(pid)}
        try:
            smi = subprocess.run(
                ["hm_smi", "-a"], capture_output=True, text=True, timeout=15, check=False
            ).stdout
            sample["device_free_mb"] = [float(value) for value in memory_pattern.findall(smi)]
        except Exception as error:
            sample["smi_error"] = str(error)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vit-model", default="qwen3.5-35b-64k")
    parser.add_argument("--text-model", default="qwen3.6-35b-16k")
    parser.add_argument("--vit-tokens", type=int, default=128)
    parser.add_argument("--text-tokens", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--session-id", default="stability-agent")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cycles_path = output_dir / "cycles.jsonl"
    checkpoints_path = output_dir / "checkpoints.jsonl"
    resources_path = output_dir / "resources.jsonl"
    summary_path = output_dir / "summary.json"
    image_url = "data:image/png;base64," + base64.b64encode(
        pathlib.Path(args.image).read_bytes()
    ).decode()

    stop_sampler = threading.Event()
    sampler = threading.Thread(
        target=resource_sampler,
        args=(stop_sampler, args.pid, resources_path, 5),
        daemon=True,
    )
    sampler.start()

    started = time.monotonic()
    cycle = 0
    request_ok = 0
    request_failed = 0
    checkpoint_ok = 0
    checkpoint_failed = 0
    try:
        while time.monotonic() - started < args.duration:
            cycle += 1
            result = mixed_cycle(args, image_url, cycle)
            request_ok += result["ok"]
            request_failed += result["failed"]
            with cycles_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")

            checkpoint_result = None
            if cycle % args.checkpoint_every == 0:
                checkpoint_result = checkpoint_cycle(args, cycle)
                checkpoint_ok += int(checkpoint_result["ok"])
                checkpoint_failed += int(not checkpoint_result["ok"])
                with checkpoints_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(checkpoint_result, ensure_ascii=False) + "\n")

            elapsed = int(time.monotonic() - started)
            print(
                f"PROGRESS elapsed={elapsed}s cycle={cycle} requests={request_ok}/{request_failed} "
                f"checkpoints={checkpoint_ok}/{checkpoint_failed} cycle_ms={result['elapsed_ms']}",
                flush=True,
            )
    finally:
        stop_sampler.set()
        sampler.join(timeout=20)

    summary = {
        "duration_target_s": args.duration,
        "duration_actual_s": round(time.monotonic() - started, 2),
        "cycles": cycle,
        "requests_ok": request_ok,
        "requests_failed": request_failed,
        "checkpoints_ok": checkpoint_ok,
        "checkpoints_failed": checkpoint_failed,
        "server_pid_start": args.pid,
        "server_pid_alive": pathlib.Path(f"/proc/{args.pid}").exists(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)
    if request_failed or checkpoint_failed or not summary["server_pid_alive"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
