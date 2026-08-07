#!/usr/bin/env python3
"""Exercise HouMo HMM embedding and reranking models through llama-server.

The script intentionally uses only the Python standard library so it can run on
bring-up machines without installing the OpenAI client or a test framework.  It
starts one server instance, performs functional, concurrency, stress, and
recovery checks, then writes a JSON result suitable for attaching to a bug or
release report.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable


@dataclasses.dataclass
class HttpResult:
    status: int
    body: Any
    elapsed_ms: float
    error: str | None = None


@dataclasses.dataclass
class Check:
    name: str
    passed: bool
    detail: str
    elapsed_ms: float = 0.0
    informational: bool = False


class Validation:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        name: str,
        passed: bool,
        detail: str,
        elapsed_ms: float = 0.0,
        *,
        informational: bool = False,
    ) -> None:
        check = Check(name, passed, detail, elapsed_ms, informational)
        self.checks.append(check)
        label = "INFO" if informational else ("PASS" if passed else "FAIL")
        print(f"[{label}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed and not check.informational]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: float) -> HttpResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            return HttpResult(response.status, json.loads(raw), elapsed_ms)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            body: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = raw.decode("utf-8", errors="replace")
        return HttpResult(exc.code, body, elapsed_ms, str(exc))
    except Exception as exc:  # Transport failures are part of the result.
        elapsed_ms = (time.perf_counter() - started) * 1000
        return HttpResult(0, None, elapsed_ms, f"{type(exc).__name__}: {exc}")


def get_health(base_url: str, timeout: float = 5.0) -> HttpResult:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as response:
            raw = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                body: Any = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = raw.decode("utf-8", errors="replace")
            return HttpResult(response.status, body, elapsed_ms)
    except Exception as exc:
        return HttpResult(0, None, (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}")


def wait_for_server(process: subprocess.Popen[bytes], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"llama-server exited during startup with code {return_code}")
        result = get_health(base_url)
        if result.status == 200:
            return
        last_error = result.error or f"HTTP {result.status}"
        time.sleep(1)
    raise TimeoutError(f"llama-server startup timed out: {last_error}")


def free_port(host: str, preferred: int) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])


@contextlib.contextmanager
def server_process(args: argparse.Namespace, log_path: pathlib.Path) -> Iterable[tuple[subprocess.Popen[bytes], str, list[str]]]:
    host = "127.0.0.1"
    port = free_port(host, args.port)
    base_url = f"http://{host}:{port}"
    command = [
        str(args.server_exe),
        "-m",
        str(args.model),
        "--host",
        host,
        "--port",
        str(port),
        "--parallel",
        str(args.parallel),
        "--embedding" if args.mode == "embedding" else "--reranking",
    ]
    environment = os.environ.copy()
    if args.runtime_bin:
        environment["PATH"] = f"{args.runtime_bin}{os.pathsep}{environment.get('PATH', '')}"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=creationflags,
        )
        try:
            wait_for_server(process, base_url, args.startup_timeout)
            yield process, base_url, command
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine(left: list[float], right: list[float]) -> float:
    denominator = vector_norm(left) * vector_norm(right)
    if denominator == 0:
        return float("nan")
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator


def validate_embedding_response(result: HttpResult, expected_count: int) -> tuple[bool, str, list[list[float]]]:
    if result.status != 200:
        return False, f"HTTP {result.status}, error={result.error}, body={result.body!r}", []
    try:
        data = result.body["data"]
        vectors = [item["embedding"] for item in data]
    except (KeyError, TypeError) as exc:
        return False, f"invalid response shape: {exc}, body={result.body!r}", []
    if len(vectors) != expected_count:
        return False, f"expected {expected_count} vectors, got {len(vectors)}", vectors
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or next(iter(dimensions), 0) <= 1:
        return False, f"inconsistent or invalid dimensions: {sorted(dimensions)}", vectors
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        return False, "response contains non-finite values", vectors
    norms = [vector_norm(vector) for vector in vectors]
    if any(abs(norm - 1.0) > 1e-3 for norm in norms):
        return False, f"vectors are not normalized, norm range={min(norms):.6f}..{max(norms):.6f}", vectors
    return True, f"count={len(vectors)}, dimension={len(vectors[0])}, norm={min(norms):.6f}..{max(norms):.6f}", vectors


def embedding_inputs(count: int) -> list[str]:
    seeds = [
        "Machine learning systems learn patterns from data.",
        "A kitten is a young cat.",
        "数据库索引用于加速查询。",
        "Paris is the capital of France.",
        "稳定性测试不应导致服务退出。",
    ]
    return [f"{seeds[index % len(seeds)]} sample={index}" for index in range(count)]


def run_embedding(args: argparse.Namespace, base_url: str, validation: Validation) -> dict[str, Any]:
    dimensions: set[int] = set()
    latencies: list[float] = []

    for count in args.batches:
        result = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": embedding_inputs(count)}, args.request_timeout)
        passed, detail, vectors = validate_embedding_response(result, count)
        validation.add(f"embedding_batch_{count}", passed, detail, result.elapsed_ms)
        latencies.append(result.elapsed_ms)
        dimensions.update(len(vector) for vector in vectors)

    baseline = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": "deterministic embedding probe"}, args.request_timeout)
    repeated = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": "deterministic embedding probe"}, args.request_timeout)
    ok_left, _, left = validate_embedding_response(baseline, 1)
    ok_right, _, right = validate_embedding_response(repeated, 1)
    maximum_delta = float("inf")
    if ok_left and ok_right and len(left[0]) == len(right[0]):
        maximum_delta = max(abs(x - y) for x, y in zip(left[0], right[0], strict=True))
    validation.add("embedding_determinism", ok_left and ok_right and maximum_delta <= 1e-6, f"max_abs_delta={maximum_delta:.9g}")

    json_result = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": "base64 output probe"}, args.request_timeout)
    base64_result = post_json(
        base_url,
        "/v1/embeddings",
        {"model": args.model.name, "input": "base64 output probe", "encoding_format": "base64"},
        args.request_timeout,
    )
    base64_passed = False
    base64_detail = f"json_status={json_result.status}, base64_status={base64_result.status}"
    try:
        expected = json_result.body["data"][0]["embedding"]
        encoded = base64_result.body["data"][0]["embedding"]
        decoded = base64.b64decode(encoded, validate=True)
        actual = struct.unpack(f"<{len(decoded) // 4}f", decoded)
        delta = max(abs(x - y) for x, y in zip(expected, actual, strict=True))
        base64_passed = delta <= 1e-6
        base64_detail = f"dimension={len(actual)}, max_abs_delta={delta:.9g}"
    except Exception as exc:
        base64_detail = f"{base64_detail}, decode_error={type(exc).__name__}: {exc}"
    validation.add("embedding_base64", base64_passed, base64_detail)

    semantic = post_json(
        base_url,
        "/v1/embeddings",
        {"model": args.model.name, "input": ["A cat rests on the sofa.", "A kitten sleeps on a couch.", "A database query uses an index."]},
        args.request_timeout,
    )
    semantic_ok, _, semantic_vectors = validate_embedding_response(semantic, 3)
    if semantic_ok:
        related = cosine(semantic_vectors[0], semantic_vectors[1])
        unrelated = cosine(semantic_vectors[0], semantic_vectors[2])
        validation.add(
            "embedding_semantic_sanity",
            related > unrelated,
            f"related_cosine={related:.6f}, unrelated_cosine={unrelated:.6f}",
            informational=True,
        )

    invalid_results = [
        post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": None}, args.request_timeout),
        post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": {"bad": "type"}}, args.request_timeout),
    ]
    invalid_ok = all(result.status == 400 for result in invalid_results)
    validation.add("embedding_invalid_input", invalid_ok, f"statuses={[result.status for result in invalid_results]}")

    if not args.skip_overlong:
        too_long = post_json(
            base_url,
            "/v1/embeddings",
            {"model": args.model.name, "input": "oversized " * args.overlong_words},
            args.request_timeout,
        )
        recovery = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": "recovery probe"}, args.request_timeout)
        recovery_ok, recovery_detail, _ = validate_embedding_response(recovery, 1)
        validation.add(
            "embedding_overlong_recovery",
            too_long.status in {400, 413, 500} and recovery_ok,
            f"overlong_status={too_long.status}, recovery={recovery_detail}",
            too_long.elapsed_ms + recovery.elapsed_ms,
        )

    def concurrent_request(index: int) -> HttpResult:
        return post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": f"concurrent embedding {index}"}, args.request_timeout)

    concurrent_results = run_concurrent(concurrent_request, args.concurrency, args.concurrency_requests)
    concurrent_ok = all(result.status == 200 for result in concurrent_results)
    validation.add(
        "embedding_concurrency",
        concurrent_ok,
        status_summary(concurrent_results),
        sum(result.elapsed_ms for result in concurrent_results),
    )

    stress_failures = 0
    stress_started = time.perf_counter()
    for index in range(args.iterations):
        count = args.batches[index % len(args.batches)]
        result = post_json(base_url, "/v1/embeddings", {"model": args.model.name, "input": embedding_inputs(count)}, args.request_timeout)
        passed, _, vectors = validate_embedding_response(result, count)
        if not passed:
            stress_failures += 1
        dimensions.update(len(vector) for vector in vectors)
    stress_ms = (time.perf_counter() - stress_started) * 1000
    validation.add("embedding_stress", stress_failures == 0, f"requests={args.iterations}, failures={stress_failures}", stress_ms)

    return {"dimensions": sorted(dimensions), "batch_latencies_ms": latencies}


RERANK_DOCUMENTS = [
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. "
    "The term is commonly applied to artificial devices, such as those employing engines or motors, but also "
    "to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, "
    "and preferences. The ability to learn is possessed by humans, non-human animals, and some machines.",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study "
    "of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks "
    "without explicit instructions.",
    "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, "
    "de la gastronomie et de la culture.",
    "Potatoes can be roasted with olive oil, salt, and herbs.",
]

RERANK_QUERY = "Machine learning is"

RERANK_CHINESE_QUERY = "我想知道如果在试用期内因为个人原因离职，是否还能拿到当月的绩效奖金？"
RERANK_CHINESE_DOCUMENTS = [
    "公司规定，员工在转正后若因个人原因辞职，需提前三十天提交申请。年终奖金的评定将参考员工全年的绩效表现，并在每年三月统一发放。",
    "关于绩效奖金的说明：绩效奖金是根据员工月度KPI达成情况计算的。试用期员工薪资通常由基本工资和岗位津贴组成，不包含季度奖金。",
    "《员工手册》第15条：处于试用期的员工如主动提出离职，其当月结算薪资仅包含实际出勤的基本工资，不予核算及发放任何形式的月度绩效奖金或补贴。",
    "如果是因为公司组织架构调整导致的裁员，无论是试用期还是正式员工，公司都将按照劳动法规定支付补偿金，并全额发放当月已产生的绩效奖金。",
    "新入职员工在试用期内需完成三项岗前培训。培训通过后方可开启绩效考核流程，考核结果将作为后续转正谈薪的重要参考依据。",
]


def rerank_documents(count: int) -> tuple[list[str], int]:
    if count == 1:
        return [RERANK_DOCUMENTS[2]], 0
    if count == 2:
        return [RERANK_DOCUMENTS[4], RERANK_DOCUMENTS[2]], 1
    documents = RERANK_DOCUMENTS[: min(count, len(RERANK_DOCUMENTS))]
    while len(documents) < count:
        documents.append(f"Unrelated inventory record number {len(documents)} describes warehouse shelving.")
    return documents, 2


def validate_rerank_response(result: HttpResult, expected_count: int, expected_top: int) -> tuple[bool, str]:
    if result.status != 200:
        return False, f"HTTP {result.status}, error={result.error}, body={result.body!r}"
    try:
        results = result.body["results"]
        indices = [int(item["index"]) for item in results]
        scores = [float(item["relevance_score"]) for item in results]
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid response shape: {exc}, body={result.body!r}"
    if len(results) != expected_count:
        return False, f"expected {expected_count} results, got {len(results)}"
    if not all(math.isfinite(score) for score in scores):
        return False, "response contains non-finite scores"
    if any(left < right for left, right in zip(scores, scores[1:])):
        return False, "scores are not sorted in descending order"
    if indices[0] != expected_top:
        return False, f"expected top index {expected_top}, got {indices[0]}, indices={indices}"
    return True, f"count={len(results)}, top={indices[0]}, score_range={min(scores):.6f}..{max(scores):.6f}"


def run_rerank(args: argparse.Namespace, base_url: str, validation: Validation) -> dict[str, Any]:
    latencies: list[float] = []
    for count in args.batches:
        documents, expected_top = rerank_documents(count)
        result = post_json(
            base_url,
            "/rerank",
            {"model": args.model.name, "query": RERANK_QUERY, "documents": documents},
            args.request_timeout,
        )
        passed, detail = validate_rerank_response(result, count, expected_top)
        validation.add(f"rerank_batch_{count}", passed, detail, result.elapsed_ms)
        latencies.append(result.elapsed_ms)

    if args.chinese_regression:
        chinese = post_json(
            base_url,
            "/rerank",
            {"model": args.model.name, "query": RERANK_CHINESE_QUERY, "documents": RERANK_CHINESE_DOCUMENTS},
            args.request_timeout,
        )
        chinese_ok = False
        chinese_detail = f"status={chinese.status}"
        if chinese.status == 200:
            try:
                results = chinese.body["results"]
                chinese_ok = len(results) == 5 and results[0]["index"] == 2 and results[-1]["index"] == 4
                chinese_detail = f"top={results[0]['index']}, bottom={results[-1]['index']}"
            except (KeyError, TypeError, IndexError) as exc:
                chinese_detail = f"invalid response: {exc}"
        validation.add("rerank_original_chinese_regression", chinese_ok, chinese_detail, chinese.elapsed_ms)

    documents, expected_top = rerank_documents(5)
    top_n = post_json(
        base_url,
        "/rerank",
        {"model": args.model.name, "query": RERANK_QUERY, "documents": documents, "top_n": 2},
        args.request_timeout,
    )
    top_n_ok = top_n.status == 200 and len(top_n.body.get("results", [])) == 2 and top_n.body["results"][0]["index"] == expected_top
    validation.add("rerank_top_n", top_n_ok, f"status={top_n.status}, result_count={len(top_n.body.get('results', [])) if isinstance(top_n.body, dict) else 0}")

    tei = post_json(
        base_url,
        "/rerank",
        {"model": args.model.name, "query": RERANK_QUERY, "texts": documents, "top_n": 3},
        args.request_timeout,
    )
    tei_ok = False
    tei_detail = f"status={tei.status}"
    if tei.status == 200 and isinstance(tei.body, list):
        try:
            scores = [float(item["score"]) for item in tei.body]
            tei_ok = len(tei.body) == 3 and tei.body[0]["index"] == expected_top and all(
                left >= right for left, right in zip(scores, scores[1:])
            )
            tei_detail = f"count={len(tei.body)}, top={tei.body[0]['index']}"
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            tei_detail = f"invalid TEI response: {exc}"
    validation.add("rerank_tei", tei_ok, tei_detail, tei.elapsed_ms)

    usage_ok = False
    usage_detail = f"status={top_n.status}"
    if top_n.status == 200:
        usage = top_n.body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        total_tokens = usage.get("total_tokens")
        usage_ok = isinstance(prompt_tokens, int) and prompt_tokens > 0 and prompt_tokens == total_tokens
        usage_detail = f"prompt_tokens={prompt_tokens}, total_tokens={total_tokens}"
    validation.add("rerank_usage", usage_ok, usage_detail)

    invalid_payloads = [
        {"query": "test", "documents": []},
        {"query": "test", "documents": None},
        {"query": "test", "documents": 123},
        {"query": "test", "documents": [1, 2, 3]},
        {"query": None, "documents": ["document"]},
    ]
    invalid_results = [post_json(base_url, "/rerank", payload, args.request_timeout) for payload in invalid_payloads]
    invalid_ok = all(result.status == 400 for result in invalid_results)
    validation.add("rerank_invalid_input", invalid_ok, f"statuses={[result.status for result in invalid_results]}")

    if not args.skip_overlong:
        too_long = post_json(
            base_url,
            "/rerank",
            {"query": "oversized " * args.overlong_words, "documents": ["normal document"]},
            args.request_timeout,
        )
        recovery = post_json(base_url, "/rerank", {"query": RERANK_QUERY, "documents": documents}, args.request_timeout)
        recovery_ok, recovery_detail = validate_rerank_response(recovery, len(documents), expected_top)
        validation.add(
            "rerank_overlong_recovery",
            too_long.status in {400, 413, 500} and recovery_ok,
            f"overlong_status={too_long.status}, recovery={recovery_detail}",
            too_long.elapsed_ms + recovery.elapsed_ms,
        )

    def concurrent_request(index: int) -> HttpResult:
        current_documents, _ = rerank_documents(5)
        return post_json(
            base_url,
            "/rerank",
            {"query": f"{RERANK_QUERY} request {index}", "documents": current_documents},
            args.request_timeout,
        )

    concurrent_results = run_concurrent(concurrent_request, args.concurrency, args.concurrency_requests)
    concurrent_ok = all(result.status == 200 for result in concurrent_results)
    validation.add("rerank_concurrency", concurrent_ok, status_summary(concurrent_results), sum(item.elapsed_ms for item in concurrent_results))

    stress_failures = 0
    stress_started = time.perf_counter()
    for index in range(args.iterations):
        count = args.batches[index % len(args.batches)]
        current_documents, current_top = rerank_documents(count)
        result = post_json(
            base_url,
            "/rerank",
            {"query": RERANK_QUERY, "documents": current_documents, "top_n": count},
            args.request_timeout,
        )
        passed, _ = validate_rerank_response(result, count, current_top)
        if not passed:
            stress_failures += 1
    stress_ms = (time.perf_counter() - stress_started) * 1000
    validation.add("rerank_stress", stress_failures == 0, f"requests={args.iterations}, failures={stress_failures}", stress_ms)

    return {"batch_latencies_ms": latencies}


def run_concurrent(function: Callable[[int], HttpResult], workers: int, requests: int) -> list[HttpResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, range(requests)))


def status_summary(results: list[HttpResult]) -> str:
    counts: dict[int, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return f"requests={len(results)}, statuses={dict(sorted(counts.items()))}"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exe", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-bin", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("embedding", "rerank"), required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--batches", type=parse_int_list, default=[1, 2, 5])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--concurrency-requests", type=int, default=8)
    parser.add_argument("--overlong-words", type=int, default=12000)
    parser.add_argument("--skip-overlong", action="store_true")
    parser.add_argument("--chinese-regression", action="store_true")
    parser.add_argument("--port", type=int, default=7901)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--skip-model-hash", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    validation = Validation()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result: dict[str, Any] = {
        "started_at": started_at,
        "mode": args.mode,
        "model": str(args.model.resolve()),
        "model_size": args.model.stat().st_size,
        "model_sha256": None if args.skip_model_hash else sha256_file(args.model),
        "parallel": args.parallel,
        "batches": args.batches,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "concurrency_requests": args.concurrency_requests,
        "server_log": str(args.log.resolve()),
    }
    exit_code = 1
    try:
        with server_process(args, args.log) as (process, base_url, command):
            result["server_command"] = command
            result["base_url"] = base_url
            result["server_pid"] = process.pid
            mode_data = run_embedding(args, base_url, validation) if args.mode == "embedding" else run_rerank(args, base_url, validation)
            result.update(mode_data)
            health = get_health(base_url)
            validation.add("final_server_health", health.status == 200 and process.poll() is None, f"status={health.status}, process_exit={process.poll()}")
            result["server_exit_during_test"] = process.poll()
        exit_code = 0 if not validation.failed else 1
    except Exception as exc:
        validation.add("validation_runtime", False, f"{type(exc).__name__}: {exc}")
        result["runtime_error"] = f"{type(exc).__name__}: {exc}"

    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["checks"] = [dataclasses.asdict(check) for check in validation.checks]
    result["summary"] = {
        "passed": sum(check.passed for check in validation.checks if not check.informational),
        "failed": len(validation.failed),
        "informational": sum(check.informational for check in validation.checks),
        "overall_passed": not validation.failed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
