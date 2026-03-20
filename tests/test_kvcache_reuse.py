#!/usr/bin/env python3
"""
KV Cache + State Cache Reuse Test Suite
========================================
Tests prefix cache reuse on an OpenAI-compatible endpoint
(llama-server with HouMo Qwen3.5).

Key design: all prompts are >256 tokens so that a state checkpoint at
position 256 is always saved during the first request.  Subsequent
requests sharing the same prefix will have kv_offset > 256, enabling
checkpoint restoration.

Disables thinking mode via chat_template_kwargs.
Uses TTFT (streaming) as the primary metric — shorter TTFT on later
turns indicates successful prefix reuse.  Server-side logs provide
the definitive evidence (checkpoint restore, rollback, etc.).

Usage:
    python test_kvcache_reuse.py [--base-url http://10.64.33.117:17701]
    python test_kvcache_reuse.py --test 2   # run only test 2
"""

import argparse
import json
import time
import requests
import sys

BASE_URL = "http://10.64.33.117:17701"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat_completion(messages, max_tokens=64):
    """Send a streaming request, measure TTFT and collect response."""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "qwen3.5",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t_start = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    full_content = ""

    resp = requests.post(url, json=payload, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_tokens += 1
            full_content += content

    t_end = time.perf_counter()
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else None
    total_ms = (t_end - t_start) * 1000

    return {
        "content": full_content,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tps": output_tokens / (total_ms / 1000) if total_ms > 0 else 0,
    }


def print_result(label, r, show_full=False):
    ttft = f"{r['ttft_ms']:.0f}" if r['ttft_ms'] else "N/A"
    print(f"  [{label}]")
    print(f"    TTFT={ttft:>6s} ms | "
          f"Tokens={r['output_tokens']:>4d} | "
          f"Total={r['total_ms']:.0f} ms | "
          f"TPS={r['tps']:.1f}")
    if r["content"]:
        if show_full:
            print(f"    Reply: {r['content']}")
        else:
            print(f"    Reply: {r['content'][:120]}...")
    print()


def separator(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# System prompt (~400+ tokens to guarantee checkpoint at pos=256)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in software engineering. "
    "You provide concise, accurate answers about programming, algorithms, "
    "data structures, system design, and best practices. "
    "Always think step by step and provide code examples when appropriate. "
    "You are running on a HouMo accelerator with hybrid Mamba-Transformer "
    "architecture (Qwen3.5). Your responses should be professional and "
    "well-structured.\n\n"
    "Here is your reference knowledge base:\n\n"
    "### Binary Search\n"
    "Binary search finds items in sorted arrays by repeatedly halving the "
    "search interval. Time complexity is O(log n). It requires the input to "
    "be sorted. Compare the target to the middle element, then narrow to "
    "the left or right half. Variants include lower_bound, upper_bound, "
    "and exponential search.\n\n"
    "### Hash Tables\n"
    "Hash tables are associative arrays that use hash functions for O(1) "
    "average-case lookup. Collisions are handled by chaining (linked lists) "
    "or open addressing (linear probing, quadratic probing, double hashing). "
    "Load factor affects performance; rehashing occurs when it exceeds a "
    "threshold.\n\n"
    "### Graph Algorithms\n"
    "BFS uses a queue for level-order traversal, DFS uses a stack (or "
    "recursion). Dijkstra's algorithm finds shortest paths with non-negative "
    "weights. Bellman-Ford handles negative edges. Floyd-Warshall computes "
    "all-pairs shortest paths. Topological sort works on DAGs.\n\n"
    "### Dynamic Programming\n"
    "DP applies when problems have overlapping subproblems and optimal "
    "substructure. Two approaches: memoization (top-down with caching) and "
    "tabulation (bottom-up iteration). Classic examples include the knapsack "
    "problem, longest common subsequence, edit distance, and coin change.\n\n"
    "### Sorting Algorithms\n"
    "Quicksort averages O(n log n) but worst-case O(n^2). Mergesort is "
    "O(n log n) guaranteed and stable. Heapsort is O(n log n) in-place. "
    "Counting sort and radix sort achieve O(n) for integers with bounded "
    "range. Python uses Timsort, a hybrid of mergesort and insertion sort.\n\n"
    "### Trees and Balanced BSTs\n"
    "BST operations are O(log n) when balanced. AVL trees enforce strict "
    "balance (height difference <= 1). Red-black trees use a relaxed "
    "balance invariant with O(1) rotations per insert. B-trees are "
    "disk-optimized and used in databases and filesystems.\n\n"
    "Use the above knowledge to answer questions accurately."
)


# ---------------------------------------------------------------------------
# Test 1: Baseline (cold start with long prompt)
# ---------------------------------------------------------------------------

def test_1_baseline():
    separator("Test 1: Baseline (cold start, long system prompt)")
    print("  Single request with long system prompt (~400+ tokens).")
    print("  Establishes checkpoint at pos=256. No reuse expected.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a binary search tree? Explain briefly."},
    ]
    r = chat_completion(messages, max_tokens=100)
    print_result("Cold start (long prompt)", r, show_full=True)
    return r


# ---------------------------------------------------------------------------
# Test 2: Same long system prompt, different queries (checkpoint reuse)
# ---------------------------------------------------------------------------

def test_2_same_system():
    separator("Test 2: Same system prompt, different queries")
    print("  3 independent requests with the SAME long system prompt.")
    print("  Request 1 creates checkpoint at pos=256.")
    print("  Requests 2-3 should find checkpoint at 256 -> much lower TTFT.")
    print("  Also verifies response CORRECTNESS under reuse.\n")

    queries = [
        ("What is the time complexity of Dijkstra's algorithm?",
         "Answer in one sentence."),
        ("Compare quicksort and mergesort.",
         "Which is faster on average and why?"),
        ("Explain the difference between BFS and DFS.",
         "Give one use-case for each."),
    ]

    results = []
    for i, (q, hint) in enumerate(queries):
        full_q = f"{q} {hint}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_q},
        ]
        r = chat_completion(messages, max_tokens=150)
        print_result(f"Query {i+1}: \"{q[:60]}\"", r, show_full=True)
        results.append(r)
        time.sleep(0.5)

    print("  --- TTFT Comparison ---")
    for i, r in enumerate(results):
        ttft = r["ttft_ms"]
        speedup = ""
        if i > 0 and results[0]["ttft_ms"] and ttft:
            ratio = results[0]["ttft_ms"] / ttft
            speedup = f"  ({ratio:.2f}x vs query 1)"
        ttft_s = f"{ttft:.0f}" if ttft else "N/A"
        status = "REUSED ✓" if (i > 0 and ttft and results[0]["ttft_ms"]
                                 and ttft < results[0]["ttft_ms"] * 0.7) else ""
        print(f"  Query {i+1}: TTFT = {ttft_s:>6s} ms{speedup}  {status}")
    print()
    return results


# ---------------------------------------------------------------------------
# Test 3: Multi-turn conversation (growing context with reuse)
# ---------------------------------------------------------------------------

def test_3_multiturn():
    separator("Test 3: Multi-turn conversation (growing prefix reuse)")
    print("  3 turns with same long system prompt. Context grows each turn.")
    print("  Turn 1 is cold. Turns 2-3 reuse checkpoint at 256.\n")

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    queries = [
        "What is the worst-case time complexity of BST search? One sentence.",
        "How does an AVL tree fix the worst case? One sentence.",
        "What about Red-Black trees compared to AVL? Brief comparison.",
    ]

    results = []
    for i, query in enumerate(queries):
        conversation.append({"role": "user", "content": query})
        r = chat_completion(conversation, max_tokens=150)
        print_result(f"Turn {i+1}: \"{query[:55]}\"", r, show_full=True)

        reply = r["content"] or "(empty)"
        conversation.append({"role": "assistant", "content": reply})
        results.append(r)
        time.sleep(0.5)

    print("  --- TTFT Comparison ---")
    for i, r in enumerate(results):
        ttft = r["ttft_ms"]
        speedup = ""
        if i > 0 and results[0]["ttft_ms"] and ttft:
            ratio = results[0]["ttft_ms"] / ttft
            speedup = f"  ({ratio:.2f}x vs turn 1)"
        ttft_s = f"{ttft:.0f}" if ttft else "N/A"
        status = "REUSED ✓" if (i > 0 and ttft and results[0]["ttft_ms"]
                                 and ttft < results[0]["ttft_ms"] * 0.7) else ""
        print(f"  Turn {i+1}: TTFT = {ttft_s:>6s} ms{speedup}  {status}")
    print()
    return results


# ---------------------------------------------------------------------------
# Test 4: Correctness verification (same question twice)
# ---------------------------------------------------------------------------

def test_4_correctness():
    separator("Test 4: Correctness (same question twice, verify consistency)")
    print("  Send the exact same request twice. Second should be faster")
    print("  AND produce a consistent answer (proving reuse is correct).\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
         "List exactly 3 advantages of mergesort over quicksort. "
         "Number them 1, 2, 3."},
    ]

    r1 = chat_completion(messages, max_tokens=200)
    print_result("Request 1 (cold)", r1, show_full=True)
    time.sleep(0.5)

    r2 = chat_completion(messages, max_tokens=200)
    print_result("Request 2 (should reuse)", r2, show_full=True)

    print("  --- Comparison ---")
    t1 = r1["ttft_ms"] or 0
    t2 = r2["ttft_ms"] or 0
    ratio = t1 / t2 if t2 > 0 else 0
    reused = t2 < t1 * 0.7
    print(f"  TTFT: {t1:.0f} ms -> {t2:.0f} ms  ({ratio:.2f}x)")
    print(f"  Reuse detected: {'YES ✓' if reused else 'NO ✗'}")
    # Simple content consistency check (temperature=0 → should be similar)
    overlap = set(r1["content"].split()) & set(r2["content"].split())
    sim = len(overlap) / max(len(set(r1["content"].split())), 1)
    print(f"  Content similarity: {sim:.0%}")
    print(f"  Content match: {'PASS ✓' if sim > 0.5 else 'WARN ⚠'}")
    print()
    return [r1, r2]


# ---------------------------------------------------------------------------
# Test 5: No reuse baseline (different system prompt)
# ---------------------------------------------------------------------------

def test_5_no_reuse():
    separator("Test 5: No reuse baseline (completely different prompt)")
    print("  Different, shorter system prompt -> no prefix shared.")
    print("  Serves as a control: TTFT should be similar to cold start.\n")

    # Use a long enough but DIFFERENT system prompt
    alt_system = (
        "You are a world-class chef specializing in French cuisine. "
        "You have studied at Le Cordon Bleu and worked at Michelin-starred "
        "restaurants for 20 years. You know every classic French technique "
        "from sous vide to flambé. You are passionate about fresh, seasonal "
        "ingredients and believe that simplicity is the ultimate sophistication. "
        "Your expertise covers sauces (béchamel, hollandaise, velouté, "
        "espagnole, tomato), pastry (croissants, éclairs, tarte tatin, "
        "mille-feuille, soufflé), and main courses (coq au vin, bouillabaisse, "
        "duck confit, beef bourguignon, ratatouille). You can explain complex "
        "techniques in simple terms and always provide measurements in both "
        "metric and imperial units. When answering, share the history and "
        "cultural significance of each dish. You believe cooking is both an "
        "art and a science, and you enjoy discussing the Maillard reaction, "
        "emulsification, and fermentation. Always end with a wine pairing "
        "suggestion."
    )
    messages = [
        {"role": "system", "content": alt_system},
        {"role": "user", "content": "How do you make a perfect hollandaise sauce?"},
    ]
    r = chat_completion(messages, max_tokens=150)
    print_result("Different system prompt (no reuse)", r, show_full=True)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="KV Cache Reuse Test Suite")
    parser.add_argument("--base-url", default=BASE_URL,
                        help=f"API base URL (default: {BASE_URL})")
    parser.add_argument("--test", type=int, default=0,
                        help="Run specific test (1-5), 0 = all")
    args = parser.parse_args()

    BASE_URL = args.base_url

    print(f"Target: {BASE_URL}/v1/chat/completions")
    print(f"Testing KV cache + state cache checkpoint reuse")
    print(f"Thinking mode: disabled")
    print(f"System prompt: ~400+ tokens (ensures checkpoint at pos=256)\n")

    # Health check
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {resp.status_code} {resp.text.strip()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    print()

    tests = {
        1: ("Baseline (cold)", test_1_baseline),
        2: ("Same sys reuse", test_2_same_system),
        3: ("Multi-turn", test_3_multiturn),
        4: ("Correctness", test_4_correctness),
        5: ("No reuse ctrl", test_5_no_reuse),
    }

    if args.test > 0:
        if args.test in tests:
            _, func = tests[args.test]
            func()
        else:
            print(f"Unknown test: {args.test}")
            sys.exit(1)
    else:
        for idx in sorted(tests):
            name, func = tests[idx]
            try:
                func()
            except Exception as e:
                print(f"  *** Test {idx} ({name}) failed: {e}\n")

    separator("Expected server log evidence")
    print("  Checkpoint saved:  'save_state_checkpoint: saved ... pos=256'")
    print("  Checkpoint reused: 'restored checkpoint ... pos=256'")
    print("  State rollback:    'rollback ... best_ckpt=256, kv_offset=N'")
    print("  Continuation:      'continuation batch kv_offset=N'")
    print("  No reuse:          'best_ckpt=-1 ... effective_kv_offset=0'")
    print()
    print("  Key metric: TTFT should drop ~2-3x when checkpoint is reused.")
    print()


if __name__ == "__main__":
    main()
