#!/usr/bin/env python3
"""PART 6 · Runtime bake-off — Ollama vs vLLM vs llama.cpp  [INTERMEDIATE]

Three sovereign runtimes, same OpenAI API, different sweet spots. This demo lays
out the decision table, then (REAL mode) runs the SAME prompt through the live
endpoint to show real tok/s — or (SIM mode) prints representative DGX numbers.

Run:  python demos/step06_runtime_bakeoff.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxsim  # noqa: E402
import dgxview  # noqa: E402

TABLE = [
    ("Ollama",     "single user, quick start, model swaps", "auto-GPU, model library, easy",  "no batching, single-stream"),
    ("vLLM",       "2+ concurrent users, production API",    "PagedAttention, high throughput", "heavier, Python, more VRAM"),
    ("llama.cpp",  "control / minimal deps / CPU+edge",      "one binary, any GGUF, mmap",      "you manage builds + flags"),
    ("TensorRT-LLM","max single-GPU + multi-Spark perf",     "fused kernels, TP/PP, NVFP4",     "compile step, NVIDIA-specific"),
]


def _bench(model: str, prompt: str) -> float:
    client = dgxview._client()
    start = time.time(); chars = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.3, stream=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        chars += len((d.content or "") + (dgxview._extract_reasoning(d) or ""))
    elapsed = time.time() - start
    return max(1, round(chars / 4)) / elapsed


def main() -> None:
    dgxview.banner("PART 6", "Runtime bake-off — pick the right server", "INTERMEDIATE")
    dgxview.mode_line()

    print(f"  {'Runtime':<14}{'Use when':<40}{'Strength':<34}weakness")
    print("  " + "─" * 118)
    for rt, when, strong, weak in TABLE:
        print(f"  {rt:<14}{when:<40}{strong:<34}{weak}")
    print()

    print("All four expose the SAME OpenAI API, so your app code never changes —")
    print("only the base_url and the ops trade-offs differ.\n")

    prompt = "In one sentence, what makes local inference 'sovereign'?"
    if dgxview.is_sim():
        print("SIM mode — representative single-stream decode on a DGX Spark:")
        for rt, model in [("Ollama", "qwen3.6:35b-a3b-q8_0"),
                          ("vLLM (NVFP4)", "qwen3.6:35b-a3b-nvfp4"),
                          ("llama.cpp (Q4_K_M)", "llama3.3:70b")]:
            print(f"  ◆ {rt:<22} ~{dgxsim.spec_for(model).tok_s:.0f} tok/s")
        print("\n  (vLLM's real edge is AGGREGATE throughput under concurrency, not")
        print("   single-stream — see Part 4.)")
    else:
        dgxview.sovereignty_line()
        print(f"Running the live endpoint ({config.MODEL}) once for a real number:\n")
        tps = _bench(config.MODEL, prompt)
        print(f"  ◆ {config.BASE_URL}  →  ~{tps:.1f} tok/s")

    print("\nTakeaway: start with Ollama; graduate to vLLM for concurrency or")
    print("TensorRT-LLM for peak perf; reach for llama.cpp when you want raw control.")


if __name__ == "__main__":
    main()
