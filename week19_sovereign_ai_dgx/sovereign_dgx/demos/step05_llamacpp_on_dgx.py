#!/usr/bin/env python3
"""PART 5 · llama.cpp on DGX — GGUF, native CUDA, total control  [INTERMEDIATE]

llama.cpp is the lightweight C/C++ engine behind Ollama. On a DGX you build it
with CUDA for the Blackwell GPU (sm_121), run any GGUF weight, and get a tiny
OpenAI-compatible server with no Python runtime. It's the choice when you want
maximum control, minimal dependencies, or to run on CPU/edge alongside the DGX.

This demo prints the real build + serve commands, explains GGUF + Q4_K_M, and
makes a real/simulated call against the server.

Run:  python demos/step05_llamacpp_on_dgx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxview  # noqa: E402

COMMANDS = """\
# DGX Spark — build llama.cpp with CUDA for Blackwell (sm_121)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build --config Release -j

# grab a GGUF weight (Q4_K_M = the edge sweet spot: ~4.5 bits, <2% quality loss)
huggingface-cli download bartowski/Qwen3.6-35B-GGUF \\
  qwen3.6-35b-a3b-Q4_K_M.gguf --local-dir models/

# serve an OpenAI-compatible endpoint on :8080, all layers on the GPU
./build/bin/llama-server \\
  -m models/qwen3.6-35b-a3b-Q4_K_M.gguf \\
  --n-gpu-layers 999 --ctx-size 40960 --host 0.0.0.0 --port 8080

export DGX_BASE_URL=http://localhost:8080/v1   # point your app here
"""


def main() -> None:
    dgxview.banner("PART 5", "llama.cpp on DGX — GGUF + native CUDA", "INTERMEDIATE")
    dgxview.mode_line()

    print("Build and serve with llama.cpp on the DGX:\n")
    print(COMMANDS)

    print("GGUF quantization formats you'll meet (single-file, mmap-able weights):")
    print(f"  {'Format':<10}{'bits/wt':>8}{'quality':>10}  use when")
    print("  " + "─" * 56)
    for fmt, bits, q, use in [
        ("Q8_0",   "8.0",  "~lossless", "you have memory to spare"),
        ("Q5_K_M", "5.5",  "excellent", "best quality that still fits"),
        ("Q4_K_M", "4.5",  "very good", "the default edge sweet spot"),
        ("Q3_K_M", "3.4",  "okay",      "squeeze a bigger model in"),
        ("Q2_K",   "2.6",  "rough",     "last resort on tiny RAM"),
    ]:
        print(f"  {fmt:<10}{bits:>8}{q:>10}  {use}")
    print()

    dgxview.require_runtime()
    dgxview.sovereignty_line()
    dgxview.generate(
        "What is GGUF and why is a single-file weight format handy on the edge?",
        max_tokens=380,
        title="llama.cpp serving on the DGX",
    )

    print("\nTakeaway: llama.cpp = no Python, one binary, any GGUF, full control —")
    print("the same engine Ollama wraps, exposed raw. Great for edge + air-gap.")


if __name__ == "__main__":
    main()
