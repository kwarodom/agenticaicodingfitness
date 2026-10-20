#!/usr/bin/env python3
"""PART 3 · Ollama on DGX — the one-command path  [BEGINNER]

Ollama is the fastest way to get a sovereign model serving on a DGX Spark: one
install, `ollama pull`, and you have an OpenAI-compatible endpoint on
localhost:11434. It auto-detects the GB10 GPU, manages a model library, and is
the backend the NVIDIA "local coding agent" playbook uses.

This demo prints the exact DGX Spark commands, shows the GPU before/after the
model loads, then makes a real (or simulated) streaming call.

Run:  python demos/step03_ollama_on_dgx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxview  # noqa: E402

COMMANDS = """\
# 1) install (Ubuntu / DGX OS — needs Ollama 0.15+ for Blackwell CUDA)
curl -fsSL https://ollama.com/install.sh | sh

# 2) pull a model that suits the Spark's 128 GB (qwen3.6 MoE is a great default)
ollama pull qwen3.6:35b-a3b-q8_0      # 35B MoE, ~3B active/token, fast on GB10

# 3) it's already serving an OpenAI-compatible API on :11434 — just call it
curl http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"qwen3.6:35b-a3b-q8_0",
       "messages":[{"role":"user","content":"What is NVFP4?"}]}'

# 4) bigger context for agents / long codebases
ollama run qwen3.6:35b-a3b-q8_0
>>> /set parameter num_ctx 40960
"""


def main() -> None:
    dgxview.banner("PART 3", "Ollama on DGX — the one-command path", "BEGINNER")
    dgxview.mode_line()

    print("The whole Ollama-on-DGX workflow:\n")
    print(COMMANDS)

    print("GPU before the model is resident:")
    dgxview.show_gpu(model=None, busy=False)
    print()
    print(f"…now serving {config.MODEL}. GPU with weights loaded:")
    dgxview.show_gpu(model=config.MODEL, busy=True)
    print()

    dgxview.require_runtime()
    dgxview.sovereignty_line()
    dgxview.generate(
        "Give me a one-paragraph plan to fine-tune you on our HVAC manuals.",
        max_tokens=400,
        title="ollama serving on the DGX",
    )

    print("\nWhen to choose Ollama: single user, quick start, frequent model swaps.")
    print("Step into vLLM (next) when you need many concurrent users / throughput.")


if __name__ == "__main__":
    main()
