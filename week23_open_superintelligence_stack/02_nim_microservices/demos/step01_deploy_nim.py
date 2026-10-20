#!/usr/bin/env python3
"""PART 1 · Deploy a NIM — one container, sovereign inference  [BEGINNER]

A NIM (NVIDIA Inference Microservice) packages a model + an optimized inference
engine (auto-selected TensorRT-LLM / vLLM / SGLang) + an OpenAI-compatible API into
one signed container that runs on your DGX. One command → a production endpoint.

Run:  python demos/step01_deploy_nim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

DEPLOY = """\
# 🖥️ on the DGX — pull + run a Nemotron NIM (get NGC_API_KEY at build.nvidia.com)
export NGC_API_KEY=nvapi-...
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

docker run -d --name nim --gpus all --shm-size=16g \\
  -e NGC_API_KEY \\
  -v ~/.cache/nim:/opt/nim/.cache \\
  -p 8000:8000 \\
  nvcr.io/nim/nvidia/nemotron-3-super:latest
#  → OpenAI-compatible API on http://localhost:8000/v1  (auto-picks TensorRT-LLM)

# point this app at it (🔌 Connection → Local/Tunnel):  http://<dgx>:8000/v1
"""


def main() -> None:
    view.banner("PART 1", "Deploy a NIM — one container, sovereign inference", "BEGINNER")
    view.mode_line()

    print("What's inside a NIM (so you don't assemble it yourself):")
    print("  • the model weights (pre-optimized, TensorRT-engine-built)")
    print("  • an auto-selected backend — TensorRT-LLM, vLLM, or SGLang")
    print("  • an OpenAI-compatible API server (drop-in for your apps)")
    print("  • security scanning + signed container + enterprise support\n")

    print("Deploy it on the DGX:\n")
    print(DEPLOY)

    print("Why this is the SOVEREIGN path:")
    print("  • Runs entirely on your hardware / AI factory — no data leaves the perimeter.")
    print("  • One command, battle-tested config — ops doesn't tune vLLM internals.")
    print("  • Same OpenAI API as the cloud, so every app + agent works unchanged.\n")

    view.generate("In one sentence, what does a NIM package into a single container?",
                  max_tokens=200, title="a NIM on the DGX")

    print("\nTakeaway: a NIM is 'sovereign inference in one command'. Next: how it")
    print("compares to running raw vLLM/Ollama yourself.")


if __name__ == "__main__":
    main()
