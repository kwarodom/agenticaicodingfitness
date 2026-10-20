#!/usr/bin/env python3
"""STEP 3 · 🖥️ Prepare the DGX (pull container, check GPU/mem) — REAL SSH  [INTERMEDIATE]

Pulls NVIDIA's PyTorch container (the right CUDA for Blackwell) and reports GPU
memory, so the real training in STEP 4 starts fast. Runs entirely on the DGX over
SSH; output streams back here.

Run:  python demos/step03_prepare_dgx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ftssh  # noqa: E402

CONTAINER = "nvcr.io/nvidia/pytorch:25.11-py3"


def main() -> None:
    print("━" * 64)
    print("  STEP 3 · 🖥️ Prepare the DGX (container + GPU check)")
    print("━" * 64, "\n")
    if not ftssh.require():
        return

    print("1) GPU + free memory on the DGX:")
    ftssh.run("nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used "
              "--format=csv,noheader || nvidia-smi -L")

    print("\n2) Docker present?")
    ftssh.run("docker --version || echo 'docker missing — install it or use a venv'")

    print(f"\n3) Pulling the training container (first time is slow): {CONTAINER}")
    rc = ftssh.run(f"docker pull {CONTAINER}")
    if rc == 0:
        print("\n✓ container ready.")

    print("\nNotes:")
    print("  • This container has the correct CUDA/PyTorch for GB10 (Blackwell).")
    print("  • Make sure the model you'll train fits free VRAM (stop other models:")
    print("    `ollama stop <model>` / free the GPU).")

    print("\nTakeaway: the DGX is primed. STEP 4 runs the REAL Unsloth fine-tune on")
    print("your chosen model and streams the loss curve here.")


if __name__ == "__main__":
    main()
