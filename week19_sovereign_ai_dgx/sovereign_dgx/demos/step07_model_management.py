#!/usr/bin/env python3
"""PART 7 · Model management & NVFP4 quantization  [ADVANCED]

A DGX is a model fleet, not one model. This demo covers the management story:
list what's resident, do the VRAM math for each quant format, and quantize a
model to NVFP4 (Blackwell's native 4-bit float) to make a big model fit. The
formula is simple: VRAM ≈ params × bytes/param × overhead.

Run:  python demos/step07_model_management.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxsim  # noqa: E402
import dgxview  # noqa: E402

FORMATS = [
    ("BF16",   2.0,  "baseline, training-grade"),
    ("FP8",    1.0,  "2x smaller, ~lossless on Hopper/Blackwell"),
    ("NVFP4",  0.55, "Blackwell-native 4-bit float, ~4x smaller"),
    ("Q4_K_M", 0.55, "llama.cpp 4-bit, the GGUF edge default"),
    ("Q2_K",   0.33, "2-bit, last-resort squeeze"),
]

QUANT_CMD = """\
# DGX — quantize a model to NVFP4 with TensorRT Model Optimizer (≈45-90 min/8B)
docker run --rm -it --gpus all --ipc host \\
  -v "$PWD/out:/workspace/out" -e HF_TOKEN=$HF_TOKEN \\
  nvcr.io/nvidia/vllm:26.01-py3 bash -c '
    git clone -b 0.41.0 https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
    cd TensorRT-Model-Optimizer && pip install -e ".[dev]"
    examples/llm_ptq/scripts/huggingface_example.sh \\
      --model meta-llama/Llama-3.3-70B-Instruct --quant nvfp4 \\
      --output /workspace/out
  '
# then just: vllm serve /workspace/out   (NVFP4 kernels are native on Blackwell)
"""


def main() -> None:
    dgxview.banner("PART 7", "Model management & NVFP4 quantization", "ADVANCED")
    dgxview.mode_line()

    installed = config.list_local_models() if not dgxview.is_sim() else dgxsim.installed_models()
    print("Models resident on this DGX (your fleet):")
    for m in installed or ["(none)"]:
        print(f"  • {m}")
    print()

    print("The VRAM formula:  bytes ≈ params × bytes/param × 1.18 (runtime overhead)\n")
    print(f"  {'Format':<9}{'B/param':>8}   a 70B model needs   note")
    print("  " + "─" * 70)
    for fmt, bpp, note in FORMATS:
        gb = 70 * bpp * 1.18
        fits = "✓ fits Spark" if gb < 118 else "✗ needs Station / 2 Sparks"
        print(f"  {fmt:<9}{bpp:>8}   {gb:>6.0f} GB  {fits:<26} {note}")
    print()

    print("So Llama-3.3-70B is 165 GB in BF16 (won't fit a Spark) but ~45 GB at NVFP4")
    print("(fits comfortably, fast native kernels). Quantization is what makes the")
    print("70B class a sovereign, on-desk model. The recipe:\n")
    print(QUANT_CMD)

    print("Management tips from the playbooks:")
    print("  • Ollama: `ollama list` / `ollama rm` to manage the library; weights live")
    print("    in ~/.ollama/models and are shared across runs.")
    print("  • Use llama-swap / LiteLLM to hot-swap models behind ONE stable URL so")
    print("    apps don't care which model is currently resident in VRAM.")
    print("  • Pin a model per task; keep a small fast model (phi3.5) warm for routing.")

    print("\nTakeaway: managing a DGX = fitting the RIGHT quant of the RIGHT model in")
    print("128 GB. NVFP4 is the lever that brings 70B-class models on-desk.")


if __name__ == "__main__":
    main()
