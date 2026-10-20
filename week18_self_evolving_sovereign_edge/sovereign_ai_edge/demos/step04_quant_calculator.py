#!/usr/bin/env python3
"""PART 5 · Quantization & the memory math that decides everything  [INTERMEDIATE]

Raw Hugging Face weights ship in BF16 — too big for most edge devices.
Quantization shrinks them to 4-bit/8-bit with minimal quality loss, and the
math is simple enough to do in your head once you've seen it:

    VRAM ≈ params(B) × bytes_per_param × overhead

This demo encodes the PDF's format table (BF16 → FP8 → NVFP4 → Q4_K_M → 2-bit),
computes the real memory footprint of each Gemma 4 size in each format, and then
INSPECTS the model you actually have running to print its true quantization
level — no guessing, straight from the local engine.

Run:  python demos/step04_quant_calculator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402

# (format, bits, relative_size_vs_bf16, quality_loss, best_runtime, best_hw)
FORMATS = [
    ("BF16",          16, 1.00,  "none (baseline)", "vLLM, Ollama",   "any modern GPU"),
    ("FP8",            8, 0.50,  "<1%",             "vLLM, TRT-LLM",  "Hopper/Blackwell"),
    ("NVFP4",          4, 0.25,  "<5%",             "vLLM + modelopt","Blackwell only"),
    ("Q8_0 (GGUF)",    8, 0.53,  "<1%",             "llama.cpp/Ollama","any GPU/CPU"),
    ("Q4_K_M (GGUF)",  4, 0.28,  "<5%",             "llama.cpp/Ollama","any GPU/CPU"),
    ("Q4 (LiteRT)",    4, 0.28,  "<5%",             "LiteRT-LM",      "Mobile/RPi"),
    ("2-bit (LiteRT)", 2, 0.14,  "moderate",        "LiteRT-LM",      "ultra-low-memory"),
]

GEMMA_SIZES = [("Gemma 4 E2B", 2.3), ("Gemma 4 E4B", 4.5),
               ("Gemma 4 26B MoE", 26.0), ("Gemma 4 31B Dense", 31.0)]


def vram_gb(params_b: float, bits: int, overhead: float = 1.20) -> float:
    """Approximate weight memory: params × (bits/8) bytes × KV/activation overhead."""
    return params_b * (bits / 8.0) * overhead


def inspect_local_quant(model: str) -> dict:
    """Ask the local engine for the real quantization of the running model."""
    try:
        req = Request(config._native_base() + "/show",
                      data=json.dumps({"model": model}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as r:
            d = json.loads(r.read().decode())
        det = d.get("details", {})
        return {"quant": det.get("quantization_level", "?"),
                "params": det.get("parameter_size", "?"),
                "family": det.get("family", "?"),
                "ctx": d.get("model_info", {}).get("gemma4.context_length")
                       or det.get("context_length", "?")}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    edgeview.banner("PART 5", "Quantization & memory math", "INTERMEDIATE")

    print("Quantization formats (size is relative to BF16):\n")
    print(f"  {'Format':<16} {'Bits':<5} {'Size':<7} {'Quality loss':<18} {'Runtime'}")
    print("  " + "─" * 70)
    for fmt, bits, size, loss, rt, _hw in FORMATS:
        print(f"  {fmt:<16} {bits:<5} {size:<7.2f} {loss:<18} {rt}")

    print("\nReal memory footprint by model × format (≈ weights + 20% overhead):\n")
    cols = ["BF16", "FP8/Q8", "Q4_K_M", "2-bit"]
    bits_for = {"BF16": 16, "FP8/Q8": 8, "Q4_K_M": 4, "2-bit": 2}
    print(f"  {'Model':<22} " + " ".join(f"{c:>9}" for c in cols))
    print("  " + "─" * 64)
    for name, params in GEMMA_SIZES:
        cells = " ".join(f"{vram_gb(params, bits_for[c]):>8.1f}G" for c in cols)
        print(f"  {name:<22} {cells}")
    print("\n  → A 31B model is 75 GB in BF16 but only ~14 GB at Q4_K_M: that 0.25×")
    print("    is what turns a data-center model into a single-GPU sovereign model.")

    print("\n  NVFP4 is Blackwell-exclusive (5th-gen Tensor Cores process 4-bit float")
    print("  natively): ~near-FP8 accuracy at 4-bit size. For Mac/Pi/RTX use Q4_K_M.")

    if not edgeview.require_local():
        return

    print(f"\nNow inspect the model actually running here ({config.MODEL}) — its")
    print("TRUE quantization, straight from the local engine:\n")
    info = inspect_local_quant(config.MODEL)
    if "error" in info:
        print(f"  (could not read model details: {info['error']})")
    else:
        print(f"  • parameters:   {info['params']}")
        print(f"  • quantization: {info['quant']}")
        print(f"  • family:       {info['family']}")
        print(f"  • context:      {info['ctx']}")
        # find params number
        try:
            pnum = float("".join(ch for ch in str(info["params"]) if ch.isdigit() or ch == "."))
            print(f"\n  Estimated footprint at {info['quant']}: "
                  f"~{vram_gb(pnum, 4):.1f} GB (matches what's loaded in memory).")
        except ValueError:
            pass

    print("\nTakeaway: quantization is the lever that makes sovereign edge AI possible.")
    print("Pick NVFP4 on Blackwell, Q4_K_M everywhere else — both lose <5% quality.")


if __name__ == "__main__":
    main()
