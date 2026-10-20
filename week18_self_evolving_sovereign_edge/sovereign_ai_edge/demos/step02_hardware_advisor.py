#!/usr/bin/env python3
"""PART 3 · Edge hardware advisor — pick your sovereign platform  [BEGINNER]

There is no single "edge" device — there's a spectrum from a $80 Raspberry Pi to
a $4000 DGX Spark. The right choice is driven by ONE number more than any other:
available RAM / unified memory, because LLM decode is memory-bandwidth bound, not
TOPS bound (a Hailo NPU accelerates vision, not LLMs).

This demo encodes the PDF's hardware + model-selection matrices as real logic:
give it a memory budget and it returns the platform, the best Gemma 4 variant,
the quantization, and the expected tokens/sec. Then it asks the LOCAL model to
justify one recommendation — sovereign advice about sovereign hardware.

Run:  python demos/step02_hardware_advisor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edgeview  # noqa: E402

# From the tutorial's "Model Selection by VRAM" + "Hardware Decision Matrix".
# (ram_ceiling_gb, platform, model, quant, expected_tok_s, note)
MATRIX = [
    (2,   "Phone (Android/iOS)",     "Gemma 4 E2B",        "2-bit (LiteRT)", "7–31",  "always with you, offline"),
    (4,   "Jetson Orin Nano",        "Gemma 4 E4B",        "Q4 (GGUF/LiteRT)", "12–20", "GPU vision + small LLM"),
    (8,   "Raspberry Pi 5 / Orin",   "Gemma 4 E4B / Llama 3.2 1B", "Q4_K_M", "20–54", "air-gapped, ultra-low-cost"),
    (16,  "Mac 16 GB / RTX 4070 Ti", "Gemma 4 26B MoE",    "Q4_K_M",       "15–30", "developer workstation"),
    (24,  "RTX 4090",                "Gemma 4 31B Dense",  "Q4_K_M",       "~35",   "consumer GPU, max single-user"),
    (64,  "Mac M2/M4 Ultra",         "Gemma 4 31B Dense",  "Q8_0",         "~20",   "silent, unified memory"),
    (9999,"DGX Spark GB10",          "Gemma 4 31B Dense",  "NVFP4 / BF16", "23–65", "production multi-user MAS"),
]

# Use-case → platform, from the decision matrix.
USE_CASES = [
    ("Production local assistant",     "DGX Spark",          "Gemma 4 31B"),
    ("Robotics / VLA",                 "Jetson Orin / Thor", "Gemma 4 E4B"),
    ("IoT / air-gapped cluster",       "Raspberry Pi 5 ×N",  "Gemma 4 E2B"),
    ("Developer workstation",          "Mac M4 Pro/Max",     "Gemma 4 26B MoE"),
    ("Mobile agent",                   "Android/iOS phone",  "Gemma 4 E2B"),
    ("Smart building / HVAC edge",     "DGX Spark + IoT GW", "Gemma 4 31B"),
]


def recommend(ram_gb: float) -> dict:
    for ceiling, platform, model, quant, speed, note in MATRIX:
        if ram_gb <= ceiling:
            return {"platform": platform, "model": model, "quant": quant,
                    "tok_s": speed, "note": note}
    return {}


def main() -> None:
    edgeview.banner("PART 3", "Edge hardware advisor", "BEGINNER")

    print("Recommendations by memory budget (decode speed scales with memory")
    print("bandwidth, so RAM is the deciding factor):\n")
    print(f"  {'RAM':>6}  {'Platform':<26} {'Model':<26} {'Quant':<14} {'tok/s':<7}")
    print("  " + "─" * 84)
    for ram in (2, 4, 8, 16, 24, 64, 128):
        r = recommend(ram)
        print(f"  {ram:>4}GB  {r['platform']:<26} {r['model']:<26} "
              f"{r['quant']:<14} {r['tok_s']:<7}")

    print("\nBy use case:\n")
    for uc, plat, model in USE_CASES:
        print(f"  • {uc:<30} → {plat:<22} ({model})")

    if not edgeview.require_local():
        return

    # Make the advice itself sovereign: ask the local model to justify a pick.
    pick = recommend(16)
    print(f"\nNow let the LOCAL model explain ONE pick — a 16 GB developer Mac → "
          f"{pick['model']} ({pick['quant']}):\n")
    edgeview.sovereignty_line()
    edgeview.generate(
        f"A developer has a 16 GB Apple Silicon Mac and wants a sovereign coding "
        f"assistant. We recommend {pick['model']} quantized to {pick['quant']}. "
        f"In 3 short bullet points, explain why a 26B Mixture-of-Experts model is "
        f"the sweet spot here versus a 31B dense model. Be concise.",
        max_tokens=800,
        title="why 26B MoE on a 16 GB Mac?",
    )

    print("\nTakeaway: match the model to the memory, not the marketing TOPS number.")
    print("MoE activates only ~3.8B params/token, so it punches far above its memory.")


if __name__ == "__main__":
    main()
