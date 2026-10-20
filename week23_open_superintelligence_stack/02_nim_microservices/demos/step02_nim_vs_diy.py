#!/usr/bin/env python3
"""PART 2 · NIM vs raw vLLM vs Ollama — when to use which  [INTERMEDIATE]

All three expose the SAME OpenAI API on your DGX. They differ in who does the
optimization, security, and support. This demo lays out the trade-off.

Run:  python demos/step02_nim_vs_diy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

ROWS = [
    ("Setup",        "one `docker run`",          "you tune flags",         "one `ollama run`"),
    ("Engine",       "auto TRT-LLM/vLLM/SGLang",  "vLLM (you configure)",   "llama.cpp/GGUF"),
    ("Optimization", "pre-built TRT engines",     "you build/tune",         "auto, general"),
    ("Security",     "scanned + signed",          "you own it",             "you own it"),
    ("Support",      "NVIDIA AI Enterprise",      "community",              "community"),
    ("Best for",     "production / AI factory",   "max flexibility",        "quick start / dev"),
    ("License/cost", "AI-Enterprise ($/GPU/yr)",  "free (OSS)",             "free (OSS)"),
]


def main() -> None:
    view.banner("PART 2", "NIM vs raw vLLM vs Ollama", "INTERMEDIATE")
    view.mode_line()

    print(f"  {'':<14}{'NIM':<28}{'raw vLLM':<26}Ollama")
    print("  " + "─" * 92)
    for k, a, b, c in ROWS:
        print(f"  {k:<14}{a:<28}{b:<26}{c}")
    print()
    print("Decision guide:")
    print("  • Production / regulated / AI factory → NIM (one container, supported, secure).")
    print("  • Need a model outside the catalog, or max control → raw vLLM (Week 19 · app 1).")
    print("  • Dev box / quick start / frequent model swaps → Ollama.")
    print("  • They all speak the OpenAI API, so you can start on Ollama and promote to NIM")
    print("    later with no client change — just the base_url.\n")

    print("Note (from NIM LLM 2.0): 'one container, one backend' — often vLLM under the")
    print("hood — so NIM is a *productionized, supported* packaging of the same engines.")

    print("\nTakeaway: NIM trades a little flexibility for one-command, secure, supported")
    print("sovereign inference. Next: call a NIM the same way you call any OpenAI endpoint.")


if __name__ == "__main__":
    main()
