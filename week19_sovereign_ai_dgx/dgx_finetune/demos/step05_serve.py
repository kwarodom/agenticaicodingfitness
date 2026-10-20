#!/usr/bin/env python3
"""STEP 5 · 🖥️ Serve the tuned model via Ollama on the DGX — REAL SSH  [ADVANCED]

Registers the GGUF that STEP 4 produced as an Ollama model on the DGX and starts
it, so it's reachable on the same OpenAI API as your other models — and selectable
in this app's model dropdown. Runs on the DGX over SSH.

Run:  python demos/step05_serve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ftssh  # noqa: E402

MODEL_NAME = "hvac-assistant"


def main() -> None:
    print("━" * 64)
    print("  STEP 5 · 🖥️ Serve the tuned model (Ollama) on the DGX")
    print("━" * 64, "\n")
    if not ftssh.require():
        return

    print("1) Locate the GGUF produced by training:")
    ftssh.run(f"ls -la {ftssh.WORKDIR}/hvac-model/*.gguf 2>/dev/null || "
              f"echo 'no GGUF found — did STEP 4 finish?'")

    print("\n2) Write a Modelfile + register it with Ollama:")
    gguf_glob = f"{ftssh.WORKDIR}/hvac-model"
    remote = (
        f"cd {ftssh.WORKDIR}/hvac-model && "
        f"GGUF=$(ls *.gguf | head -1) && "
        f"printf 'FROM ./%s\\nSYSTEM \"You are HotelHVAC-Assistant.\"\\n' \"$GGUF\" > Modelfile && "
        f"cat Modelfile && "
        f"ollama create {MODEL_NAME} -f Modelfile"
    )
    rc = ftssh.run(remote)

    if rc == 0:
        print(f"\n3) Quick smoke test of '{MODEL_NAME}':")
        ftssh.run(f'ollama run {MODEL_NAME} "Filter dP is 260 Pa - priority?" --verbose 2>&1 | head -20')
        print(f"\n✓ '{MODEL_NAME}' is now served on the DGX (OpenAI API on :11434).")
        print("  In this app's model dropdown (tunnel/local connection), pick "
              f"'{MODEL_NAME}'. STEP 6 evaluates it.")
    else:
        print("\n✗ couldn't register the model — check the GGUF exists (STEP 4) and "
              "that Ollama is installed on the DGX.")

    print(f"\n(unused ref: {gguf_glob})")
    print("\nTakeaway: your domain-tuned model is now a first-class citizen on the DGX,")
    print("served on the same API as everything else — sovereign, end to end.")


if __name__ == "__main__":
    main()
