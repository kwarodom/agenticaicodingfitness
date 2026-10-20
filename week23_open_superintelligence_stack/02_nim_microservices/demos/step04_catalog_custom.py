#!/usr/bin/env python3
"""PART 4 · The NIM catalog + your own custom-model NIM  [ADVANCED]

build.nvidia.com is the catalog of ready NIMs (Nemotron, Llama, embeddings, RAG,
speech…). You can also wrap YOUR fine-tuned model (e.g. the LoRA from Week 19) as a
NIM and deploy it the same way — the sovereign AI-factory pattern.

Run:  python demos/step04_catalog_custom.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

CUSTOM = """\
# 🖥️ wrap YOUR fine-tuned weights as a NIM (LoRA from Week 19 / a merged model)
docker run --gpus all -p 8000:8000 \\
  -e NIM_FT_MODEL=/models/hvac-assistant \\
  -v $PWD/hvac-assistant:/models/hvac-assistant \\
  nvcr.io/nim/nvidia/llm-nim:latest
#  → your domain model, served with a production engine + OpenAI API, on-prem
"""


def main() -> None:
    view.banner("PART 4", "The NIM catalog + your own custom NIM", "ADVANCED")
    view.mode_line()

    print("A slice of the build.nvidia.com catalog (model → auto-selected engine):\n")
    print(f"  {'Model (NIM)':<28}{'Engine':<16}{'~tok/s':>7}   use")
    print("  " + "─" * 72)
    for m, eng, tok, use in sim.CATALOG:
        print(f"  {m:<28}{eng:<16}{tok:>7}   {use}")
    print()
    print("Deploy YOUR fine-tuned model as a NIM (the sovereign AI-factory loop):\n")
    print(CUSTOM)
    print("The full sovereign lifecycle:")
    print("  build.nvidia.com NIM  →  fine-tune on your DGX (Week 19)  →  wrap as a NIM")
    print("  →  serve on-prem  →  improve with the Data Flywheel (app 3)  →  repeat.\n")
    print("Enterprise notes: NIM production use needs an NVIDIA AI Enterprise license")
    print("(bundled with DGX). Containers are free to pull for dev/eval.")

    print("\nTakeaway: the catalog gets you frontier models in one command; wrapping your")
    print("own model as a NIM makes your domain expert a first-class sovereign service.")


if __name__ == "__main__":
    main()
