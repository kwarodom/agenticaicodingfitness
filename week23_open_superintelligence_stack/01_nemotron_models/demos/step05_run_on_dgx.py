#!/usr/bin/env python3
"""PART 5 · Run Nemotron on the DGX — 1 Spark & 2 Sparks  [ADVANCED]

How to actually stand up Nemotron on your own hardware: pull from build.nvidia.com
or Ollama, serve on one Spark (Nano/Super), or link two Sparks over the QSFP 200GbE
cable to run Ultra with tensor parallelism. Prints the real commands and shows the
memory math.

Run:  python demos/step05_run_on_dgx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ntsim  # noqa: E402
import ntview  # noqa: E402

ONE_SPARK = """\
# 🖥️ 1 DGX Spark — Nano or Super (fits at NVFP4)
# via Ollama:
ollama pull nemotron-3-nano:30b-a3b
ollama run  nemotron-3-nano:30b-a3b        # OpenAI API on :11434
# or as a NIM microservice (see the nim_microservices app):
docker run --gpus all -p 8000:8000 nvcr.io/nim/nvidia/nemotron-3-super:latest
# get the weights + cards at build.nvidia.com
"""

TWO_SPARK = """\
# 🖥️ 2 DGX Sparks over QSFP 200GbE — Ultra (550B) with tensor parallelism
# 1) cable the QSFP ports, static IPs (spark-0 .10, spark-1 .11), verify NCCL:
docker run --gpus all --network host nvcr.io/nvidia/pytorch:latest \\
  all_reduce_perf -b 8 -e 256M -f 2 -g 1        # expect tens of GB/s busbw
# 2) serve Ultra across both with TP=2 (vLLM/TRT-LLM under NIM):
mpirun -H spark-0,spark-1 -np 2 \\
  vllm serve nvidia/Nemotron-3-Ultra-NVFP4 --tensor-parallel-size 2 --port 8000
"""


def main() -> None:
    ntview.banner("PART 5", "Run Nemotron on the DGX — 1 & 2 Sparks", "ADVANCED")
    ntview.mode_line()

    print("What fits where (NVFP4 on the 128 GB GB10):\n")
    print(f"  {'Model':<18}{'VRAM':>10}{'Sparks':>8}   note")
    print("  " + "─" * 56)
    for s in ntsim.FAMILY[:3]:
        fits = "1 Spark ✓" if s.sparks == 1 else "2 Sparks (TP=2)"
        print(f"  {s.name:<18}{str(s.vram_gb_nvfp4)+'GB':>10}{s.sparks:>8}   {fits}")
    print()
    print("Serve on ONE Spark (Nano / Super):\n")
    print(ONE_SPARK)
    print("Scale to TWO Sparks for Ultra (over the QSFP 200GbE cable):\n")
    print(TWO_SPARK)

    print("Then call it — same OpenAI API, only the model name changes:")
    print("    from openai import OpenAI")
    print('    OpenAI(base_url="http://<spark>:11434/v1", api_key="…")\\')
    print('      .chat.completions.create(model="nemotron-3-super:120b-a12b", messages=[...])\n')

    if not ntview.is_sim():
        print("Live check — a quick reasoning call on the connected endpoint:\n")
        ntview.reason("In one sentence, why run open Nemotron on our own DGX?",
                      max_tokens=200, show_reasoning=False, title="live Nemotron on the DGX")

    print("\nTakeaway: Nano/Super run on one Spark today; two linked Sparks unlock Ultra —")
    print("frontier open reasoning, entirely sovereign. Next apps: serve it (NIM), scale")
    print("it (Dynamo), improve it (Data Flywheel / NeMo Gym), guard it (OpenShell).")


if __name__ == "__main__":
    main()
