#!/usr/bin/env python3
"""PART 8 · Scaling out — link two DGX Sparks  [ADVANCED]

One Spark is 128 GB. Link two over the built-in ConnectX-7 200GbE and you have
256 GB of coherent memory — enough to run a 235B (or up to ~405B) model with
tensor parallelism. This is the NVIDIA "connect-two-sparks" + NCCL + TensorRT-LLM
multi-node story. This demo explains the wiring, the parallelism, and does the
memory math for what two/three Sparks unlock.

Run:  python demos/step08_multi_spark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgxview  # noqa: E402

WIRING = """\
# 1) cable the two Sparks QSFP↔QSFP (ConnectX-7 200GbE), give them static IPs
#    spark-0: 192.168.100.10   spark-1: 192.168.100.11
# 2) verify the high-speed link with NCCL
docker run --gpus all --network host nvcr.io/nvidia/pytorch:25.11-py3 \\
  all_reduce_perf -b 8 -e 256M -f 2 -g 1     # expect ~tens of GB/s busbw
# 3) serve a big model across both with TensorRT-LLM tensor parallelism (TP=2)
mpirun -H 192.168.100.10,192.168.100.11 -np 2 \\
  trtllm-serve nvidia/Qwen3-235B-A22B-NVFP4 --tp_size 2 --port 8355
"""


def main() -> None:
    dgxview.banner("PART 8", "Scaling out — link two DGX Sparks", "ADVANCED")
    dgxview.mode_line()

    print("Parallelism, in one line each:")
    print("  • Tensor Parallel (TP)   — split EACH layer's matrices across GPUs;")
    print("                             needs fast interconnect (that's the 200GbE).")
    print("  • Pipeline Parallel (PP) — split the layer STACK across GPUs; cheaper")
    print("                             comms, but GPUs take turns (bubble).")
    print("  • Data Parallel (DP)     — replicate the model, split the requests;")
    print("                             for throughput, not for fitting a big model.\n")

    print("Memory the cluster unlocks (NVFP4, ~0.55 B/param × 1.18 overhead):\n")
    print(f"  {'Cluster':<18}{'Memory':>10}   largest model that fits (NVFP4)")
    print("  " + "─" * 64)
    for nodes, label in [(1, "1 Spark"), (2, "2 Sparks (TP=2)"), (3, "3 Sparks (ring)")]:
        mem = 128 * nodes
        max_params = mem * 0.9 / (0.55 * 1.18)
        print(f"  {label:<18}{mem:>7} GB   up to ~{max_params:.0f}B params")
    print()

    print("How to wire and serve across two Sparks:\n")
    print(WIRING)

    print("Notes from the playbooks:")
    print("  • The 200GbE QSFP link (not the Ethernet jack) is what makes TP viable —")
    print("    TP is comms-bound, so the fast fabric is the whole point.")
    print("  • NCCL all_reduce_perf is your 'is the link healthy?' smoke test.")
    print("  • Three Sparks → ring topology; beyond that, go through a switch.")

    print("\nTakeaway: sovereignty scales. Two desk-side Sparks run a 235B model with")
    print("tensor parallelism — frontier-class capability, still entirely on-prem.")


if __name__ == "__main__":
    main()
