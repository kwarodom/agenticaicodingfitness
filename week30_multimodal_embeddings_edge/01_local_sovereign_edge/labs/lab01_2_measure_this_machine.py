#!/usr/bin/env python3
"""Lab 01-2 · Measure this machine: capacity, bandwidth, and which rung of the ladder it is.

The video's lesson is that capacity (how much RAM) decides WHAT fits, and bandwidth (how fast the memory bus
moves bytes) decides HOW FAST it runs. This lab reads your RAM from the OS and measures bandwidth for real:
a large array copy on the CPU (numpy) and, if there is one, on the GPU (PyTorch MPS or CUDA). Measured
copy speed is always below the vendor's peak number — the gap is part of the lesson.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_2_measure_this_machine.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, machine, note, ok, result, step, table, warn  # noqa: E402

import numpy as np  # noqa: E402

banner("Lab 01-2 · measure this machine", "capacity from the OS · bandwidth measured · no network")

step(1, "capacity — what the OS reports")
m = machine()
table([[k, v] for k, v in m.items()], ["field", "value"])
ram = m["ram_gb"] or 0
usable = ram * 0.75
ok(f"usable ≈ {usable:.1f} GB → Tina's rule: up to ≈ {usable / 0.6:.0f}B parameters (4-bit chat model)")


def copy_gbs(n_bytes: int, repeats: int = 5) -> float:
    """Best-of-N memcpy speed. A copy reads n bytes and writes n bytes → 2n bytes moved."""
    src = np.ones(n_bytes // 4, dtype=np.float32)
    dst = np.empty_like(src)
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        dt = time.perf_counter() - t0
        best = max(best, 2 * n_bytes / dt / 1e9)
    return best


step(2, "bandwidth — CPU memory copy (numpy, one thread)")
cpu = copy_gbs(1 << 30)
ok(f"CPU copy: {cpu:.0f} GB/s (best of 5, 1 GiB array)")

step(3, "bandwidth — GPU memory copy (PyTorch)")
gpu = None
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "")
    if dev:
        sync = torch.mps.synchronize if dev == "mps" else torch.cuda.synchronize
        x = torch.ones((1 << 30) // 4, dtype=torch.float32, device=dev)
        y = torch.empty_like(x)
        y.copy_(x)
        sync()
        best = 0.0
        for _ in range(10):
            t0 = time.perf_counter()
            y.copy_(x)
            sync()
            best = max(best, 2 * (1 << 30) / (time.perf_counter() - t0) / 1e9)
        gpu = best
        ok(f"GPU copy on {dev}: {gpu:.0f} GB/s (best of 10, 1 GiB tensor)")
        del x, y
    else:
        warn("no GPU backend here — CPU only (like the Raspberry Pi in the video)")
except Exception as e:  # noqa: BLE001
    warn(f"GPU test skipped: {type(e).__name__}: {e}")

step(4, "where this machine sits on the ladder")
tier = ("chip" if ram < 0.5 else "mini computer" if ram <= 8 else "personal computer" if ram <= 48
        else "home server" if ram <= 192 else "GPU server")
table([["capacity", f"{ram:.0f} GB", tier],
       ["CPU copy bandwidth (measured)", f"{cpu:.0f} GB/s", "one core's view of the bus"],
       ["GPU copy bandwidth (measured)", f"{gpu:.0f} GB/s" if gpu else "—", "closer to the vendor peak"]],
      ["measure", "value", "meaning"])
note("vendor peaks for comparison (RESEARCH.md §4): Raspberry Pi 5 ~17 GB/s · Apple M4 Max (14-core) 410 GB/s · "
     "AMD Ryzen AI Max+ 395 256 GB/s · RTX 4090 1,008 GB/s · H100 SXM 3,350 GB/s")
result(f"this machine is a '{tier}' — every number later this week is a LAPTOP STAND-IN, measured here, "
       "not on a Pi or a phone.")
