#!/usr/bin/env python3
"""Exercise 06 · Benchmark maths — reference solution.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/exercises/ex06_benchmark_math.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from embedkit import banner, check  # noqa: E402


# ── TODO 1 ── percentile with linear interpolation
def percentile(values: list, p: float) -> float:
    s = sorted(values)
    i = (len(s) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


# ── TODO 2 ── items per second for `items` processed in `ms` milliseconds
def throughput(items: int, ms: float) -> float:
    return items / (ms / 1000)


# ── TODO 3 ── the bandwidth floor of one forward pass, in milliseconds
def bandwidth_floor_ms(weights_mb: float, gbps: float) -> float:
    return weights_mb / 1024 / gbps * 1000


# ── TODO 4 ── "bandwidth" if the measured time is within 2× of the floor, else "compute or overhead"
def limited_by(measured_ms: float, floor_ms: float) -> str:
    return "bandwidth" if measured_ms <= 2 * floor_ms else "compute or overhead"


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 06 · benchmark maths", "offline checker · no model needed")
    good = True
    times = [22.1, 21.8, 23.0, 22.4, 35.0, 22.0, 21.9, 22.6, 22.2, 60.0]
    try:
        p = [round(percentile(times, 50), 2), round(percentile(times, 95), 2), round(percentile([5.0], 95), 2)]
    except Exception as e:  # noqa: BLE001
        p = f"error: {e}"
    good &= check(p == [22.3, 48.75, 5.0], "percentile: p50 22.3 ms · p95 48.75 ms (linear interpolation) · one value → itself",
                  f"TODO 1: percentile gave {p} — sort, index (n−1)·p/100, interpolate between neighbours")
    try:
        t = [round(throughput(32, 135.5), 1), round(throughput(1, 22.2), 1)]
    except Exception as e:  # noqa: BLE001
        t = f"error: {e}"
    good &= check(t == [236.2, 45.0], "throughput: 32 items in 135.5 ms = 236.2 items/s · 1 in 22.2 ms = 45.0/s",
                  f"TODO 2: throughput gave {t}")
    try:
        f = [round(bandwidth_floor_ms(540, 410), 2), round(bandwidth_floor_ms(310, 17), 1)]
    except Exception as e:  # noqa: BLE001
        f = f"error: {e}"
    good &= check(f == [1.29, 17.8], "bandwidth_floor_ms: 540 MB at 410 GB/s = 1.29 ms · 310 MB at 17 GB/s = 17.8 ms",
                  f"TODO 3: bandwidth_floor_ms gave {f} — MB ÷ 1024 = GB; GB ÷ (GB/s) = s")
    try:
        v = [limited_by(measured_ms=135.5, floor_ms=1.29), limited_by(measured_ms=1.5, floor_ms=1.29)]
    except Exception as e:  # noqa: BLE001
        v = f"error: {e}"
    good &= check(v == ["compute or overhead", "bandwidth"],
                  "limited_by: far above the floor → compute or overhead · within 2× of it → bandwidth",
                  f"TODO 4: limited_by gave {v}")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Quote p50 AND p95, say what limits you, and never compare a laptop number with a device floor.")


if __name__ == "__main__":
    main()
