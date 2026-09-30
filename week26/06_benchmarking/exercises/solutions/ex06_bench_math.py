#!/usr/bin/env python3
"""Exercise 06 · Benchmark math — reference solution.

Fill in the four TODOs, save, then run:
    .venv/bin/python week26/06_benchmarking/exercises/ex06_bench_math.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from clawkit import banner, check  # noqa: E402


# ── TODO 1 ── percentile with linear interpolation (numpy's default, and what benchkit uses).
#   Sort the values. The rank is k = (n − 1) × p / 100. Interpolate between the values at floor(k) and ceil(k).
#   percentile([1..10], 50) == 5.5 · percentile([1..10], 95) == 9.55
def percentile(xs: list[float], p: float) -> float:
    s = sorted(float(x) for x in xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── TODO 2 ── research tutorial Part 5, exercise 4: ai-muninn measured 786 tok/s AGGREGATE at c=16 on Nemotron 3
#   Nano W4A4 (cited in the research tutorial). Roughly what does each of 16 concurrent sessions see?
def per_session_tok_s(aggregate_tok_s: float, sessions: int) -> float:
    return aggregate_tok_s / sessions          # 786 / 16 ≈ 49 tok/s: batching raises the total, not each stream


# ── TODO 3 ── least-squares straight line through (concurrency, p95 runtime) points: return (slope, intercept).
#   slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²) · intercept = (Σy − slope·Σx) / n
def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy, sxx = sum(x * y for x, y in zip(xs, ys)), sum(x * x for x in xs)
    slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    return slope, (sy - slope * sx) / n


# ── TODO 4 ── the sizing calculator's GPU estimate (NAT 1.9 calc/calculations.py):
#   the concurrency one test run sustains at the target is c* = (target − intercept) / slope,
#   and GPUs = target_users / c* × test_gpu_count. Raise ValueError when target ≤ intercept or c* ≤ 0.
def gpus_needed(target_s: float, target_users: int, slope: float, intercept: float, test_gpu_count: int = 1) -> float:
    if target_s <= intercept:
        raise ValueError(f"target {target_s} s must be above the intercept {intercept} s")
    c_star = (target_s - intercept) / slope
    if c_star <= 0:
        raise ValueError(f"target concurrency {c_star} is not positive — check the slope")
    return target_users / c_star * test_gpu_count


# ─────────────────────────── checker — no need to edit below ────────────────
# SYNTHETIC practice numbers (not a measurement): p95 workflow runtime of a claw at 4 concurrencies on 1 Spark.
PRACTICE_C = [1, 2, 4, 8]
PRACTICE_P95 = [13.0, 14.0, 16.0, 20.0]


def main() -> None:
    banner("Exercise 06 · benchmark math", "offline checker · free · no Spark needed", status=False)
    good = True
    ten = list(range(1, 11))
    p50, p95 = percentile(ten, 50), percentile(ten, 95)
    good &= check(math.isclose(p50, 5.5) and math.isclose(p95, 9.55) and math.isclose(percentile([7.0], 95), 7.0),
                  "percentile: p50 of 1..10 = 5.5 · p95 = 9.55 · one value is its own p95",
                  f"TODO 1: got p50={p50} p95={p95} for 1..10 — rank k = (n−1)·p/100, then interpolate")
    ps = per_session_tok_s(786, 16)
    good &= check(math.isclose(ps, 786 / 16, rel_tol=1e-6),
                  f"per session: 786 tok/s ÷ 16 sessions ≈ {786 / 16:.0f} tok/s each (bandwidth is shared)",
                  f"TODO 2: got {ps} — aggregate throughput is split across the sessions")
    m, b = linear_fit(PRACTICE_C, PRACTICE_P95)
    good &= check(math.isclose(m, 1.0, rel_tol=1e-6) and math.isclose(b, 12.0, rel_tol=1e-6),
                  "linear fit: p95 ≈ 1.00 s × concurrency + 12.00 s on the practice points",
                  f"TODO 3: got slope={m} intercept={b} — expected 1.0 and 12.0")
    try:
        g = gpus_needed(15, 40, 1.0, 12.0)
        raised = False
        try:
            gpus_needed(10, 40, 1.0, 12.0)
        except ValueError:
            raised = True
    except Exception as e:  # noqa: BLE001
        g, raised = float("nan"), False
        print(f"  ({type(e).__name__}: {e})")
    good &= check(math.isclose(g, 40 / 3, rel_tol=1e-6) and raised,
                  "sizing: ≤ 15 s → c* = 3 per Spark → 40 users need ≈ 13.3 Sparks · a target below the intercept raises",
                  f"TODO 4: got {g} (want 40 / ((15 − 12) / 1) = 13.33) and ValueError for target 10 < intercept 12: {raised}")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. The sizing docs call this estimate rough, not for production: fit ≥ 10 concurrencies, "
          "measured on the Spark.")


if __name__ == "__main__":
    main()
