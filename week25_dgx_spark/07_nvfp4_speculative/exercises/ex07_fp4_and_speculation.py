#!/usr/bin/env python3
"""Exercise 07 · Build a 4-bit quantizer and a speculative-decoding planner yourself.

Fill in the four TODOs, save, then run:
    .venv/bin/python week25/07_nvfp4_speculative/exercises/ex07_fp4_and_speculation.py

The checker is free and offline: it tests your functions against known answers, then
uses them to quantize 4,096 numbers and to pick a draft length. Stuck? Compare with
exercises/solutions/.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]      # every magnitude an FP4 (E2M1) value can hold


# ── TODO 1 ── round x to the nearest E2M1 value, keeping its sign.
#   Anything bigger than 6 in magnitude saturates to ±6. Example: 2.6 → 3.0, -0.3 → -0.5, 100 → 6.0.
def e2m1_round(x: float) -> float:
    return None


# ── TODO 2 ── quantize one block the NVFP4 way and return the DECODED values.
#   scale = (largest |value| in the block) / 6, so the largest value lands on ±6.
#   decoded value = e2m1_round(v / scale) * scale. A block of all zeros stays all zeros.
#   (Real NVFP4 also rounds the scale to FP8 — lab 01 does that; skip it here.)
def quantize_block(block: list[float]) -> list[float]:
    return None


# ── TODO 3 ── expected tokens per target step with draft length k and acceptance rate alpha:
#   E = (1 − alpha^(k+1)) / (1 − alpha). When alpha == 1 every draft is accepted: E = k + 1.
def expected_tokens(alpha: float, k: int) -> float:
    return None


# ── TODO 4 ── the draft length (1..kmax) with the best speed-up E / (1 + k·c).
#   c = cost of one draft pass as a fraction of one target pass. Return the smallest k on a tie.
def best_draft_len(alpha: float, c: float, kmax: int = 10) -> int:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
def close(a, b, tol=1e-6) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) <= tol * max(1.0, abs(b))


def sqnr_db(xs, ys) -> float:
    se = sum((a - b) ** 2 for a, b in zip(xs, ys))
    return 10 * math.log10(sum(a * a for a in xs) / se) if se else float("inf")


def main() -> None:
    banner("Exercise 07 · a 4-bit quantizer and a draft-length planner", "offline checker · free · no Spark needed",
           status=False)
    ok = True
    cases = [(2.6, 3.0), (-0.3, -0.5), (100, 6.0), (0.1, 0.0), (-4.9, -4.0), (5.2, 6.0), (1.2, 1.0)]
    got = [e2m1_round(x) for x, _ in cases]
    ok &= check(all(close(g, w) for g, (_, w) in zip(got, cases)),
                "e2m1_round: 2.6→3 · -0.3→-0.5 · 100→6 · 0.1→0 · -4.9→-4 · 5.2→6 · 1.2→1",
                f"TODO 1: e2m1_round gave {got} — want {[w for _, w in cases]}")
    blk = [0.012, -0.03, 0.0045, 0.06, -0.018, 0.0, 0.027, -0.051]
    q = quantize_block(blk)
    want = [0.01, -0.03, 0.005, 0.06, -0.02, 0.0, 0.03, -0.06]
    ok &= check(isinstance(q, list) and len(q) == len(blk) and all(close(a, b, 1e-9) for a, b in zip(q, want))
                and quantize_block([0.0] * 4) == [0.0] * 4,
                "quantize_block: scale 0.06/6 = 0.01 → [0.01, -0.03, 0.005, 0.06, -0.02, 0, 0.03, -0.06]",
                f"TODO 2: quantize_block gave {q!r} — want {want}")
    ok &= check(close(expected_tokens(0.8, 4), 3.3616) and close(expected_tokens(0.5, 1), 1.5)
                and close(expected_tokens(1.0, 5), 6.0),
                "expected_tokens: α=0.8 k=4 → 3.36 · α=0.5 k=1 → 1.5 · α=1 k=5 → 6",
                f"TODO 3: expected_tokens(0.8, 4) should be 3.3616 (got {expected_tokens(0.8, 4)!r})")
    ok &= check(best_draft_len(0.7, 0.113) == 4 and best_draft_len(0.9, 0.0, 8) == 8 and best_draft_len(0.3, 0.5) == 1,
                "best_draft_len: α=0.7 c=0.113 → 4 · free drafts → kmax · weak draft → 1",
                f"TODO 4: best_draft_len(0.7, 0.113) should be 4 (got {best_draft_len(0.7, 0.113)!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your quantizer, applied to 4,096 weight-like numbers (seed 7, 0.5 % outliers)")
    rng = random.Random(7)
    xs = [rng.gauss(0, 0.02) * (12 if rng.random() < 0.005 else 1) for _ in range(4096)]
    for size in (16, 64, 4096):
        ys = [y for i in range(0, len(xs), size) for y in quantize_block(xs[i:i + size])]
        print(f"│ one scale per {size:5d} values   SQNR {sqnr_db(xs, ys):5.1f} dB")
    print("\n▣ your planner, applied (Draft-Target c = 0.113, from lab 04)")
    for a in (0.6, 0.7, 0.8, 0.9):
        k = best_draft_len(a, 0.113)
        print(f"│ α = {a}  best k = {k}  E = {expected_tokens(a, k):.2f} tokens per target pass")
    print("\n═ Smaller blocks → higher SQNR: that is NVFP4's 16. Higher α → longer drafts pay off.")


if __name__ == "__main__":
    main()
