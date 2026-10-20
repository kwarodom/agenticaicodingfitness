#!/usr/bin/env python3
"""Exercise 04 · Pick a GGUF quant that fits: block sizes, file sizes, a memory budget.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/04_llama_cpp_lm_studio/exercises/ex04_pick_a_quant.py

The checker is free and offline: it checks your functions against ggml's block sizes (the
same ones lab 02 verified against a real GGUF file), then picks a quant for three models
under a memory budget. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

SPARK_GB = 128
# whole-file bits/weight for the recipes you can download (K-quant "_M" recipes mix types — see lab 02)
QUANTS = {"BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K_M": 5.7, "Q4_K_M": 4.85, "Q3_K_M": 3.9, "Q2_K": 2.625}

# ── TODO 1 ── bits per weight of one GGML block type.
#   A block stores `block_weights` weights in `block_bytes` bytes, scales included.
#   Q8_0: 32 weights in 34 bytes → 8.5 bits.  Q4_K: 256 weights in 144 bytes → 4.5 bits.
def bits_per_weight(block_bytes: int, block_weights: int) -> float:
    return None


# ── TODO 2 ── size of a GGUF file in GB (1 GB = 1e9 bytes) for params_b billion weights at bpw bits each.
def gguf_gb(params_b: float, bpw: float) -> float:
    return None


# ── TODO 3 ── the best quant that fits: the HIGHEST bits/weight in `quants` whose gguf_gb() is ≤ budget_gb.
#   Return its name (e.g. "Q6_K"), or None when even the smallest does not fit.
def pick_quant(params_b: float, budget_gb: float, quants: dict[str, float] = QUANTS) -> str | None:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
def close(a, b, tol=0.005) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) <= tol * max(1.0, abs(b))


def main() -> None:
    banner("Exercise 04 · pick a quant", "offline checker · free · no Spark needed", status=False)
    ok = True
    blocks = {"Q8_0": (34, 32, 8.5), "Q4_0": (18, 32, 4.5), "Q6_K": (210, 256, 6.5625),
              "Q5_K": (176, 256, 5.5), "Q4_K": (144, 256, 4.5), "Q3_K": (110, 256, 3.4375), "Q2_K": (84, 256, 2.625)}
    got = {k: bits_per_weight(b, n) for k, (b, n, _) in blocks.items()}
    ok &= check(all(close(got[k], v[2]) for k, v in blocks.items()),
                "bits_per_weight: Q8_0 8.5 · Q6_K 6.5625 · Q4_K 4.5 · Q2_K 2.625",
                f"TODO 1: bits_per_weight(34, 32) should be 8.5 (got {got['Q8_0']!r})")
    ok &= check(close(gguf_gb(35, 4.85), 21.21875) and close(gguf_gb(8, 16), 16.0),
                "gguf_gb: Qwen3.6-35B-A3B at Q4_K_M ≈ 21.2 GB · 8B at BF16 = 16 GB",
                f"TODO 2: gguf_gb(35, 4.85) should be ≈ 21.2 (got {gguf_gb(35, 4.85)!r})")
    picks = [pick_quant(35, 68), pick_quant(70.6, 68), pick_quant(405, 118), pick_quant(8, 100)]
    ok &= check(picks == ["Q8_0", "Q6_K", None, "BF16"],
                "pick_quant: 35B→Q8_0 · 70.6B→Q6_K · 405B→None · 8B→BF16",
                f"TODO 3: expected ['Q8_0', 'Q6_K', None, 'BF16'] (got {picks!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your picker, applied: 1 Spark keeps 60 GB free for KV cache and a second model; 2 Sparks keep 10 GB")
    for name, params, sparks in [("Qwen3.6-35B-A3B", 35, 1), ("Llama 3.3 70B", 70.6, 1),
                                 ("Llama 3.1 405B", 405, 1), ("Llama 3.1 405B", 405, 2)]:
        budget = sparks * SPARK_GB - 60 if sparks == 1 else sparks * SPARK_GB - 10
        q = pick_quant(params, budget)
        size = f"{gguf_gb(params, QUANTS[q]):6.1f} GB" if q else "   —    "
        print(f"│ {name:16s} on {sparks} Spark{'s' if sparks > 1 else ' '}  budget {budget:3d} GB → {q or 'does not fit':12s} {size}")
    print("\n═ One Spark keeps a 35B MoE at Q8_0 (lab 03: 0.6 % weight error) with room to spare; 405B needs two Sparks "
          "(Module 02) and ~4-bit weights. Now try budget = 128 − 10: which quant does the 70B get?")


if __name__ == "__main__":
    main()
