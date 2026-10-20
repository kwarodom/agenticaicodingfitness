#!/usr/bin/env python3
"""Exercise 01 · Write the "will it fit?" calculator yourself.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/01_meet_your_spark/exercises/ex01_will_it_fit.py

The checker is free and offline: it compares your functions with known answers,
then uses them to decide where three models should run. Stuck? Compare with
exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

SPARK_GB = 128

# ── TODO 1 ── bytes for the weights, in GB (1 GB = 1e9 bytes).
#   params_b is in billions; bits is bits per weight (16 for bf16, 8 for fp8, 4.5 for nvfp4 incl. scales).
def weights_gb(params_b: float, bits: float) -> float:
    return None


# ── TODO 2 ── KV cache in GB: K and V (×2), for every layer, every KV head, every head dim,
#   every token in the context, every concurrent sequence, bytes_per element (2 for bf16).
def kv_cache_gb(layers: int, kv_heads: int, head_dim: int, ctx: int, batch: int = 1,
                bytes_per: float = 2) -> float:
    return None


# ── TODO 3 ── where does it run? Return "1 Spark", "2 Sparks", or "too big".
#   need = weights + kv + overhead_gb. One Spark holds 128 GB; two hold 256 GB.
def placement(need_gb: float) -> str:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
def close(a, b, tol=0.02) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) <= tol * max(1.0, abs(b))


def main() -> None:
    banner("Exercise 01 · will it fit?", "offline checker · free · no Spark needed", status=False)
    ok = True
    ok &= check(close(weights_gb(8, 16), 16.0) and close(weights_gb(70, 4.5), 39.375),
                "weights_gb: 8B bf16 = 16 GB · 70B nvfp4 ≈ 39.4 GB",
                f"TODO 1: weights_gb(8, 16) should be 16.0 (got {weights_gb(8, 16)!r})")
    ok &= check(close(kv_cache_gb(32, 8, 128, 32_768), 4.295) and close(kv_cache_gb(80, 8, 128, 8_192, 4), 10.737),
                "kv_cache_gb: Llama 8B @32K ≈ 4.3 GB · Llama 70B @8K × 4 users ≈ 10.7 GB",
                f"TODO 2: kv_cache_gb(32, 8, 128, 32768) should be ≈ 4.295 (got {kv_cache_gb(32, 8, 128, 32_768)!r})")
    ok &= check([placement(x) for x in (100, 128, 129, 256, 300)] == ["1 Spark", "1 Spark", "2 Sparks", "2 Sparks", "too big"],
                "placement: 100→1 · 128→1 · 129→2 · 256→2 · 300→too big",
                f"TODO 3: placement(129) should be '2 Sparks' (got {placement(129)!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your calculator, applied (32K context, one user, 10 GB overhead)")
    for name, params, bits, arch in [("Qwen3 32B · fp8", 32.8, 8, (64, 8, 128)),
                                     ("Llama 3.3 70B · bf16", 70.6, 16, (80, 8, 128)),
                                     ("Llama 3.1 405B · nvfp4", 405, 4.5, (126, 8, 128))]:
        need = weights_gb(params, bits) + kv_cache_gb(*arch, 32_768) + 10
        print(f"│ {name:24s} needs {need:6.1f} GB → {placement(need)}")
    print("\n═ Compare with lab02's table. Now change the context to 131072 — which model moves?")


if __name__ == "__main__":
    main()
