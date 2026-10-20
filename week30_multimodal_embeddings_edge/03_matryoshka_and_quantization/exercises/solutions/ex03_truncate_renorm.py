#!/usr/bin/env python3
"""Exercise 03 · Truncate, re-normalise, pack — reference solution.

Fill in the three TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/exercises/ex03_truncate_renorm.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from embedkit import DIMS, banner, check  # noqa: E402


# ── TODO 1 ── keep the first `dim` numbers of a 768-d vector and re-normalise to length 1.
#   Raise ValueError if dim is not one of DIMS (the model was only trained for those).
def truncate(v, dim: int):
    if dim not in DIMS:
        raise ValueError(f"EmbeddingGemma 2 supports {DIMS}, not {dim}")
    cut = np.asarray(v, dtype=np.float32)[:dim]
    return cut / np.linalg.norm(cut)


# ── TODO 2 ── bytes needed for n vectors of `dim` numbers stored as "float32" (4 B), "int8" (1 B) or "binary" (1 bit)
def storage_bytes(n: int, dim: int, precision: str) -> int:
    per = {"float32": dim * 4, "int8": dim, "binary": dim // 8}[precision]
    return n * per


# ── TODO 3 ── binary quantization: each number > 0 becomes bit 1, else 0, packed 8 per byte (np.packbits);
#   hamming(a, b) = how many bits differ (XOR, then count the ones).
def pack_bits(v):
    return np.packbits(np.asarray(v) > 0)


def hamming(a, b) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 03 · truncate, re-normalise, pack", "offline checker · numpy only · no model needed")
    good = True
    rng = np.random.default_rng(27)
    v = rng.normal(size=768).astype(np.float32)
    v /= np.linalg.norm(v)
    try:
        t = truncate(v, 128)
        r1 = (t.shape == (128,) and abs(float(np.linalg.norm(t)) - 1) < 1e-5
              and np.allclose(t, v[:128] / np.linalg.norm(v[:128]), atol=1e-6))
    except Exception as e:  # noqa: BLE001
        r1, t = False, f"error: {e}"
    good &= check(r1, "truncate(v, 128): the first 128 numbers, re-normalised to length 1",
                  f"TODO 1: truncate gave {getattr(t, 'shape', t)} — slice the LEADING dims, then divide by the norm")
    try:
        bad = type(truncate(v, 300)).__name__ + " returned"
        r2 = False
    except ValueError:
        r2, bad = True, None
    except Exception as e:  # noqa: BLE001
        r2, bad = False, f"{type(e).__name__}"
    good &= check(r2, "truncate(v, 300) raises ValueError — only 768/512/256/128 were trained",
                  f"TODO 1b: truncate(v, 300) should raise ValueError (got: {bad})")
    try:
        s = [storage_bytes(1_000_000, 768, "float32"), storage_bytes(1_000_000, 256, "int8"),
             storage_bytes(1_000_000, 128, "binary")]
    except Exception as e:  # noqa: BLE001
        s = f"error: {e}"
    good &= check(s == [3_072_000_000, 256_000_000, 16_000_000],
                  "storage_bytes: 1M × 768 fp32 = 3.07 GB · 256 int8 = 256 MB · 128 binary = 16 MB",
                  f"TODO 2: storage_bytes gave {s}")
    try:
        p = pack_bits(np.array([0.3, -0.1, 0.0, 2.0, -5.0, 0.1, -0.2, 0.9], dtype=np.float32))
        h = hamming(np.array([0b10110101], dtype=np.uint8), np.array([0b10011101], dtype=np.uint8))
    except Exception as e:  # noqa: BLE001
        p, h = f"error: {e}", None
    good &= check(isinstance(p, np.ndarray) and p.dtype == np.uint8 and p.tolist() == [0b10010101] and h == 2,
                  "pack_bits: positive → 1, else 0, 8 per byte · hamming counts differing bits",
                  f"TODO 3: pack_bits → {p!r}, hamming → {h!r} (want [149] and 2)")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. These three are a complete edge vector store's storage layer.")


if __name__ == "__main__":
    main()
