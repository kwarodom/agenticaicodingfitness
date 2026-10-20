#!/usr/bin/env python3
"""Lab 07-1 · NVFP4 simulator: quantize real numbers to 4 bits and measure the error.

Pure Python (standard library only), runs anywhere, no Spark needed. It builds the
E2M1 grid (the 4-bit values NVFP4 stores), quantizes one 16-element block by hand,
then quantizes 65,536 weight-like numbers five ways — FP8 E4M3, NVFP4, MXFP4,
INT4 per-tensor and INT4 group-128 — and prints the real error of each. The random
seed is fixed, so every machine prints the same table.

Run: .venv/bin/python week25/07_nvfp4_speculative/labs/lab01_nvfp4_simulator.py
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, note, result, step, table  # noqa: E402


# ── minifloat rounding: one function for E2M1 (FP4) and E4M3 (FP8) ────────────
def minifloat_round(x: float, exp_bits: int, man_bits: int, max_val: float) -> float:
    """Round x to the nearest value of a small float format (round-half-even, saturate at max_val)."""
    if x == 0 or math.isnan(x):
        return 0.0
    bias = 2 ** (exp_bits - 1) - 1
    emin = 1 - bias                                   # smallest normal exponent; below it: subnormals
    a = min(abs(x), max_val)
    e = max(math.floor(math.log2(a)), emin)
    step_ = 2.0 ** (e - man_bits)                     # spacing of representable values in this binade
    q = min(round(a / step_) * step_, max_val)
    return math.copysign(q, x)


def e2m1(x: float) -> float:                          # FP4: 1 sign, 2 exponent, 1 mantissa bit → max 6
    return minifloat_round(x, 2, 1, 6.0)


def e4m3(x: float) -> float:                          # FP8: 1 sign, 4 exponent, 3 mantissa bits → max 448
    return minifloat_round(x, 4, 3, 448.0)


# ── the five formats ─────────────────────────────────────────────────────────
def fp8_per_tensor(xs):
    s = max(map(abs, xs)) / 448.0
    return [e4m3(v / s) * s for v in xs]


def nvfp4(xs, block=16):
    """NVFP4: E2M1 values · one FP8 (E4M3) scale per 16 values · one FP32 scale per tensor."""
    g = max(map(abs, xs)) / (6.0 * 448.0)             # second-level FP32 scale keeps block scales in FP8 range
    out = []
    for i in range(0, len(xs), block):
        blk = xs[i:i + block]
        s = e4m3(max(map(abs, blk)) / 6.0 / g) * g    # block scale, itself rounded to FP8
        out += [e2m1(v / s) * s if s else 0.0 for v in blk]
    return out


def mxfp4(xs, block=32):
    """MXFP4 (OCP Microscaling): E2M1 values · one power-of-two (E8M0) scale per 32 values."""
    out = []
    for i in range(0, len(xs), block):
        blk = xs[i:i + block]
        amax = max(map(abs, blk))
        s = 2.0 ** (math.floor(math.log2(amax)) - 2) if amax else 0.0   # 2 = largest E2M1 exponent
        out += [e2m1(v / s) * s if s else 0.0 for v in blk]
    return out


def int4(xs, group=None):
    """Symmetric INT4 (-8…7). group=None → one scale for the whole tensor; else one FP16 scale per group."""
    group = group or len(xs)
    out = []
    for i in range(0, len(xs), group):
        blk = xs[i:i + group]
        s = max(map(abs, blk)) / 7.0
        out += [max(-8, min(7, round(v / s))) * s if s else 0.0 for v in blk]
    return out


def errors(xs, ys):
    se = sum((a - b) ** 2 for a, b in zip(xs, ys))
    sig = sum(a * a for a in xs)
    rmse = math.sqrt(se / len(xs))
    return {"rmse": rmse, "rel": math.sqrt(se / sig), "sqnr": 10 * math.log10(sig / se) if se else float("inf"),
            "max": max(abs(a - b) for a, b in zip(xs, ys))}


FORMATS = [  # name, function, effective bits per value INCLUDING scales
    ("FP8 E4M3 (per-tensor)", fp8_per_tensor, 8.0),
    ("NVFP4 (16-blk FP8 scale)", nvfp4, 4 + 8 / 16),
    ("MXFP4 (32-blk 2^k scale)", mxfp4, 4 + 8 / 32),
    ("INT4 group-128 (FP16 sc.)", lambda xs: int4(xs, 128), 4 + 16 / 128),
    ("INT4 per-tensor", int4, 4.0),
]

banner("Lab 07-1 · NVFP4 simulator", "quantize 65,536 numbers five ways · pure Python · same numbers on every machine",
       status=False)

step(1, "the E2M1 grid: every magnitude a 4-bit NVFP4 value can hold")
grid = sorted({abs(e2m1(i / 8)) for i in range(0, 60)})
print("│ " + "  ".join(f"{g:g}" for g in grid) + "   (and their negatives: 15 distinct values, ±0 share a code)")
note("Only 8 magnitudes, spaced unevenly: 0.5 apart near zero, 2 apart at the top. That is floating point — "
     "precision where the values are small. INT4 spaces its 16 levels evenly instead.")

step(2, "one 16-value block by hand")
rng = random.Random(7)
blk = [round(rng.gauss(0, 0.02), 4) for _ in range(16)]
blk[5] = 0.0913                                              # one outlier, as real weight rows have
amax = max(map(abs, blk))
TENSOR_AMAX = 0.25                                            # the largest |value| in the whole weight matrix
g = TENSOR_AMAX / (6 * 448)                                   # per-tensor FP32 scale
s_raw = amax / 6 / g
s_fp8 = e4m3(s_raw)
print(f"│ amax of block = {amax}  →  block scale = amax / 6 = {amax / 6:.6f}")
print(f"│ per-tensor FP32 scale = tensor amax {TENSOR_AMAX} / (6 × 448) = {g:.3e}")
print(f"│ block scale in FP8 units = {s_raw:.2f} → rounded to E4M3 = {s_fp8:g}  (FP8 has 3 mantissa bits)")
rows = []
for v in blk[:8]:
    code = e2m1(v / (s_fp8 * g))
    rows.append([f"{v:+.4f}", f"{v / (s_fp8 * g):+.3f}", f"{code:+g}", f"{code * s_fp8 * g:+.4f}", f"{abs(v - code * s_fp8 * g):.4f}"])
table(rows, ["value", "÷ scale", "E2M1 code", "decoded", "abs error"])
note("The first 8 of 16 values shown. The largest value lands near ±6, the top of the grid (not exactly: "
     "the FP8 scale was rounded); the rest fall on the grid below it. 16 × 4 bits + one 8-bit scale = 72 bits = 4.5 bits per value.")

step(3, "65,536 weight-like numbers, five formats")
N = 65_536
rng = random.Random(2025)
gauss = [rng.gauss(0, 0.02) for _ in range(N)]
spiky = [v * (12 if rng.random() < 0.005 else 1) for v in gauss]    # 0.5 % outliers ×12, like real LLM rows
rows, sqnr = [], {}
for name, fn, bits in FORMATS:
    eg, es = errors(gauss, fn(gauss)), errors(spiky, fn(spiky))
    sqnr[name] = (eg["sqnr"], es["sqnr"])
    rows.append([name, f"{bits:.3f}", f"{eg['rel'] * 100:5.2f} %", f"{es['rel'] * 100:5.2f} %",
                 f"{eg['sqnr']:5.1f} dB", f"{es['sqnr']:5.1f} dB"])
table(rows, ["format", "bits/value", "rel. err (gauss)", "rel. err (outliers)", "SQNR gauss", "SQNR outliers"])
print()
for name, (_, s_out) in sqnr.items():
    print(f"│ {name:26s} {bar(s_out, 35)} {s_out:5.1f} dB (outliers)")
note("SQNR = signal-to-quantization-noise ratio; each +6 dB ≈ one more bit of real precision. "
     "Relative error = ‖x − x̂‖ / ‖x‖.")

step(4, "what 4.5 bits buys: an 8B model's weights")
for name, _, bits in [("BF16 (no quantization)", None, 16.0)] + FORMATS[:2]:
    gb = 8.03e9 * bits / 8 / 1e9
    print(f"│ {name:26s} {gb:5.1f} GB  {bar(gb, 16.1)}")
print(f"│ BF16 ÷ NVFP4 = {16 / 4.5:.2f}×   FP8 ÷ NVFP4 = {8 / 4.5:.2f}×   (the playbook says ~3.5× and ~1.8×)")

result("Per-block scales are why NVFP4 beats per-tensor INT4 at the same size: one outlier only hurts its own "
       "16 neighbours. The FP8 scale (not a power of two, unlike MXFP4) fits each block more tightly. "
       "Quantization error is not quality loss yet — Section 4 measures the model.")
