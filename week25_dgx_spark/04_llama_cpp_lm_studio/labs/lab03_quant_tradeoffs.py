#!/usr/bin/env python3
"""Lab 04-3 · Quantization trade-offs: fewer bits, smaller files, bigger errors.

Pure arithmetic, runs anywhere (no Spark needed). Part 1 quantizes one million synthetic
weights (Gaussian, with 0.1 % outliers like real LLM weights) using ggml's own Q8_0, Q4_0 and
Q4_1 recipes, plus symmetric 6-, 3- and 2-bit variants of the same recipe, and measures the
reconstruction error. Part 2 shows why GGUF quantizes in small blocks. Part 3 turns bits per
weight into file size, memory headroom and the decode ceiling for the playbook's model,
Qwen3.6-35B-A3B, on one DGX Spark.

Honest scope: this measures how well the NUMBERS survive, not how well the MODEL answers.
Model quality needs a benchmark on the real model.

Run: .venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab03_quant_tradeoffs.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import BITS, SPEC, banner, bar, note, result, step, table  # noqa: E402

rng = np.random.default_rng(25)
N = 1 << 20                                           # 1,048,576 weights = 32,768 blocks of 32
w = rng.normal(0, 0.02, N).astype(np.float32)
out = rng.choice(N, N // 1000, replace=False)
w[out] *= 10                                          # a few large weights, as real checkpoints have


def q_sym(x: np.ndarray, bits: int, block: int) -> np.ndarray:
    """Q8_0-style: one fp16 scale per block, d = absmax / (2^(bits-1) - 1), values rounded to integers."""
    b = x.reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    d = (np.abs(b).max(axis=1, keepdims=True) / qmax).astype(np.float16).astype(np.float32)
    d[d == 0] = 1
    return (np.clip(np.round(b / d), -qmax, qmax) * d).reshape(-1)


def q4_0(x: np.ndarray, block: int = 32) -> np.ndarray:
    """ggml Q4_0: d = (signed value with the largest magnitude) / -8, q = x/d + 8 in 0..15, x ≈ (q - 8)·d."""
    b = x.reshape(-1, block)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(len(b)), idx][:, None]
    d = (mx / -8).astype(np.float16).astype(np.float32)
    d[d == 0] = 1
    q = np.clip(np.floor(b / d + 8.5), 0, 15)
    return ((q - 8) * d).reshape(-1)


def q4_1(x: np.ndarray, block: int = 32) -> np.ndarray:
    """ggml Q4_1: a scale AND a minimum per block, d = (max - min) / 15, x ≈ q·d + min."""
    b = x.reshape(-1, block)
    lo, hi = b.min(axis=1, keepdims=True), b.max(axis=1, keepdims=True)
    d = ((hi - lo) / 15).astype(np.float16).astype(np.float32)
    lo = lo.astype(np.float16).astype(np.float32)
    d[d == 0] = 1
    q = np.clip(np.round((b - lo) / d), 0, 15)
    return (q * d + lo).reshape(-1)


def err(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rel = float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x ** 2)))
    return rel, -20 * np.log10(rel)


banner("Lab 04-3 · quantization trade-offs", "arithmetic only · no Spark needed · the same on every machine",
       status=False)

step(1, f"quantize {N:,} weights with each recipe and measure the error")
# (name, bits per weight incl. the scale, function, where the recipe comes from)
RECIPES = [("F16", 16.0, lambda x: x.astype(np.float16).astype(np.float32), "half precision, no blocks"),
           ("Q8_0", 8.5, lambda x: q_sym(x, 8, 32), "ggml Q8_0: 32 × int8 + fp16 scale"),
           ("sym-6", 6.5, lambda x: q_sym(x, 6, 32), "same recipe, 6-bit values"),
           ("Q4_1", 5.0, q4_1, "ggml Q4_1: 32 × 4-bit + fp16 scale + fp16 min"),
           ("Q4_0", 4.5, q4_0, "ggml Q4_0: 32 × 4-bit + fp16 scale"),
           ("sym-3", 3.5, lambda x: q_sym(x, 3, 32), "same recipe, 3-bit values"),
           ("sym-2", 2.5, lambda x: q_sym(x, 2, 32), "same recipe, 2-bit values")]
rows = []
for name, bpw, fn, what in RECIPES:
    rel, snr = err(w, fn(w))
    rows.append([name, f"{bpw:g}", f"{N * bpw / 8 / 1e6:5.2f} MB", f"{100 * rel:6.2f}%", f"{snr:5.1f} dB",
                 bar(min(rel, 0.6), 0.6, 18), what])
table(rows, ["recipe", "bits/w", "size", "rel. error", "SNR", "", "how"])
note("Each bit you remove roughly doubles the error (≈ 6 dB per bit). Q4_1 spends half a bit more than Q4_0 "
     "on a per-block minimum and gets a lower error; K-quants (Q4_K, Q6_K) refine the same idea with 256-weight "
     "super-blocks and 6-bit sub-scales.")

step(2, "why blocks: one outlier ruins the scale for everyone who shares it")
rows = []
for block in (32, 256, 4096, N):
    rel, snr = err(w, q_sym(w, 4, block))
    extra_bits = 16 / block
    rows.append([f"{block:,}" if block < N else "whole tensor", f"{4 + extra_bits:.3f}", f"{100 * rel:6.2f}%",
                 f"{snr:5.1f} dB", bar(min(rel, 0.6), 0.6, 18)])
table(rows, ["weights per scale", "bits/w", "rel. error", "SNR", ""])
note("A per-tensor scale is stretched by the largest outlier, so small weights round to zero. A block of 32 "
     "costs 0.5 bit for its fp16 scale and keeps each outlier's damage local.")

step(3, "what it means for the playbook's model: Qwen3.6-35B-A3B on one Spark")
PARAMS_B, ACTIVE_B = 35.0, 3.0                      # "35B-A3B": 35B total, ~3B active per token (the name says so)
# effective bits/weight: sparkkit's averages for the mixed K-quant recipes (lab 02 measures one: 4.82 in the layers);
# Q2_K is its block value (84 B / 256 weights) — a real Q2_K file mixes in larger types and ends up bigger.
QUANTS = [("BF16", BITS["bf16"]), ("Q8_0", BITS["q8_0"]), ("Q6_K", BITS["q6_k"]), ("Q5_K_M", BITS["q5_k_m"]),
          ("Q4_K_M", BITS["q4_k_m"]), ("Q3_K_M", BITS["q3_k_m"]), ("Q2_K", 84 * 8 / 256)]
rows = []
for q, bpw in QUANTS:
    gb = PARAMS_B * bpw / 8
    left = SPEC["memory_gb"] - gb
    ceiling = SPEC["mem_bw_gbs"] / (ACTIVE_B * bpw / 8)
    rows.append([q, f"{bpw:g}", f"{gb:5.1f} GB", f"{left:5.1f} GB" if left > 0 else "✕ does not fit",
                 f"{ceiling:5.0f} tok/s", bar(gb, SPEC["memory_gb"], 16)])
table(rows, ["quant", "bits/w", "weights", "left of 128 GB", "decode ceiling", "share of memory"])
note("The playbook's pick is unsloth's UD-Q4_K_XL (an Unsloth 'dynamic' ~4-bit recipe that keeps more tensors at "
     "higher precision). The playbook budgets about 30 GB of memory and a ~35 GB-order download for it.")
note("On a Spark, even BF16 of this 35B model fits. You quantize here for SPEED (fewer bytes per token) and for "
     "room: KV cache for long contexts and several models loaded side by side.")

result("Q8_0 is near-lossless for the numbers; Q4_K_M halves the size again with a visible but usually acceptable "
       "error; below 4 bits the error grows fast. The playbooks' default — Q4_K_M or NVFP4 — sits at the knee.")
