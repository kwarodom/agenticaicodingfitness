#!/usr/bin/env python3
"""Lab 02-4 · Link cost: what the 200 Gb/s cable costs a tensor-parallel model, per token.

Pure arithmetic, runs anywhere. With tensor parallelism across two Sparks (TP=2) each Spark
holds half of every layer, so each token reads only half the weights from memory — but every
layer ends with two all-reduces over the QSFP link. This lab adds both up for five models:

    per-token time ≈ (active weight bytes ÷ 2) ÷ 273 GB/s  +  2 × layers × (α + message ÷ link busbw)

α is the fixed latency of one all-reduce. The playbooks publish no latency number, so α is an
ASSUMPTION here (swept over 10, 30, 100 µs). Measure yours with the playbook's ib_write_lat
and pass it:  --alpha-us 25   ·   pass your NCCL busbw from lab03:  --busbw 22.4

Run: .venv/bin/python week25/02_two_sparks_nccl/labs/lab04_tp_link_cost.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import SPEC, banner, bar, note, result, step, table, weights_gb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--busbw", type=float, default=21.875, help="link bus bandwidth, GB/s (default: the playbook's pass mark)")
ap.add_argument("--alpha-us", type=float, default=None, help="measured all-reduce latency in µs (else a sweep)")
args = ap.parse_args()

BW_MEM = SPEC["mem_bw_gbs"]                 # 273 GB/s per Spark
LINK = args.busbw                           # GB/s across the QSFP link (all-reduce bus bandwidth)
ALPHAS = [args.alpha_us] if args.alpha_us is not None else [10.0, 30.0, 100.0]

# name, total params (B), active params per token (B), layers, hidden size
MODELS = [
    ("Llama 3.1 8B",          8.0,   8.0,  32,  4096),
    ("Llama 3.3 70B",        70.6,  70.6,  80,  8192),
    ("gpt-oss-120b (MoE)",  117.0,   5.1,  36,  2880),
    ("Qwen3 235B-A22B (MoE)", 235.0, 22.0, 94,  4096),
    ("Llama 3.1 405B",      405.0, 405.0, 126, 16384),
]
FMT = "nvfp4"

banner("Lab 02-4 · what the link costs tensor parallelism",
       f"arithmetic only · link busbw {LINK:g} GB/s · memory {BW_MEM} GB/s per Spark · weights {FMT}", status=False)

step(1, "the two bandwidths side by side")
print(f"│ memory, per Spark  {bar(BW_MEM, BW_MEM)} {BW_MEM:7.1f} GB/s")
print(f"│ QSFP link (busbw)  {bar(LINK, BW_MEM)} {LINK:7.2f} GB/s   ({BW_MEM / LINK:.0f}× slower)")
note("So you never stream WEIGHTS over the cable. Tensor parallelism keeps each weight on one Spark and sends "
     "only small activations — two all-reduces per layer.")

step(2, "bytes that cross the link per decoded token (batch 1, bf16 activations)")
rows = []
for name, _, _, L, h in MODELS:
    msg = h * 2                                     # one all-reduce message: hidden × 2 bytes
    per_tok = 2 * L * msg                           # two all-reduces per layer
    rows.append([name, L, h, f"{msg / 1024:.1f} KiB", 2 * L, f"{per_tok / 1e6:.2f} MB",
                 f"{per_tok / (LINK * 1e9) * 1e6:.0f} µs"])
table(rows, ["model", "layers", "hidden", "message", "all-reduces", "per token", "wire time"])
note("For n = 2 a ring all-reduce moves 2(n−1)/n = 1× the message per Spark, so wire time = bytes ÷ busbw.")

step(3, "per-token time: one Spark vs TP=2 (single stream, upper-bound arithmetic)")
for alpha in ALPHAS:
    label = "measured" if args.alpha_us is not None else "assumed"
    print(f"\n◆ α = {alpha:g} µs per all-reduce ({label})")
    rows = []
    for name, p_total, p_active, L, h in MODELS:
        w = weights_gb(p_active, FMT)
        fits_one = weights_gb(p_total, FMT) + 10 <= SPEC["memory_gb"]
        t1 = w / BW_MEM * 1000 if fits_one else None                          # ms, one Spark
        comm = 2 * L * (alpha * 1e-6 + h * 2 / (LINK * 1e9)) * 1000           # ms
        t2 = (w / 2) / BW_MEM * 1000 + comm
        rows.append([name, f"{t1:6.1f} ms" if t1 else "   — (too big)", f"{t2:6.1f} ms",
                     f"{comm:5.1f} ms", f"{t1 / t2:4.2f}×" if t1 else "fits only on 2",
                     f"{1000 / t2:6.1f}"])
    table(rows, ["model", "1 Spark", "TP=2", "of which link", "speed-up", "TP=2 tok/s ceiling"])

step(4, "prefill is different: a 4,096-token prompt sends 4,096× the bytes")
rows = []
for name, _, _, L, h in MODELS:
    b = 2 * L * h * 2 * 4096
    rows.append([name, f"{b / 1e9:6.2f} GB", f"{b / (LINK * 1e9):5.2f} s"])
table(rows, ["model", "all-reduce bytes", "wire time at busbw"])
note("Long prompts on TP=2 spend real time on the cable. Serving engines overlap some of it with compute; "
     "Module 05 measures what vLLM actually achieves.")

result("Two Sparks buy CAPACITY first: 256 GB holds Llama 405B or Qwen3 235B at 4-bit. For speed, TP=2 is at "
       "most 2× per token, the link latency eats into it on every layer, and the smaller or sparser the model, "
       "the less you gain. Measure α and busbw, then rerun with --alpha-us and --busbw.")
