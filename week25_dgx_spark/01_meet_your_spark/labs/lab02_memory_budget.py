#!/usr/bin/env python3
"""Lab 01-2 · Memory budget: which models fit on one Spark, which need two?

Pure arithmetic, runs anywhere (no Spark needed). For six open models it adds up
weights + KV cache at bf16, fp8 and nvfp4, compares the total with one Spark
(128 GB unified) and two Sparks (256 GB), and estimates the single-stream decode
ceiling from memory bandwidth. These numbers drive every choice this week.

Run: .venv/bin/python week25/01_meet_your_spark/labs/lab02_memory_budget.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (SPEC, banner, bar, decode_ceiling_tok_s, kv_cache_gb, note, result, step,  # noqa: E402
                      table, weights_gb)

banner("Lab 01-2 · memory budget — will it fit in 128 GB?",
       "arithmetic only · no Spark needed · the same on every machine", status=False)

# name, total params (B), active params per token (B; MoE < total), layers, KV heads, head dim
MODELS = [
    ("Llama 3.1 8B",       8.0,   8.0,  32, 8, 128),
    ("Qwen3 32B",         32.8,  32.8,  64, 8, 128),
    ("Llama 3.3 70B",     70.6,  70.6,  80, 8, 128),
    ("gpt-oss-120b (MoE)", 117.0,  5.1,  36, 8,  64),
    ("Qwen3 235B-A22B (MoE)", 235.0, 22.0, 94, 4, 128),
    ("Llama 3.1 405B",   405.0, 405.0, 126, 8, 128),
]
CTX, BATCH = 32_768, 1              # one 32K-token conversation
OVERHEAD_GB = 10                    # OS, CUDA context, activations, the serving engine itself
ONE, TWO = SPEC["memory_gb"], 2 * SPEC["memory_gb"]

step(1, f"weights + KV cache ({CTX // 1024}K context, batch {BATCH}) + {OVERHEAD_GB} GB overhead")
rows = []
for name, p_total, p_active, layers, kvh, hd in MODELS:
    kv = kv_cache_gb(layers, kvh, hd, CTX, BATCH)          # bf16 KV cache
    cells = [name]
    for fmt in ("bf16", "fp8", "nvfp4"):
        need = weights_gb(p_total, fmt) + kv + OVERHEAD_GB
        where = "1 Spark" if need <= ONE else ("2 Sparks" if need <= TWO else "✕ too big")
        cells.append(f"{need:6.0f} GB  {where}")
    rows.append(cells)
table(rows, ["model", "bf16", "fp8", "nvfp4"])

step(2, "why the KV cache matters — the same 70B model at longer contexts")
for ctx in (8_192, 32_768, 131_072):
    kv = kv_cache_gb(80, 8, 128, ctx, 1)
    print(f"│ Llama 3.3 70B · {ctx // 1024:>3}K context  KV cache {kv:5.1f} GB  {bar(kv, 50)}")
note("KV cache grows linearly with context × concurrent users. Serving engines reserve it up front "
     "(vLLM's --gpu-memory-utilization, SGLang's --mem-fraction-static).")

step(3, f"decode speed ceiling — {SPEC['mem_bw_gbs']} GB/s of memory bandwidth, one stream")
rows = []
for name, p_total, p_active, *_ in MODELS:
    ceil_bf16 = decode_ceiling_tok_s(p_active, "bf16")
    ceil_fp4 = decode_ceiling_tok_s(p_active, "nvfp4")
    rows.append([name, f"{p_active:g}B active", f"{ceil_bf16:6.1f} tok/s", f"{ceil_fp4:6.1f} tok/s"])
table(rows, ["model", "read per token", "bf16 ceiling", "nvfp4 ceiling"])
note("Each new token reads every ACTIVE weight once, so tok/s ≤ bandwidth ÷ bytes of active weights. "
     "Real engines land below this ceiling; batching many users raises total throughput, not this number.")

result("MoE models (gpt-oss-120b, Qwen3 235B-A22B) are the Spark's sweet spot: big total size for quality, "
       "few active weights for speed. Llama 405B needs two Sparks and 4-bit weights — Module 02.")
