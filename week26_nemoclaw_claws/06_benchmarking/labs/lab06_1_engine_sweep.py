#!/usr/bin/env python3
"""Lab 06-1 · Engine benchmark: the bandwidth ceiling, a vLLM concurrency sweep, and a laptop stand-in sweep.

Layer 1 of 4 (research tutorial Part 5, Lab 5.1). Three parts:
  1. Arithmetic: the single-stream decode ceiling for a 3B-active MoE on 273 GB/s, per weight format.
  2. The Spark: `vllm bench serve`, single stream then concurrency 2/4/8/16 (read-only; DRY → EXAMPLE shape).
  3. THIS laptop: the same method against Ollama (concurrency 1, 2, 4), measured for real — a LAPTOP STAND-IN.
     Other labs may share this Ollama, so its numbers are noisy. They teach the method; they are not Spark numbers.
The third-party Spark figures from the research tutorial are printed with their sources, never as your results.

Run: .venv/bin/python week26/06_benchmarking/labs/lab06_1_engine_sweep.py [--model gemma3:4b] [--max-tokens 96]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import (LAPTOP_OLLAMA, SPEC, banner, bar, chat, decode_ceiling_tok_s, note, ok, pick_laptop_model,  # noqa: E402
                     result, sh, step, table, up, warn, weights_gb)
from benchkit import percentile, save_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="", help="laptop Ollama model for the stand-in sweep (default: gemma3:4b if pulled)")
ap.add_argument("--max-tokens", type=int, default=96)
args = ap.parse_args()

banner("Lab 06-1 · engine benchmark", "ceiling arithmetic · vllm bench serve on the Spark · a laptop stand-in sweep")

# ── 1 · the ceiling ──────────────────────────────────────────────────────────────
step(1, f"the single-stream ceiling: every decoded token reads every ACTIVE weight once ({SPEC['mem_bw_gbs']} GB/s)")
ACTIVE_B = 3.0                           # Nemotron 3 Nano 30B-A3B: ~3B parameters active per token
rows = []
for fmt in ("bf16", "fp8", "nvfp4"):
    gb = weights_gb(ACTIVE_B, fmt)
    ceil = decode_ceiling_tok_s(ACTIVE_B, fmt)
    rows.append([fmt, f"{gb:.2f} GB", f"{ceil:.0f} tok/s", bar(ceil, 180, 24)])
table(rows, ["format", "active weights / token", "naive ceiling", ""])
note("Naive = weights only, at peak bandwidth. Real decode also reads the KV cache and the non-expert layers, "
     "and never reaches peak bandwidth. That is why ai-muninn (cited in the research tutorial) puts the practical "
     "single-stream ceiling for a 3B-active MoE on the Spark 'in the low 80s tok/s regardless of engine tricks'.")

# ── 2 · third-party reference points (quoted, with sources) ───────────────────────
step(2, "third-party Spark figures, as quoted in the research tutorial (reference points to reproduce, not specs)")
table([
    ["Nemotron 3 Nano 30B-A3B · vLLM · W4A16 NVFP4", "single-stream", "74.75 tok/s", "ai-muninn"],
    ["same · W4A4 NVFP4", "single / aggregate c=16", "58.27 / 786 tok/s", "ai-muninn"],
    ["same · W4A16", "aggregate", "~400 tok/s", "ai-muninn"],
    ["Nemotron 3 Nano NVFP4 · vLLM", "single-stream", "65+ tok/s", "NVIDIA developer forum"],
    ["nemotron-3-nano:30b · Ollama", "avg over agent tasks", "64.7 tok/s", "Exxact benchmark"],
    ["nemotron-3-super:120b-a12b · Ollama", "avg tok/s · task pass", "16.4 · 17/17", "Exxact benchmark"],
    ["gemma4:26b · Ollama vs vLLM", "single-stream", "~64 vs ~30 tok/s", "Exxact engines"],
    ["gemma4:26b · vLLM", "aggregate, 10+ concurrent", ">300 tok/s", "Exxact engines"],
    ["NAT ReAct + 1 tool · Nemotron 3 Nano FP8", "end-to-end per query", "~13 s", "Classmethod"],
], ["model / engine", "metric", "value", "source (per the research tutorial)"])
note("Each is one machine, one software version, measured by its author. Reproduce them on your Spark; do not "
     "quote them as yours.")

# ── 3 · the Spark: vllm bench serve (read-only) ───────────────────────────────────
step(3, "the Spark — vLLM's built-in benchmark: single stream, then a concurrency sweep")
BENCH = ("docker exec vllm-nat vllm bench serve --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 "
         "--backend openai-chat --endpoint /v1/chat/completions --host 127.0.0.1 --port 8000 "
         "--dataset-name random --random-input-len 512 --random-output-len 256")
EXAMPLE_BENCH = """============ Serving Benchmark Result ============
Successful requests:                     32
Maximum request concurrency:             1
Benchmark duration (s):                  …
Output token throughput (tok/s):         …
Mean TTFT (ms):                          …
Mean TPOT (ms):                          …
=================================================="""
live = sh("docker ps --filter name=vllm-nat --format '{{.Names}} {{.Status}}'",
          example="vllm-nat Up … (started in Module 04, research tutorial Lab 3.2)", timeout=30).live
sh(f"{BENCH} --num-prompts 32 --max-concurrency 1", example=EXAMPLE_BENCH, timeout=900)
sh(f"for c in 2 4 8 16; do {BENCH} --num-prompts $((c*8)) --max-concurrency $c 2>&1 | "
   "grep -E 'concurrency|Output token throughput|Mean TTFT'; done",
   example="Maximum request concurrency:             2\nOutput token throughput (tok/s):         …\n"
           "Mean TTFT (ms):                          …\n… (one block per concurrency: 2, 4, 8, 16)", timeout=1800)
if not live:
    note("DRY: the blocks above are EXAMPLE shapes (the field names vLLM prints), with no numbers. Your Spark fills them.")

# ── 4 · the same method on THIS laptop ────────────────────────────────────────────
step(4, "LAPTOP STAND-IN — the same sweep against Ollama on this Mac (concurrency 1, 2, 4)")
model = args.model or pick_laptop_model(["gemma3:4b"])
if not model or not up(LAPTOP_OLLAMA):
    warn("no laptop Ollama model — start Ollama (`ollama serve`) and pull gemma3:4b, then re-run this step")
    result("The ceiling arithmetic and the Spark commands above are the lab. Re-run with Ollama up for the sweep.")
    sys.exit(0)
PROMPT = [{"role": "user", "content": "In four short bullet points, explain why a hotel chiller plant's kW/RT "
                                      "gets worse at low cooling load."}]
EXTRA = {"reasoning_effort": "none"}
mt = max(32, min(args.max_tokens, 160))
print(f"→ POST {LAPTOP_OLLAMA}/chat/completions · model={model} · stream · max_tokens={mt} · temperature 0 · "
      "Ollama on THIS laptop (stand-in, not the Spark)")
chat(LAPTOP_OLLAMA, model, [{"role": "user", "content": "Say OK."}], max_tokens=8, extra=EXTRA)   # warm-up, not counted
ok("warm-up done (model loaded; not counted)")

LEVELS = {1: 2, 2: 2, 4: 4}                  # concurrency → number of requests (9 calls in total with the warm-up)
sweep = []
for c, n in LEVELS.items():
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        runs = list(ex.map(lambda _: chat(LAPTOP_OLLAMA, model, PROMPT, max_tokens=mt, temperature=0.0,
                                          extra=EXTRA, timeout=300), range(n)))
    wall = time.perf_counter() - t0
    toks = sum(r["out_tokens"] for r in runs)
    row = {"concurrency": c, "requests": n, "ttft_p50_ms": percentile([r["ttft_ms"] for r in runs], 50),
           "per_request_tok_s": sum(r["tok_s"] for r in runs) / n, "aggregate_tok_s": toks / wall,
           "out_tokens": toks, "wall_s": wall}
    sweep.append(row)
    print(f"◆ c={c}: {n} requests · {toks} tokens in {wall:.1f}s · TTFT p50 {row['ttft_p50_ms']:.0f} ms · "
          f"{row['per_request_tok_s']:.1f} tok/s per request · {row['aggregate_tok_s']:.1f} tok/s aggregate")

vmax = max(r["aggregate_tok_s"] for r in sweep)
table([[r["concurrency"], r["requests"], f"{r['ttft_p50_ms']:.0f} ms", f"{r['per_request_tok_s']:.1f}",
        f"{r['aggregate_tok_s']:.1f}", bar(r["aggregate_tok_s"], vmax, 20)] for r in sweep],
      ["concurrency", "requests", "TTFT p50", "tok/s per request", "tok/s aggregate", ""])
gain = sweep[-1]["aggregate_tok_s"] / max(0.1, sweep[0]["aggregate_tok_s"])
busy = sweep[0]["ttft_p50_ms"] > 1500
if busy:
    warn(f"TTFT at c=1 was {sweep[0]['ttft_p50_ms'] / 1000:.1f} s for a one-line prompt: this Ollama was busy (another "
         "lab's requests or a model load were ahead in the queue). Decode tok/s per request stayed "
         f"≈ {sweep[0]['per_request_tok_s']:.0f}, so the time went into WAITING, not decoding. The aggregate column is "
         "not a batching result on this run. Re-run when the laptop is quieter.")
elif gain > 1.3:
    note(f"Aggregate rose ×{gain:.1f} from c=1 to c=4 while each request slowed: the server batches requests. "
         "This is the same shape vLLM shows on the Spark, at a different scale.")
else:
    note(f"Aggregate changed only ×{gain:.1f}: this Ollama is queueing requests (OLLAMA_NUM_PARALLEL) or other labs "
         "are using it. The research tutorial cites Exxact: Ollama wins single-user latency, vLLM wins multi-user "
         "throughput.")
warn("LAPTOP STAND-IN: these numbers are this Mac's, with other labs possibly sharing its Ollama. They show the "
     "method; never compare them with the Spark figures above.")
save_summary("engine", {"model": model, "max_tokens": mt, "sweep": sweep, "ollama_busy": busy})
result("Ceiling first, then single stream, then the sweep. On the Spark, run step 3 on vLLM AND Ollama with the "
       "same prompts, so you have both engines on one prompt set.")
