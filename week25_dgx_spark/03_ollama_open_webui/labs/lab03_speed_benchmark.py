#!/usr/bin/env python3
"""Lab 03-3 · Streaming speed: time-to-first-token, tokens/second, and the bandwidth ceiling.

Streams the same prompt a few times through Ollama's OpenAI API and measures, on the client:
TTFT (how long until the first token arrives) and decode speed (tokens ÷ time after the first
token). Then it asks the native API for the server's own count (eval_count ÷ eval_duration),
and compares the measured speed with Module 01's ceiling: memory bandwidth ÷ bytes read per
token. Spark numbers are compared with the Spark's 273 GB/s; laptop stand-in numbers only with
the laptop's own bandwidth (set LAPTOP_BW_GBS), never with the Spark.

Run: .venv/bin/python week25/03_ollama_open_webui/labs/lab03_speed_benchmark.py
     LAPTOP_BW_GBS=410 .venv/bin/python …   # your laptop's memory bandwidth, from its spec sheet
"""
import os
import statistics
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (SPEC, banner, bar, chat, http_json, note, pick_laptop_model, resolve, result,  # noqa: E402
                      step, table, warn)

SPARK_MODEL = "gpt-oss:20b"
LAPTOP_MODEL = "gemma3:4b"            # dense and non-thinking: the cleanest thing to time
RUNS, MAX_TOKENS = 3, 160
PROMPT = [{"role": "user", "content": "In about 120 words, explain why memory bandwidth limits how fast "
                                      "a language model generates tokens."}]
# Mixture-of-Experts models read only their active experts per token: (total B, active B), from the model cards.
ACTIVE = {"gpt-oss:20b": (21.0, 3.6), "gpt-oss:120b": (117.0, 5.1), "qwen3.6:35b-a3b": (35.0, 3.0)}
PLAYBOOK_SIZE_GB = {"gpt-oss:20b": 15}  # "about 15 GB for gpt-oss:20b" — Open WebUI playbook, prerequisites

base, src = resolve("ollama")
if src == "dry":
    banner("Lab 03-3 · streaming speed", "no Ollama reachable on the Spark or this laptop")
    print("◈ DRY — start Ollama on the Spark (Section 1) or on this laptop, then run again.")
    sys.exit(0)
model = SPARK_MODEL if src == "spark" else pick_laptop_model([LAPTOP_MODEL])
who = "LIVE on the Spark" if src == "spark" else "LAPTOP STAND-IN (Ollama on this laptop, not the Spark)"
banner("Lab 03-3 · streaming speed: TTFT, tok/s, and the ceiling", f"{model} · {who}")
root = base[:-3]

step(1, "what are we timing?")
tags = {m["name"]: m for m in http_json("GET", f"{root}/api/tags", timeout=10).get("models", [])}
size_gb = tags.get(model, {}).get("size", 0) / 1e9
caps = http_json("POST", f"{root}/api/show", {"model": model}, timeout=30).get("capabilities", [])
extra = {"reasoning_effort": "none"} if "thinking" in caps else {}
total_b, active_b = ACTIVE.get(model, (1.0, 1.0))
per_token_gb = size_gb * active_b / total_b
print(f"◆ {model}: {size_gb:.2f} GB on disk · {'MoE, ' + str(active_b) + 'B of ' + str(total_b) + 'B active' if model in ACTIVE else 'dense (every weight read per token)'}"
      f" · ≈ {per_token_gb:.2f} GB read per token{' · thinking off' if extra else ''}")

step(2, f"stream the same prompt {RUNS + 1} times (run 0 may include loading the model)")
rows, speeds, ttfts = [], [], []
for i in range(RUNS + 1):
    try:
        r = chat(base, model, PROMPT, max_tokens=MAX_TOKENS, extra=extra, echo=(i == 0))
    except (HTTPError, URLError, TimeoutError, OSError) as e:      # a busy or shared server: skip the run
        warn(f"run {i} failed: {e} — the server may be busy loading another model; skipping it")
        continue
    rows.append([f"run {i}" + (" (warm-up)" if i == 0 else ""), f"{r['ttft_ms']:.0f} ms", r["out_tokens"],
                 f"{r['total_ms'] / 1000:.2f} s", f"{r['tok_s']:.1f}", bar(r["tok_s"], 150, 20)])
    if i:
        speeds.append(r["tok_s"])
        ttfts.append(r["ttft_ms"])
table(rows, ["run", "TTFT", "tokens", "total", "tok/s", ""])
if not speeds:
    warn("no timed run succeeded — try again when the server is idle.")
    sys.exit(0)
median_tok_s = statistics.median(speeds)
note(f"median over runs 1–{RUNS}: TTFT {statistics.median(ttfts):.0f} ms · decode {median_tok_s:.1f} tok/s "
     f"(client-side: tokens after the first ÷ time after the first). Source: {who}.")

step(3, "the server's own count (native /api/chat: eval_count ÷ eval_duration)")
try:
    d = http_json("POST", f"{root}/api/chat", {"model": model, "messages": PROMPT, "stream": False,
                                               "options": {"num_predict": MAX_TOKENS}, **({"think": False} if extra else {})},
                  timeout=240)
except (HTTPError, URLError, TimeoutError, OSError) as e:
    warn(f"native call failed: {e}")
    d = {}
ev_n, ev_s = d.get("eval_count", 0), d.get("eval_duration", 1) / 1e9
pe_n, pe_s = d.get("prompt_eval_count", 0), max(d.get("prompt_eval_duration", 1), 1) / 1e9
table([["decode", ev_n, f"{ev_s:.2f} s", f"{ev_n / ev_s:.1f} tok/s"],
       ["prefill (prompt)", pe_n, f"{pe_s:.3f} s", f"{pe_n / pe_s:.0f} tok/s"],
       ["model load", "—", f"{d.get('load_duration', 0) / 1e9:.2f} s", "0 s when it was already in memory"]],
      ["phase", "tokens", "time", "rate"])
note("Prefill processes the whole prompt in parallel, so it is compute-bound and much faster per token. "
     "Decode makes one token at a time and re-reads the weights each time, so it is bandwidth-bound.")

step(4, "compare with the bandwidth ceiling (Module 01)")
if src == "spark":
    bw, where_bw = SPEC["mem_bw_gbs"], "DGX Spark spec"
else:
    bw, where_bw = float(os.environ.get("LAPTOP_BW_GBS", "0") or 0), "LAPTOP_BW_GBS"
if bw and per_token_gb:
    ceiling = bw / per_token_gb
    table([[model, f"{bw:g} GB/s ({where_bw})", f"{per_token_gb:.2f} GB", f"{ceiling:.0f} tok/s",
            f"{median_tok_s:.1f} tok/s", f"{100 * median_tok_s / ceiling:.0f}%"]],
          ["model", "bandwidth", "read/token", "ceiling", "measured", "of ceiling"])
    note("The file size includes weights that text generation never reads (gemma3:4b carries a 0.84 GB vision "
         "encoder — Module 04, lab 02), so the true ceiling is higher. Kernel overheads, sampling and other "
         "requests sharing the server keep real speed below it.")
else:
    print("◆ laptop bandwidth unknown — set LAPTOP_BW_GBS to your laptop's memory bandwidth to compute its ceiling.")
if src != "spark":
    gb = PLAYBOOK_SIZE_GB[SPARK_MODEL] * ACTIVE[SPARK_MODEL][1] / ACTIVE[SPARK_MODEL][0]
    print(f"◆ on the Spark, {SPARK_MODEL} (~{PLAYBOOK_SIZE_GB[SPARK_MODEL]} GB, 3.6B of 21B active) reads ≈ {gb:.1f} GB "
          f"per token → ceiling ≈ {SPEC['mem_bw_gbs'] / gb:.0f} tok/s (arithmetic). Run this lab on the Spark to measure it.")
    note("Do not compare the laptop's tok/s with the Spark's: different chips, different models.")

result("TTFT is what a user feels first (load + prefill); tok/s is what they feel after (bandwidth). "
       "Measure both, and judge tok/s against the ceiling, not against another machine.")
