#!/usr/bin/env python3
"""Lab 06-1 · Engine bake-off: probe every OpenAI endpoint, run the same prompts on each one that is up.

Probes vLLM (:8000), SGLang (:30000), TensorRT-LLM (:8355, and :8123 for the Nemotron Super recipe),
NIM (:8000 — the same port as vLLM, told apart by its model id), llama.cpp (:30080), LM Studio (:1234) and
Ollama (:11434) on your Spark. Every engine that answers gets the SAME prompts with the SAME settings:
one warm-up request (not counted), then each prompt once, temperature 0, the same max_tokens, one request at
a time. The table reports medians of time-to-first-token and decode tok/s.

With no Spark, only this laptop's Ollama is up: it is run as ONE clearly labelled LAPTOP STAND-IN row, so you
see the method working. It is not an engine result and must not be compared with Spark rows.

Run: .venv/bin/python week25/06_sglang_trtllm_nim/labs/lab01_engine_bakeoff.py [--max-tokens 100]
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, PORTS, api_host, banner, chat, mode, models, note, pick_laptop_model,  # noqa: E402
                      result, step, table, up, url, warn)

PROMPTS = [
    "Explain tensor parallelism in two sentences.",            # the TensorRT-LLM playbook's test prompt
    "What is the difference between speed and velocity?",      # the SGLang playbook's multi-turn prompt
    "In three bullet points, why does a hotel chiller plant use more energy at night than it should?",
]
# (engine label, sparkkit kind, port) — NIM shares vLLM's :8000, so it is identified by its model id below.
ENDPOINTS = [("vLLM / NIM", "vllm", PORTS["vllm"]), ("SGLang", "sglang", PORTS["sglang"]),
             ("TensorRT-LLM", "trtllm", PORTS["trtllm"]), ("TensorRT-LLM (Nemotron Super recipe)", "trtllm", 8123),
             ("llama.cpp", "llamacpp", PORTS["llamacpp"]), ("LM Studio", "lmstudio", PORTS["lmstudio"]),
             ("Ollama", "ollama", PORTS["ollama"])]

ap = argparse.ArgumentParser()
ap.add_argument("--max-tokens", type=int, default=100, help="same budget for every engine (≤ 150)")
args = ap.parse_args()
max_tokens = max(16, min(args.max_tokens, 150))


def engine_url(kind: str, port: int) -> str:
    if port == PORTS.get(kind):
        return url(kind)                       # honours SPARK_URL_<KIND> overrides
    h = api_host("a")
    return f"http://{h}:{port}/v1" if h else ""


def bench(base: str, model: str, extra: dict | None) -> dict:
    chat(base, model, [{"role": "user", "content": "Say OK."}], max_tokens=8, temperature=0.0, extra=extra)  # warm-up
    runs = []
    for p in PROMPTS:
        runs.append(chat(base, model, [{"role": "user", "content": p}], max_tokens=max_tokens, temperature=0.0,
                         extra=extra, timeout=240))
    return {"ttft": statistics.median(r["ttft_ms"] for r in runs),
            "tok_s": statistics.median(r["tok_s"] for r in runs),
            "tokens": sum(r["out_tokens"] for r in runs),
            "sample": (runs[0]["text"] or runs[0]["reasoning"] or "").replace("\n", " ")[:48]}


banner("Lab 06-1 · engine bake-off", "same prompts · same settings · one engine at a time · medians")

step(1, "probe every OpenAI endpoint on the Spark")
live = mode() == "live"
rows, up_list = [], []
for name, kind, port in ENDPOINTS:
    base = engine_url(kind, port) if live else ""
    is_up = bool(base) and up(base, 2)
    ids = models(base, timeout=2) if is_up else []
    if is_up and port == PORTS["vllm"] and any(i.startswith("meta/") for i in ids):
        name = "NIM"                           # NIM serves ids like meta/llama-3.1-8b-instruct
    rows.append([name, port, base or "(no Spark host)", "● up" if is_up else "○ down", ", ".join(ids[:2]) or "—"])
    if is_up and ids:
        up_list.append((name, base, ids[0]))
table(rows, ["engine", "port", "base URL", "status", "models"])
note("vLLM and NIM both listen on :8000 — only one of them can run at a time (or map one to -p 8001:8000).")

step(2, f"run {len(PROMPTS)} prompts on each engine that is up (warm-up first · temperature 0 · max_tokens {max_tokens})")
results = []
for name, base, model in up_list:
    print(f"→ {name} · {base} · {model}")
    try:
        results.append([name, model] + list(bench(base, model, None).values()))
    except Exception as e:  # noqa: BLE001 — one broken engine must not stop the bake-off
        warn(f"{name} failed: {type(e).__name__}: {str(e)[:160]}")
if not up_list:
    lm = pick_laptop_model(["gemma3:4b"])
    if lm and up(LAPTOP_OLLAMA):
        print(f"→ no Spark engine is up · running the method on Ollama on THIS laptop ({lm}) as a LAPTOP STAND-IN")
        b = bench(LAPTOP_OLLAMA, lm, {"reasoning_effort": "none"})
        results.append(["LAPTOP STAND-IN (Ollama, this Mac)", lm] + list(b.values()))
    else:
        print("◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):")
        print("│ engine   model   median TTFT   median tok/s\n│ vLLM     …       … ms          …")

step(3, "comparison")
if results:
    table([[r[0], r[1], f"{r[2]:.0f} ms", f"{r[3]:.1f}", r[4], r[5]] for r in results],
          ["engine", "model", "median TTFT", "median tok/s", "tokens", "first answer starts"])
if any(r[0].startswith("LAPTOP") for r in results):
    note("LAPTOP STAND-IN: this row shows the method, not an engine. Its numbers are this Mac's and are never "
         "compared with a Spark engine.")
note("Fair only if every engine served the same model at the same precision (e.g. Llama 3.1 8B Instruct NVFP4 — "
     "it is in the vLLM, SGLang and TensorRT-LLM support lists, and NIM has a DGX Spark container for it).")
note("This is one user at a time. For serving many users, repeat lab 05-3 against each engine's URL.")
result("Start the engines one at a time (Sections 2–5), re-run this lab after each, and keep every table: "
       "your own measurements, on your Spark, decide the engine — not a blog post.")
