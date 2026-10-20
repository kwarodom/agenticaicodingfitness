#!/usr/bin/env python3
"""Lab 05-3 · Continuous batching: send 1, 2, 4 and 8 requests at once and compare total vs per-stream tok/s.

One decode step reads every active weight from memory once, whether it produces a token for one
conversation or for eight. vLLM's continuous batching adds new requests to the batch that is already
running, so the same memory read serves more users: total tokens per second climbs with concurrency while
each user's own speed drops only a little — until compute or KV cache runs out.

This lab measures that curve against vLLM on your Spark. Without one, it runs the SAME client code against
Ollama on this laptop, labelled LAPTOP STAND-IN: the numbers are real but they are your laptop's, and
Ollama batches differently from vLLM (it serves OLLAMA_NUM_PARALLEL requests at a time and queues the rest).
Never compare the two machines' numbers; compare the SHAPE of each curve.

Run: .venv/bin/python week25/05_vllm/labs/lab03_continuous_batching.py [--levels 1,2,4,8] [--max-tokens 64]
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, bar, chat, models, note, pick_laptop_model, resolve, result, step,  # noqa: E402
                      table, warn)

TOPICS = ["a chiller plant", "a hotel lobby", "a cooling tower", "an air handling unit",
          "a parking garage", "a server room", "a rooftop solar array", "a hotel kitchen"]

ap = argparse.ArgumentParser()
ap.add_argument("--levels", default="1,2,4,8", help="concurrency levels (max 8: the laptop stand-in is shared)")
ap.add_argument("--max-tokens", type=int, default=64, help="tokens per request (≤ 150)")
ap.add_argument("--model", default="", help="model id (default: what the server lists / a small laptop model)")
args = ap.parse_args()
levels = [max(1, min(8, int(x))) for x in args.levels.split(",") if x.strip()]
max_tokens = max(8, min(args.max_tokens, 150))

banner("Lab 05-3 · continuous batching — does total throughput grow with concurrency?",
       "same prompt shape at 1, 2, 4, 8 parallel requests · threads + sparkkit.chat")

base, src = resolve("vllm")
if src == "dry":
    print("◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):")
    print("│ parallel  total tok/s  per-stream tok/s\n│ 1         x            x\n│ 8         ~k·x         a bit below x")
    result("No vLLM endpoint and no laptop Ollama answered. Start lab 05-2 on your Spark, or Ollama here.")
    sys.exit(0)
if src == "spark":
    model = args.model or (models(base) or ["default"])[0]
    label = f"vLLM on the Spark · {base}"
    extra = None
else:
    model = args.model or pick_laptop_model(["gemma3:4b"])
    label = "LAPTOP STAND-IN · Ollama on this Mac (not Spark numbers)"
    extra = {"reasoning_effort": "none"}
print(f"◆ endpoint: {label} · model {model} · max_tokens {max_tokens}")


def one(i: int, out: list) -> None:
    msgs = [{"role": "user", "content": f"In exactly three sentences, describe {TOPICS[i % len(TOPICS)]} "
                                        "to a new facilities engineer."}]
    try:
        out[i] = chat(base, model, msgs, max_tokens=max_tokens, temperature=0.0, extra=extra, timeout=240)
    except Exception as e:  # noqa: BLE001 — one failed stream should not hide the others
        out[i] = {"error": f"{type(e).__name__}: {e}"}


def run(n: int) -> dict:
    out: list = [None] * n
    threads = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    good = [r for r in out if r and "error" not in r]
    toks = sum(r["out_tokens"] for r in good)
    return {"n": n, "ok": len(good), "wall": wall, "tokens": toks,
            "total_tok_s": toks / wall if wall else 0.0,
            "stream_tok_s": sum(r["tok_s"] for r in good) / len(good) if good else 0.0,
            "ttft": sum(r["ttft_ms"] for r in good) / len(good) if good else 0.0,
            "errors": [r["error"] for r in out if r and "error" in r]}


step(1, "warm-up (one request, not counted: loads the model and fills caches)")
w = run(1)
if w["errors"]:
    warn(f"warm-up failed: {w['errors'][0][:200]}")
    sys.exit(0)
print(f"│ warm-up: {w['tokens']} tokens in {w['wall']:.1f}s")

step(2, f"measure at concurrency {', '.join(map(str, levels))}")
rows_raw = []
for n in levels:
    r = run(n)
    rows_raw.append(r)
    print(f"│ {n} parallel → {r['tokens']} tokens in {r['wall']:.1f}s"
          + (f"  ⚠ {len(r['errors'])} failed" if r["errors"] else ""))

step(3, "total vs per-stream throughput")
base_total = rows_raw[0]["total_tok_s"] or 1.0
vmax = max(r["total_tok_s"] for r in rows_raw) or 1.0
table([[r["n"], f"{r['ok']}/{r['n']}", f"{r['total_tok_s']:6.1f}", f"{r['stream_tok_s']:6.1f}",
        f"{r['ttft']:7.0f} ms", f"×{r['total_tok_s'] / base_total:4.2f}", bar(r["total_tok_s"], vmax, 20)]
       for r in rows_raw],
      ["parallel", "ok", "total tok/s", "per-stream tok/s", "mean TTFT", "vs 1", "total"])
top = rows_raw[-1]
gain = top["total_tok_s"] / base_total
if rows_raw[0]["ttft"] > 1000:
    print(f"⚠ measured: the solo request waited {rows_raw[0]['ttft']:.0f} ms for its first token — the server was busy "
          "with other clients, so the 'vs 1' ratios are not trustworthy. Run the lab again when it is idle.")
elif gain >= 1.5:
    print(f"◆ measured: total throughput grew ×{gain:.1f} from 1 to {top['n']} parallel requests, while each stream "
          f"kept {top['stream_tok_s'] / (rows_raw[0]['stream_tok_s'] or 1) * 100:.0f}% of its solo speed — batching at work.")
else:
    print(f"◆ measured: total throughput grew only ×{gain:.2f} from 1 to {top['n']} parallel, while mean TTFT went from "
          f"{rows_raw[0]['ttft']:.0f} ms to {top['ttft']:.0f} ms — most requests waited in a queue instead of joining "
          "one batch. Run it twice: a shared machine is noisy.")
if src == "laptop":
    note("LAPTOP STAND-IN: these are this Mac's numbers, shared with anything else using Ollama right now. "
         "Ollama serves OLLAMA_NUM_PARALLEL requests together and queues the rest — watch TTFT grow when it queues.")
else:
    note("vLLM on the Spark: per-stream tok/s should fall slowly while total tok/s climbs — that is continuous "
         "batching sharing each weight read across every sequence in the batch.")
note("total tok/s = all tokens ÷ wall-clock time. per-stream tok/s = one user's decode speed, averaged. "
     "Both are client-side measurements that include network and queueing.")
result("A serving engine is judged by the total curve (users served per box); a chat UI by the per-stream "
       "column and TTFT. Module 01's bandwidth ceiling limits per-stream speed, not the total.")
