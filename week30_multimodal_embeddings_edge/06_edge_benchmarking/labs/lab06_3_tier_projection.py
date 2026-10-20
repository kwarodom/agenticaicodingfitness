#!/usr/bin/env python3
"""Lab 06-3 · Tier projection: bandwidth floor vs compute — which one limits an EMBEDDING model?

Tina Huang's conveyor belt says an LLM is limited by memory bandwidth: to write each new token it must stream
all its weights again. An embedding model is different: it reads a whole input (dozens to thousands of tokens)
in ONE pass, so the weights cross the bus once while the maths grows with the token count. This lab computes
both limits — the bandwidth floor (weight bytes ÷ bandwidth) and the work (≈ 2 × transformer parameters ×
tokens) — puts them next to what this machine measured in Lab 06-1, and plans an index build from MEASURED
numbers. Every number for another device is labelled ESTIMATE.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_3_tier_projection.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

# peak memory bandwidth, GB/s (RESEARCH.md §4, vendor specs)
DEVICES = [("Raspberry Pi 5 (CPU only)", 17), ("DGX Spark (GB10)", 273), ("AMD Ryzen AI Max+ 395", 256),
           ("Apple M4 Max 14-core (this Mac, peak)", 410), ("RTX 4090", 1008), ("H100 SXM", 3350)]
WEIGHTS_MB = {"text Q4 GGUF": 176, "text Q8_0 GGUF": 310, "text bf16": 540, "full bf16": 1480}
TRANSFORMER_PARAMS = 130e6        # model card: 130M transformer + 140M embedder (a lookup table: no matmuls)

ek.banner("Lab 06-3 · tier projection", "bandwidth floor vs compute · measured here · other devices = ESTIMATE")

ek.step(1, "bandwidth floor per forward pass (ms) — weights cross the bus at least once")
rows = []
for dev, bw in DEVICES:
    rows.append([dev, f"{bw} GB/s"] + [f"{mb / 1024 / bw * 1000:.1f}" for mb in WEIGHTS_MB.values()])
ek.table(rows, ["device", "peak bandwidth"] + list(WEIGHTS_MB))
ek.note("ESTIMATE — a lower bound from vendor peak bandwidth, never a measurement of that device")

ek.step(2, "this machine: measured time vs the bandwidth floor vs the work")
lat = ek.runs(__file__) / "latency.json"
if not lat.exists():
    ek.warn("run Lab 06-1 first — it saves this machine's measured latencies")
    sys.exit(0)
res = json.loads(lat.read_text())["results"]
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(ek.MODEL_ID)
kb = ek.load_kb()
batch = ([f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]] * 4)[:32]
n_tokens = {"text ×1": len(tok("task: search result | query: water under the air handler")["input_ids"]),
            "text ×32": sum(len(tok(t)["input_ids"]) for t in batch)}
floor_ms = WEIGHTS_MB["text bf16"] / 1024 / 410 * 1000
rows = []
for name in ("text ×1", "text ×32"):
    if "mps" not in res.get(name, {}):
        continue
    ms = res[name]["mps"]["p50_ms"]
    flops = 2 * TRANSFORMER_PARAMS * n_tokens[name]
    rows.append([name, n_tokens[name], f"{ms:.1f} ms", f"{floor_ms:.1f} ms", f"{ms / floor_ms:.0f}×",
                 f"{flops / 1e9:.1f} GFLOP", f"{flops / (ms / 1000) / 1e12:.2f} TFLOP/s"])
ek.table(rows, ["work (MPS bf16)", "tokens", "measured p50", "bandwidth floor", "measured ÷ floor", "work",
                "achieved"])
ek.note("measured is 10–100× the bandwidth floor: an encoder is NOT limited by the conveyor belt. At batch 1 fixed "
        "overheads dominate; at batch 32 the arithmetic grows with the tokens. The belt limits LLM decoding, where "
        "every new token re-reads all the weights — the video's case.")

ek.step(3, "index plan for 50,000 hotel notes — from MEASURED per-item times on this machine")
rows = []
for name, dev in (("text ×32", "mps"), ("text ×32", "cpu")):
    if dev in res.get(name, {}):
        per_item_s = res[name][dev]["p50_ms"] / 32 / 1000
        rows.append([f"this machine · {dev} (measured)", f"{per_item_s * 1000:.1f} ms",
                     f"{50_000 * per_item_s / 60:.1f} min"])
pi_floor = 50_000 / 32 * (WEIGHTS_MB["text Q8_0 GGUF"] / 1024 / 17)
rows.append(["Raspberry Pi 5 · bandwidth floor only (ESTIMATE)", "—", f"≥ {pi_floor:.0f} s, real: far more"])
ek.table(rows, ["where", "per note", "50,000 notes"])
ek.result("for embeddings, plan with measured per-item time and the device's COMPUTE (CPU/GPU/NPU), not its "
          "bandwidth alone. Even slow, it is a nightly batch job: re-embed what changed, search all day.")
