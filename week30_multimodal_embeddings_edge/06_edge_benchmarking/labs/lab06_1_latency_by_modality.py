#!/usr/bin/env python3
"""Lab 06-1 · Latency by modality: p50 / p95 for text, photos, voice and video — GPU and CPU.

An edge engineer quotes percentiles, not one lucky run. This lab warms the model up, then times every
modality repeatedly on the GPU (MPS, bfloat16) and on the CPU (float32, the model card's CPU advice): one
short text, a batch of 32 texts, one photo, one 12-second voicemail, one 10-second CCTV clip. Numbers are a
LAPTOP STAND-IN — measured on this machine, not on a Pi or a phone.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/06_edge_benchmarking/labs/lab06_1_latency_by_modality.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import torch  # noqa: E402

REPEATS = {"text ×1": 20, "text ×32": 8, "photo": 8, "voicemail 12 s": 6, "CCTV 10 s": 4}

ek.banner("Lab 06-1 · latency by modality", "p50 / p95 · MPS bfloat16 vs CPU float32 · LAPTOP STAND-IN")
kb = ek.load_kb()
batch = [f"title: {d['title']} | text: {d['text']}" for d in kb["docs"]] * 2
batch = (batch * 2)[:32]
WORK = {
    "text ×1": "task: search result | query: water under the air handler",
    "text ×32": batch,
    "photo": [{"image": ek.media("leak_under_ahu.jpg")}],
    "voicemail 12 s": [{"audio": ek.media("voicemail_burning_th.wav")}],
    "CCTV 10 s": [{"video": ek.media("cctv_plantroom.mp4")}],
}
results = {}
for n, (dev, dtype) in enumerate((("mps", torch.bfloat16), ("cpu", torch.float32)), 1):
    if dev == "mps" and not torch.backends.mps.is_available():
        ek.warn("no MPS here — CPU only")
        continue
    ek.step(n, f"{dev} · {str(dtype).replace('torch.', '')}")
    model = ek.load(("text", "image", "audio"), dev=dev, dtype=dtype)
    rows = []
    for name, work in WORK.items():
        model.encode(work)                                   # warm-up (kernels, caches)
        times = []
        for _ in range(REPEATS[name]):
            t0 = time.perf_counter()
            model.encode(work)
            if dev == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        p50, p95 = ek.percentile(times, 50), ek.percentile(times, 95)
        per = p50 / (32 if name == "text ×32" else 1)
        results.setdefault(name, {})[dev] = {"p50_ms": p50, "p95_ms": p95}
        rows.append([name, len(times), f"{p50:.0f} ms", f"{p95:.0f} ms", f"{per:.1f} ms", ek.bar(p50, 3000, 18)])
    ek.table(rows, ["work", "runs", "p50", "p95", "per item", ""])
    del model
    ek._MODELS.clear()

ek.step(3, "GPU vs CPU")
rows = []
for name, r in results.items():
    if "mps" in r and "cpu" in r:
        rows.append([name, f"{r['mps']['p50_ms']:.0f} ms", f"{r['cpu']['p50_ms']:.0f} ms",
                     f"{r['cpu']['p50_ms'] / r['mps']['p50_ms']:.1f}×"])
ek.table(rows, ["work", "MPS p50", "CPU p50", "CPU / MPS"])
out = ek.runs(__file__) / "latency.json"
out.write_text(json.dumps({"machine": ek.machine(), "results": results}, indent=2))
ek.note(f"saved → week30_multimodal_embeddings_edge/06_edge_benchmarking/.runs/{out.name} (Lab 06-3 reads it)")
ek.result("batching amortises the fixed cost: look at 'per item' for ×32. Media costs more than text because "
          "each photo is 280 tokens and each second of video 140 — the same token maths as Lab 04-2.")
