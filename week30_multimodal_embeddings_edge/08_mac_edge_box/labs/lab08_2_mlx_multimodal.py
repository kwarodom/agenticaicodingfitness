#!/usr/bin/env python3
"""Lab 08-2 · MLX with photos and voicemails: Apple's own framework, every modality, bf16 vs 4-bit.

Module 04's cross-modal search, now in MLX: six SYNTHETIC hotel photos and three voicemails (one Thai) embedded
with mlx-community's bf16 and 4-bit builds, then the same seven plain-English questions. It also measures how
far each MLX build drifts from the PyTorch original, per modality — the 4-bit model card warns that audio is
the modality quantization hurts most. This lab runs in week30_multimodal_embeddings_edge/.venv-mlx (it re-launches itself there).

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/labs/lab08_2_mlx_multimodal.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import mackit  # noqa: E402

mackit.ensure_mlx()                                      # → week30_multimodal_embeddings_edge/.venv-mlx

import embedkit as ek  # noqa: E402
import numpy as np  # noqa: E402

IMAGES = ["leak_under_ahu.jpg", "plantroom_normal.jpg", "plantroom_smoke.jpg", "thermostat_room1203.png",
          "invoice_bank_change.png", "bms_trend_ahu3.png"]
AUDIO = ["voicemail_flood_en.wav", "voicemail_burning_th.wav", "request_towels_en.wav"]
QUESTIONS = [("water leaking on the floor under equipment", "leak_under_ahu.jpg"),
             ("smoke or haze in the plant room", "plantroom_smoke.jpg"),
             ("a hotel room thermostat that is too hot", "thermostat_room1203.png"),
             ("a suspicious invoice asking to change bank details", "invoice_bank_change.png"),
             ("a guest reports flooding in the bathroom", "voicemail_flood_en.wav"),
             ("a guest needs more towels", "request_towels_en.wav"),
             ("a Thai speaker reports a burning smell from the air conditioner", "voicemail_burning_th.wav")]

# The reference: the same items through PyTorch float32 on the CPU, computed in week30_multimodal_embeddings_edge/.venv-embed.
REF = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import embedkit as ek, torch
m = ek.load(("text", "image", "audio"), dev="cpu", dtype=torch.float32, quiet=True)
imgs, auds, qs = json.loads(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4])
out = {"image": m.encode([{"image": ek.media(f)} for f in imgs]).tolist(),
       "audio": m.encode([{"audio": ek.media(f)} for f in auds]).tolist(),
       "text": m.encode(["task: search result | query: " + q for q in qs]).tolist()}
print("JSON" + json.dumps(out))
"""

ek.banner("Lab 08-2 · MLX with photos and voicemails", "mlx-community bf16 vs 4-bit · every modality · SYNTHETIC media")
ek.step(1, "the reference — PyTorch float32 on the CPU (week30_multimodal_embeddings_edge/.venv-embed)")
qs = [q for q, _ in QUESTIONS]
p = subprocess.run([str(mackit.EMBED_PY), "-c", REF, str(ek.COMMON), json.dumps(IMAGES), json.dumps(AUDIO),
                    json.dumps(qs)], capture_output=True, text=True, timeout=900)
ref = {k: np.array(v) for k, v in json.loads(next(ln for ln in p.stdout.splitlines() if ln.startswith("JSON"))[4:]).items()}
ek.ok(f"reference vectors: {len(IMAGES)} photos · {len(AUDIO)} voicemails · {len(qs)} questions")

rows, names = [], IMAGES + AUDIO
for n, variant in enumerate(("bf16", "4bit"), 2):
    ek.step(n, f"MLX {variant}")
    e = mackit.MLXEmbedder(variant, multimodal=True)
    e.encode_image(ek.media(IMAGES[0]))                  # warm-up
    e.encode_audio(ek.media(AUDIO[0]))
    t0 = time.perf_counter()
    img = np.vstack([e.encode_image(ek.media(f)) for f in IMAGES])
    ti = time.perf_counter() - t0
    t0 = time.perf_counter()
    aud = np.vstack([e.encode_audio(ek.media(f)) for f in AUDIO])
    ta = time.perf_counter() - t0
    qv = e.encode(["task: search result | query: " + q for q in qs])
    index = np.vstack([img, aud])
    hits = 0
    for (q, want), v in zip(QUESTIONS, qv):
        i, s = ek.top_k(v, index, 1)[0]
        hits += names[i] == want
        print(f"  {'✓' if names[i] == want else '✕'} {q[:56]:56s} → {names[i]} {s:.3f}")
    agree = {m: min(ek.cosine(a, b) for a, b in zip(ref[m], x)) for m, x in (("text", qv), ("image", img), ("audio", aud))}
    ek.ok(f"{hits}/{len(QUESTIONS)} first · photo {1000 * ti / len(IMAGES):.0f} ms · voicemail {1000 * ta / len(AUDIO):.0f} ms · "
          f"MLX peak {e.peak_mb():.0f} MB · loaded in {e.load_s:.1f}s")
    rows.append([f"MLX {variant}", f"{e.disk_mb:.0f} MB", f"{hits}/{len(QUESTIONS)}", f"{agree['text']:.4f}",
                 f"{agree['image']:.4f}", f"{agree['audio']:.4f}", f"{1000 * ti / len(IMAGES):.0f} ms",
                 f"{1000 * ta / len(AUDIO):.0f} ms"])
    del e
    import mlx.core as mx
    mx.clear_cache()
    mx.reset_peak_memory()

ek.step(4, "compare")
ek.table(rows, ["build", "weights", "questions right", "min cos text", "min cos photo", "min cos voice",
                "per photo", "per voicemail"])
ek.note("REFERENCE (mlx-community 4-bit card): minimum cosine vs FP32 — text 0.980877 · image 0.990386 · audio 0.974435")
ek.result("MLX runs the whole multimodal model natively on Apple silicon. 4-bit trims 26 % of the weights (text and "
          "audio projection are quantized; vision and audio towers stay bf16) — check the voice column before you "
          "ship it for voicemails.")
