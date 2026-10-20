#!/usr/bin/env python3
"""Lab 04-1 · Cross-modal search: one text question finds the right photo or voicemail.

Loads every encoder (740M), embeds six SYNTHETIC hotel images and three SYNTHETIC voicemails into the SAME
768-d space as text, then asks seven plain-English questions. No captions, no transcripts, no tags: the
question vector is compared directly with the image and audio vectors. Media gets no task prefix (model card:
"Prefixes apply to text only"); questions use the SearchQuery prompt.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_1_cross_modal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

IMAGES = ["leak_under_ahu.jpg", "plantroom_normal.jpg", "plantroom_smoke.jpg", "thermostat_room1203.png",
          "invoice_bank_change.png", "bms_trend_ahu3.png"]
AUDIO = ["voicemail_flood_en.wav", "voicemail_burning_th.wav", "request_towels_en.wav"]
QUESTIONS = [
    ("water leaking on the floor under equipment", "leak_under_ahu.jpg"),
    ("smoke or haze in the plant room", "plantroom_smoke.jpg"),
    ("a hotel room thermostat that is too hot", "thermostat_room1203.png"),
    ("a suspicious invoice asking to change bank details", "invoice_bank_change.png"),
    ("a guest reports flooding in the bathroom", "voicemail_flood_en.wav"),
    ("a guest needs more towels", "request_towels_en.wav"),
    ("a Thai speaker reports a burning smell from the air conditioner", "voicemail_burning_th.wav"),
]

ek.banner("Lab 04-1 · cross-modal search", "text → images and voicemails in one 768-d space · SYNTHETIC media")
ek.step(1, "load all encoders")
model = ek.load(("text", "image", "audio"))

ek.step(2, "embed the media — no captions, no transcripts")
with ek.timer() as ti:
    img = model.encode([{"image": ek.media(f)} for f in IMAGES])
with ek.timer() as ta:
    aud = model.encode([{"audio": ek.media(f)} for f in AUDIO])
ek.ok(f"{len(IMAGES)} images in {ti.s:.2f}s ({1000 * ti.s / len(IMAGES):.0f} ms each) · "
      f"{len(AUDIO)} voicemails (8.8–12.1 s each) in {ta.s:.2f}s ({1000 * ta.s / len(AUDIO):.0f} ms each)")
names = IMAGES + AUDIO
index = np.vstack([img, aud])

ek.step(3, "seven questions, ranked against all nine items")
qv = model.encode([q for q, _ in QUESTIONS], prompt_name="SearchQuery")
hits = 0
for (q, want), v in zip(QUESTIONS, qv):
    top = ek.top_k(v, index, k=3)
    first = names[top[0][0]]
    hits += first == want
    mark = "✓" if first == want else "✕"
    print(f"  {mark} {q[:58]:58s} → " + "  ".join(f"{names[i]} {s:.3f}" for i, s in top[:2]))
(ek.ok if hits == len(QUESTIONS) else ek.warn)(f"{hits}/{len(QUESTIONS)} questions found the right item first")

ek.step(4, "image ↔ image: which photo is closest to the normal plant room?")
sims = ek.renorm(img) @ ek.renorm(img[1])
for i in np.argsort(-sims)[1:4]:
    print(f"  {sims[i]:.3f}  {IMAGES[i]}")
ek.result("the smoky plant room sits right next to the normal one — near-duplicate detection for free, "
          "and a hint why 'smoke' queries must score the difference, not the room.")
