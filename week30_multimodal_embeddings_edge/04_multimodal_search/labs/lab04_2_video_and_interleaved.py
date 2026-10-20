#!/usr/bin/env python3
"""Lab 04-2 · Video and interleaved records: one vector for a whole incident.

A CCTV clip becomes frames (1 per second by default, 140 tokens each) through the vision encoder. An incident
record can mix text, photos and video in ONE input: placeholders <|image|> and <|video|> in the text are filled
from the matching keys, in order, and the model returns a single 768-d vector. This lab budgets the 8,192-token
context, embeds the SYNTHETIC 10-second plant-room clip and two interleaved incident records, and checks which
text questions find them.

Run: week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_2_video_and_interleaved.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import embedkit as ek  # noqa: E402

import numpy as np  # noqa: E402

COST = {"image": 280, "frame": 140, "audio_s": 25, "ctx": 8192}   # model card, "Context Limits"

ek.banner("Lab 04-2 · video and interleaved records", "CCTV clip + photo + text in one vector · SYNTHETIC media")

ek.step(1, "budget the 8,192-token context (model card token costs)")
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            ek.media("cctv_plantroom.mp4")], capture_output=True, text=True).stdout or 10)
frames = int(dur)                                    # 1 frame per second (default sampling)
rows = [["text (≈ 20 words)", "≈ 30"], ["1 photo", COST["image"]], [f"CCTV clip {dur:.0f}s → {frames} frames", frames * COST["frame"]]]
total = 30 + COST["image"] + frames * COST["frame"]
rows.append(["total", f"{total} of {COST['ctx']}"])
ek.table(rows, ["part", "tokens"])
ek.note(f"room left: ≈ {(COST['ctx'] - total) // COST['frame']} more frames, or "
        f"{(COST['ctx'] - total) // COST['audio_s']} s of audio, in the same record")

ek.step(2, "embed the clip, a still, and two interleaved incident records")
model = ek.load(("text", "image", "audio"))
items = {
    "cctv_plantroom.mp4 (video only)": {"video": ek.media("cctv_plantroom.mp4")},
    "plantroom_normal.jpg (image only)": {"image": ek.media("plantroom_normal.jpg")},
    "incident: B2 haze (text + photo + CCTV)": {
        "text": "Incident 0412, plant room B2, 02:10. Night shift saw this: <|image|> CCTV from the same camera: <|video|>",
        "image": [ek.media("plantroom_smoke.jpg")], "video": ek.media("cctv_plantroom.mp4")},
    "incident: AHU-3 leak (text + photo)": {
        "text": "Incident 0413, basement AHU-3, 06:45. Engineer's photo: <|image|> Drain pan blocked, unit isolated.",
        "image": [ek.media("leak_under_ahu.jpg")]},
}
with ek.timer() as t:
    vecs = model.encode(list(items.values()))
ek.ok(f"{len(items)} records → {vecs.shape} in {t.s:.2f}s")

ek.step(3, "which record answers which question?")
names = list(items)
expect = [("haze appearing on the plant room camera", "incident: B2 haze (text + photo + CCTV)"),
          ("water leaking from an air handler", "incident: AHU-3 leak (text + photo)"),
          ("incident report about smoke at night", "incident: B2 haze (text + photo + CCTV)"),
          ("what does the plant room normally look like", "plantroom_normal.jpg (image only)")]
hits = 0
for q, want in expect:
    v = model.encode(q, prompt_name="SearchQuery")
    top = ek.top_k(v, vecs, k=2)
    got = names[top[0][0]]
    hits += got == want
    print(f"  {'✓' if got == want else '✕'} {q[:44]:44s} → {got[:40]} {top[0][1]:.3f}   (2nd: {names[top[1][0]][:26]} {top[1][1]:.3f})")
(ek.ok if hits == len(expect) else ek.warn)(f"{hits}/{len(expect)} as expected")
if hits < len(expect):
    ek.note("an embedding scores what IS in the input. 'Normally' asks about smoke being ABSENT — the haze record "
            "is still a plant room, plus text, so it wins. Absence and negation need a rule or a classifier.")

ek.step(4, "how close is the video to its still frames?")
s = ek.renorm(vecs) @ ek.renorm(vecs[0])
for n, x in zip(names, s):
    print(f"  {x:.3f}  {n}")
ek.result("one vector per incident keeps the index small; keep the per-file vectors too if you need to say "
          "WHICH photo or clip matched.")
