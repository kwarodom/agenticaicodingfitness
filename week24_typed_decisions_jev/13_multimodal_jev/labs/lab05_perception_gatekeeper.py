#!/usr/bin/env python3
"""Lab 13-5 · Jev as the gatekeeper — only pay for slow, expensive perception when the text says you need it.

Five staff-app reports arrive, each with an attachment. Instead of sending every attachment
to a vision/video model (seconds each, and billed per image/frame), ONE Jev call reads all
five report TEXTS and says which attachments are worth looking at. Code then perceives only those.

Run: .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab05_perception_gatekeeper.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perception as P  # noqa: E402
from jevkit import banner, guarded, noul, step, summary, table  # noqa: E402
import mediakit as mk  # noqa: E402

REPORTS = [
    {"text": "Housekeeping: water on the floor next to AHU-2 in plant room B1, photo attached.",
     "file": "leak_under_ahu.jpg", "kind": "image"},
    {"text": "Front desk: guest in 812 asked for two extra towels, no rush. Voice note attached.",
     "file": "request_towels_en.wav", "kind": "audio"},
    {"text": "CCTV motion alert: CAM-03 plant room B2, 10-second clip attached.",
     "file": "cctv_plantroom.mp4", "kind": "video"},
    {"text": "Accounts: routine supplier newsletter with our Q3 price list, image attached.",
     "file": "bms_trend_ahu3.png", "kind": "image"},
    {"text": "Guest in 1203 says the room is too warm; they sent a photo of the thermostat.",
     "file": "thermostat_room1203.png", "kind": "image"},
]

banner("Lab 13-5 · Jev as the perception gatekeeper", f"{len(REPORTS)} reports · one Jev call decides what to look at")

step(1, "ONE Jev call over all report texts (one Noul per report — fan-out)")
qs = guarded({f"look_{i}": noul(
    f"Does `reports[{i}]` describe a possible problem whose seriousness someone should confirm by looking at or "
    f"listening to the attached media before acting? Routine requests and newsletters do not need it.")
    for i in range(len(REPORTS))})
resp = P.ask({"reports": [r["text"] for r in REPORTS]}, qs)
a = P.validate(resp, qs)
summary(resp)

step(2, "code perceives ONLY what Jev flagged (threshold 0.5)")
rows, spent_ms, called = [], 0.0, 0
for i, rep in enumerate(REPORTS):
    look = a[f"look_{i}"]["noul"]
    action, detail = "skip", ""
    if look >= 0.5:
        try:
            if rep["kind"] == "video":
                r = mk.watch(P.MEDIA / rep["file"], "In two sentences: is anything unusual or unsafe happening in this plant-room clip?")
            elif rep["kind"] == "audio":
                r = mk.hear(mk.pick("hear"), P.MEDIA / rep["file"])
            elif "thermostat" in rep["file"]:
                r = mk.see(mk.pick("see"), P.MEDIA / rep["file"], P.SEE_THERMOSTAT, max_tokens=290)
            else:
                r = mk.see(mk.pick("see"), P.MEDIA / rep["file"], P.SEE_SCENE, max_tokens=380)
            called += 1
            spent_ms += r.get("latency_ms") or 0
            action, detail = f"{rep['kind']} → {r.get('provider')}", (r.get("text") or "").replace("\n", " ")[:48]
        except mk.MediaError as e:
            action, detail = "error", str(e)[:48]
    rows.append([i, rep["text"][:52], f"{look:.2f}", action, detail])
table(rows, ["#", "report", "look?", "perception", "what it saw"])

step(3, "what the gate saved")
avg = spent_ms / called if called else 0
skipped = len(REPORTS) - called
print(f"◆ perception calls made: {called} of {len(REPORTS)} · time spent {spent_ms / 1000:.1f} s")
print(f"◆ skipped {skipped} attachment(s) ≈ {skipped * avg / 1000:.1f} s of perception avoided "
      f"(at this run's average of {avg / 1000:.1f} s) · Jev gate cost {resp['usage']['input_tokens']} tokens "
      f"≈ ${resp['usage']['input_tokens'] * 0.042 / 1e6:.6f}")
print("═ Jev reads the cheap part (text) first and decides whether the expensive part (pixels, frames, audio) is needed.")
print("  Tune the 0.5 gate on your own data — a missed hazard costs more than one extra image call.")
