#!/usr/bin/env python3
"""Lab 13-4 · CCTV video → events → one Jev call → alert decision.

  1. watch (plain prompt)   — the video model lists events as JSON text
  2. code                   — parse the JSON
  3. Jev                    — ONE call, one hazard Noul + one kind Choice PER EVENT (fan-out)
  4. code                   — alert if any event is hazardous; earliest time wins
  5. watch (careful prompt) — ask the model for an alternative interpretation (smoke vs steam) and compare

Run: .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab04_video_to_alert.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perception as P  # noqa: E402
from jevkit import banner, step, summary, table  # noqa: E402
import mediakit as mk  # noqa: E402

banner("Lab 13-4 · video → alert", "watch: gemini (video) · decide: Jev · alert: your code")
CLIP = P.MEDIA / "cctv_plantroom.mp4"


def judge(prompt: str, label: str):
    try:
        r = mk.watch(CLIP, prompt, json_output=True)
    except mk.MediaError as e:
        print(f"✕ {e}")
        return None
    mk.show(r, label)
    events = P.parse_events(r["text"])
    if not events:
        print("⚠ no parseable events → human review")
        return None
    qs = P.event_questions(len(events))
    resp = P.ask({"events": events}, qs)
    a = P.validate(resp, qs)
    rows, hazards = [], []
    for i, e in enumerate(events):
        hz, kind = a[f"hazard_{i}"]["noul"], a[f"kind_{i}"]["choice"]
        rows.append([e.get("t_seconds"), e["event"][:62], e.get("could_also_be", "")[:24], f"{hz:.2f}", kind])
        if hz >= 0.5:
            hazards.append((e.get("t_seconds", 0), kind, e["event"]))
    table(rows, ["t (s)", "event (from the video model)", "could also be", "hazard", "Jev kind"])
    summary(resp)
    return hazards


step(1, "plain prompt: list the events")
h1 = judge(P.WATCH_PLAIN, "watch plain")
step(2, "careful prompt: list events AND the most likely alternative interpretation")
h2 = judge(P.WATCH_CAREFUL, "watch careful")

step(3, "code decides")
for label, hz in (("plain", h1), ("careful", h2)):
    if hz is None:
        print(f"→ {label}: perception failed → human review")
    elif hz:
        t, kind, ev = min(hz, key=lambda x: x[0] or 0)
        print(f"→ {label}: ALERT at t={t}s — {kind} — “{ev[:70]}” → page the duty engineer to look in person")
    else:
        print(f"→ {label}: no hazard found")
print("═ Smoke or steam? The camera cannot always tell, and neither can the words the video model chooses.")
print("  So the Jev hazard question covers smoke, steam AND haze, and the policy sends a person to look either way.")
print("═ execute: False — the alert is a proposal; the fire-alarm system stays independent of this pipeline.")
