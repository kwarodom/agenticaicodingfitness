#!/usr/bin/env python3
"""Lab 13-1 · The capability map — which model can see, hear, speak or watch, and how fast Jev is next to them.

Step 1 prints which provider can do which capability ON THIS MACHINE (keys only, no calls).
Step 2 makes ONE call per capability (see · hear · speak · watch) plus ONE Jev call on
the text, and puts the latencies side by side.

Run: .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab01_capability_map.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perception as P  # noqa: E402
from jevkit import banner, show, step, table  # noqa: E402
import mediakit as mk  # noqa: E402

banner("Lab 13-1 · capability map", "the model perceives · Jev judges · code decides")

step(1, "who can do what on this machine (from your keys — no API calls)")
caps = list(mk.DEFAULTS)
provs = sorted({p for c in caps for p in mk.DEFAULTS[c]})
rows = []
for p in provs:
    rows.append([p] + [("✓ " + mk.DEFAULTS[c][p]) if p in mk.capable(c) else ("· no key" if p in mk.DEFAULTS[c] else "—")
                       for c in caps])
table(rows, ["provider"] + caps)
print("│ live (streaming voice): gemini-3.8-live · gpt-realtime-2.1-mini → Module 14")
print("│ also on these keys: gemini-omni-1.1-flash (any-to-any) · gpt-image-* / veo / sora (generation) — not used here")

step(2, "one call per capability, then one Jev call on the resulting text")
timings = []


def timed(label, fn):
    try:
        r = fn()
    except mk.MediaError as e:
        print(f"✕ {label}: {e}")
        return None
    mk.show(r, label)
    timings.append([label, r.get("provider"), r.get("model"), f"{(r.get('latency_ms') or 0) / 1000:.2f} s", r.get("_source")])
    return r


see = timed("see", lambda: mk.see(mk.pick("see"), P.MEDIA / "thermostat_room1203.png", P.SEE_THERMOSTAT, max_tokens=280))
heard = timed("hear", lambda: mk.hear(mk.pick("hear"), P.MEDIA / "request_towels_en.wav"))
spoke = timed("speak", lambda: mk.speak(mk.pick("speak"), "Your towels are on the way to room 812.",
                                        P.RUNS / "hello_speak.wav"))
if spoke and spoke.get("file"):
    print(f"🎧 saved → week24_typed_decisions_jev/13_multimodal_jev/.runs/hello_speak.wav")
timed("watch", lambda: mk.watch(P.MEDIA / "cctv_plantroom.mp4", "In one sentence: what happens in this CCTV clip?"))

transcript = P.clean_transcript((heard or {}).get("text") or "Could someone bring two extra towels to room 812?")
resp = P.ask({"transcript": transcript}, P.VOICE_QUESTIONS)
show(resp, top=2)
timings.append(["Jev (decide)", "typesafe", resp.get("model"), f"{(resp.get('_latency_ms') or 0) / 1000:.2f} s",
                resp.get("_source")])

step(3, "latency side by side")
table(timings, ["step", "provider", "model", "time", "source"])
print("═ Perception turns media into text in seconds; Jev turns that text into typed decisions in about half a second.")
print("  Only the perception step depends on the media type — the Jev step and your policy stay the same.")
