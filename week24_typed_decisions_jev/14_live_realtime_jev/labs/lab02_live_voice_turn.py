#!/usr/bin/env python3
"""Lab 14-2 · A voice turn and its timeline — stream a caller's audio, watch the model answer.

We play a synthetic caller ("…smoke coming out of the AC vent… burning plastic!")
into both live models exactly like a phone line would: 100 ms chunks of PCM audio.
Gemini streams in real time and uses server-side voice-activity detection (VAD);
OpenAI's realtime API lets us commit the audio ourselves, so we stream it faster.

Read the timeline: when the caller stopped, when the model started answering,
and the gap between the two — the latency budget every live product lives in.

Run: .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab02_live_voice_turn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, step, table  # noqa: E402
import livekit  # noqa: E402

SYSTEM = R.system_for(R.NAIVE_SYSTEM, "L14-2")
PLAN = [("gemini", 1.0, "real-time pace — server VAD needs a natural stream + trailing silence"),
        ("chatgpt", 0.3, "3× faster than real time — we commit the audio buffer ourselves")]

banner("Lab 14-2 · a voice turn and its timeline", f"caller audio: {R.CALL_TURN2.name} (synthetic)")
rows = []
for n, (provider, pace, why) in enumerate(PLAN, 1):
    step(n, f"{provider} · {livekit.MODELS[provider]} · pace {pace} — {why}")
    t = R.turn(provider, audio=R.CALL_TURN2, system=SYSTEM, pace=pace)
    if t.get("error"):
        print(f"✕ {provider}: {t['error']} — skipping")
        continue
    livekit.show(t)
    tm = t["timing"]
    rows.append([provider, f"{tm['speech_end']:.2f}s", f"{tm['first_output']:.2f}s", f"{R.gap(t):.2f}s",
                 f"{tm['total']:.2f}s", t["said"][:48]])

step(3, "the latency budget, side by side")
table(rows, ["live model", "caller stopped", "first output", "gap", "turn done", "said"])
print("◆ 'gap' = how long the caller hears silence after they stop talking. Humans notice > ~1 s.")
print("◆ Gemini's gap includes its server VAD waiting for ~1–1.5 s of silence after the last word;")
print("  OpenAI's gap starts at our manual commit (no VAD wait). Same caller, different turn-taking.")
print("═ execute: False — the models only talked; nothing was dispatched.")
