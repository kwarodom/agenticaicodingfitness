#!/usr/bin/env python3
"""Lab 14-1 · Your first live turn — one typed question, two live models, every stream event.

A live (real-time) model keeps a websocket open and streams its answer while it is
still generating. Here we send ONE text question to each live provider and print
the stream: when the session opened, when the first audio/text arrived, and what
the assistant said. Notice what it says about breakfast — we never told it the hours.

Run: .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab01_live_hello.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, step  # noqa: E402
import livekit  # noqa: E402

QUESTION = "Hello, what time is breakfast?"
SYSTEM = R.system_for(R.NAIVE_SYSTEM, "L14-1")      # the naive prompt: no hotel facts at all

banner("Lab 14-1 · your first live turn", "text in · streamed answer out · two live providers")
print(f"» caller (typed): {QUESTION}")
print(f"» system prompt : {R.NAIVE_SYSTEM}")

results = []
for n, provider in enumerate(["gemini", "chatgpt"], 1):
    step(n, f"{provider} · {livekit.MODELS[provider]} — open a session, send one turn, stream the reply")
    t = R.turn(provider, text=QUESTION, system=SYSTEM)
    if t.get("error"):
        print(f"✕ {provider}: {t['error']} — skipping")
        continue
    livekit.show(t)
    results.append(t)

step(3, "what did we just see?")
for t in results:
    tm = t["timing"]
    print(f"◆ {t['provider']:<8} first output {tm['first_output']:.2f}s after we opened the socket "
          f"(answer began {R.gap(t):.2f}s after the question was sent)")
    print(f"   │ said: {t['said'][:110]}")
print("⚠ Nobody gave the model the breakfast hours. Any time it states is INVENTED — a live model")
print("  improvises confidently. Lab 14-3 makes Jev catch this on every turn.")
print("═ execute: False — nothing was dispatched; this lab only opens and closes two sessions.")
