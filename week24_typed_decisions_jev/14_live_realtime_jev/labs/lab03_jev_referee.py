#!/usr/bin/env python3
"""Lab 14-3 · Jev referees every turn — catch promises and invented facts.

Two live turns with the NAIVE prompt, then ONE Jev request per turn with five
typed checks on what the caller said (heard) and what the assistant answered (said):

    caller_emergency · needs_human · assistant_promised_action · assistant_invented_fact · caller_language

The live model is fluent. It is not an authority: it has no pager, no booking system
and no facts beyond what you give it. Jev turns "did it overstep?" into numbers code can act on.

Run: .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab03_jev_referee.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, show, step  # noqa: E402
import livekit  # noqa: E402

SYSTEM = R.system_for(R.NAIVE_SYSTEM, "L14-3")
TURNS = [("smoke call (voice)", {"audio": R.CALL_TURN2}), ("breakfast question (text)", {"text": "Hi, what time is breakfast tomorrow?"})]

banner("Lab 14-3 · Jev referees every turn", "gemini-3.8-live talks · Jev checks · code decides")
n = 0
for name, kw in TURNS:
    n += 1
    step(n, f"live turn — {name}")
    t = R.turn("gemini", system=SYSTEM, **kw)
    if t.get("error"):
        print(f"✕ gemini: {t['error']} — skipping this turn")
        continue
    print(f"← heard: {t['heard'] or '(typed) ' + kw.get('text', '')}")
    print(f"→ said : {t['said']}")
    n += 1
    step(n, f"Jev referees the turn — {name}")
    ref = R.referee(t["heard"] or kw.get("text", ""), t["said"])
    show(ref["resp"], top=2)
    state, actions = R.next_state("calm", ref["answers"], t["heard"] or kw.get("text", ""))
    print(f"→ call state: calm → {state}")
    for a in actions:
        print(f"→ proposed: {a}")
    if not actions:
        print("→ proposed: nothing — the turn is within bounds")
print("\n═ execute: False — every action above is a PROPOSAL for a real system (pager, TTS line),")
print("  never something the live model did or should do on its own.")
