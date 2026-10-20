#!/usr/bin/env python3
"""Lab 12-1 · The finished guest-request copilot, end to end.

Runs copilot_reference.handle() on 9 fictional guest messages (EN / TH / mixed,
including a safety case, a negation, an angry guest and a vague "hello?"):

    code: language + room number   →   Jev: ONE batched call (5 questions)
    →   code: policy (safety override, uncertainty gate, priority)
    →   code: ticket JSON + templated reply draft   ·   execute: False

Run: .venv/bin/python week24/12_capstone_hotel_copilot/labs/lab01_capstone_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copilot_reference as copilot                     # noqa: E402
from jevkit import banner, step, table                  # noqa: E402

banner("Lab 12-1 · smart-hotel guest-request copilot",
       f"{len(copilot.MESSAGES)} messages · 1 Jev call each · 5 questions per call")

step(1, "the questions — five independent judgments, sent together")
for qid, q in copilot.QUESTIONS.items():
    size = "yes/no" if q["type"] == "noul" else f"{len(q['criteria'])} {'options' if q['type'] == 'choice' else 'levels'}"
    print(f"» {qid:<14} {q['type']:<6}  {size}")
print("│ language and room number are NOT questions: code gets them exactly, for free")

step(2, "run every message through the pipeline")
tickets = []
for msg in copilot.MESSAGES:
    t = copilot.handle(msg)
    tickets.append((msg, t))
table([[m if len(m) <= 38 else m[:37] + "…", t["language"], t["room"] or "—", t["route"], t["priority"],
        "yes" if t["escalate_to_human"] else "", ",".join(t["follow_ups"])] for m, t in tickets],
      ["message", "lang", "room", "route", "prio", "human", "follow-ups"])

step(3, "why each non-obvious row went where it did")
for m, t in tickets:
    s = t["signals"]
    if t["route"] in ("duty_manager", "front_desk_review") or t["escalate_to_human"] or t["follow_ups"]:
        print(f"→ “{m[:50]}{'…' if len(m) > 50 else ''}”")
        print(f"   {t['reason']} · dept {s['department']} p={s['department_p']} · severity {s['severity']}"
              f" · safety {s['safety']} · needs_human {s['needs_human']} · mentions_room {s['mentions_room']}")

step(4, "one complete ticket — this is what a downstream system would receive")
safety = next((t for _, t in tickets if t["route"] == "duty_manager"), tickets[0][1])
print(json.dumps(safety, ensure_ascii=False, indent=2))

tokens = sum(t["jev"]["input_tokens"] for _, t in tickets)
print(f"\n◆ {len(tickets)} calls · {tokens:,} input tokens · ${tokens * 0.042 / 1e6:.6f} total")
print("═ execute: False on every ticket — nothing was sent to a guest, no work order was filed.")
print("  A safety ticket additionally needs a named person's approval; the hotel's fire and")
print("  emergency procedures run independently of this classifier and are never suppressed by it.")
