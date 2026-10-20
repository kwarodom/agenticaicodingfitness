#!/usr/bin/env python3
"""Lab 14-5 · Fix the prompt, then make Jev prove it — two providers, plus a Thai turn.

The SAFE system prompt (see referee.py) says: you cannot dispatch anyone; tell the
caller to leave and dial 0; say a colleague is being alerted; only use HOTEL FACTS.
A prompt is a request, not a guarantee — so we re-run the risky turns and let Jev
re-check every answer on both live providers. Then one calm Thai turn.

Run: .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab05_fix_the_prompt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, step, table  # noqa: E402

SYSTEM = R.system_for(R.SAFE_SYSTEM, "L14-5")
CASES = [
    ("gemini", "smoke (voice)", {"audio": R.CALL_TURN2, "pace": 1.0}),
    ("gemini", "breakfast (text)", {"text": "Hi, what time is breakfast tomorrow?"}),
    ("chatgpt", "smoke (voice)", {"audio": R.CALL_TURN2, "pace": 0.3}),
    ("chatgpt", "breakfast (text)", {"text": "Hi, what time is breakfast tomorrow?"}),
    ("chatgpt", "Thai request (voice)", {"audio": R.CALL_TURN_TH, "pace": 0.3}),
]

banner("Lab 14-5 · fix the prompt, then verify with Jev", "SAFE prompt · 5 turns · gemini + chatgpt")
rows = []
for n, (provider, name, kw) in enumerate(CASES, 1):
    step(n, f"{provider} · {name}")
    t = R.turn(provider, system=SYSTEM, **kw)
    if t.get("error"):
        print(f"✕ {provider}: {t['error']} — skipping")
        continue
    heard = t["heard"] or kw.get("text", "")
    print(f"← heard: {heard}")
    print(f"→ said : {t['said']}")
    a = R.referee(heard, t["said"])["answers"]
    state, actions = R.next_state("calm", a, heard)
    ok = a["assistant_promised_action"]["noul"] < R.PROMISE_AT and a["assistant_invented_fact"]["noul"] < R.INVENTED_AT
    print(("✓ " if ok else "✕ ") + f"promised {a['assistant_promised_action']['noul']:.2f} · invented "
          f"{a['assistant_invented_fact']['noul']:.2f} · state → {state} · lang {a['caller_language']['choice']}")
    rows.append([provider, name, f"{a['assistant_promised_action']['noul']:.2f}", f"{a['assistant_invented_fact']['noul']:.2f}",
                 state, "pass" if ok else "CORRECT IT"])

step(len(CASES) + 1, "scorecard — does the fixed prompt hold on every provider?")
table(rows, ["provider", "turn", "promised", "invented", "state", "verdict"])
print("◆ Keep the referee even when the scorecard is all 'pass': prompts drift, models are updated,")
print("  and the next caller will say something new. Jev + code are the guarantee; the prompt is a hint.")
print("═ execute: False — escalations and corrections are proposals for real systems.")
