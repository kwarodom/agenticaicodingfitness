#!/usr/bin/env python3
"""Lab 14-4 · An escalating call — a per-call state machine, and Jev's latency budget.

One phone call, two caller turns:
    turn 1  "…the AC is making a clicking noise… not urgent."          → calm
    turn 2  "Oh wait — now there's smoke coming out of the AC vent!"    → escalated (sticky)

After each turn Jev referees (heard + said) and code moves the call through a
state machine. We also measure the budget: in production Jev starts the moment the
caller's transcript arrives, so its verdict must land before — or just after — the
live model starts talking. We compose the two measured timelines to show that.

Run: .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab04_escalating_call.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, step, table  # noqa: E402
import livekit  # noqa: E402

SYSTEM = R.system_for(R.NAIVE_SYSTEM, "L14-4")
banner("Lab 14-4 · an escalating call", "two caller turns · one state machine · a latency budget")

state, rows, budget = "calm", [], []
for n, audio in enumerate([R.CALL_TURN1, R.CALL_TURN2], 1):
    step(n, f"caller turn {n} · {audio.name}")
    t = R.turn("gemini", audio=audio, system=SYSTEM)
    if t.get("error"):
        print(f"✕ gemini: {t['error']} — skipping this turn")
        continue
    print(f"← heard: {t['heard']}")
    print(f"→ said : {t['said']}")
    ref = R.referee(t["heard"], t["said"])
    a = ref["answers"]
    before = state
    state, actions = R.next_state(state, a, t["heard"])
    print(f"» Jev   emergency {a['caller_emergency']['noul']:.2f} · needs_human {a['needs_human']['noul']:.2f} · "
          f"promised {a['assistant_promised_action']['noul']:.2f} · invented {a['assistant_invented_fact']['noul']:.2f} · "
          f"lang {a['caller_language']['choice']}")
    print(f"→ state {before} → {state}")
    for act in actions:
        print(f"→ proposed: {act}")
    rows.append([n, before, state, f"{a['caller_emergency']['noul']:.2f}", f"{a['assistant_promised_action']['noul']:.2f}",
                 "; ".join(x.split(" (")[0] for x in actions)[:52] or "—"])
    heard_at = next((ts for ts, kind, _ in t["events"] if kind == "heard"), None)
    budget.append((n, t, heard_at, ref["jev_ms"]))

step(3, "the call, turn by turn")
table(rows, ["turn", "state before", "state after", "emergency", "promised", "proposed actions"])

step(4, "the latency budget — would Jev's verdict be ready in time?")
brows = []
for n, t, heard_at, jev_ms in budget:
    first = (t.get("timing") or {}).get("first_output")
    if jev_ms is None or heard_at is None or first is None:
        brows.append([n, f"{heard_at}s" if heard_at is not None else "—", "n/a (DRY)" if jev_ms is None else f"{jev_ms:.0f} ms",
                      f"{first:.2f}s" if first is not None else "—", "measure it LIVE"])
        continue
    ready = heard_at + jev_ms / 1000
    brows.append([n, f"{heard_at:.2f}s", f"{jev_ms:.0f} ms", f"{first:.2f}s",
                  f"{ready:.2f}s ({'before' if ready <= first else 'after'} first word, {ready - first:+.2f}s)"])
table(brows, ["turn", "caller transcript at", "Jev took", "live model spoke at", "Jev verdict ready at"])
print("◆ Jev runs BESIDE the conversation (a thread or task), so it never adds silence to the call.")
print("  Even when its verdict lands a moment after the model starts speaking, code can still")
print("  interrupt (barge-in) with the safety line and page the real on-call — within the same turn.")
print("═ execute: False — the pager and the spoken lines are proposals; this lab changed nothing.")
