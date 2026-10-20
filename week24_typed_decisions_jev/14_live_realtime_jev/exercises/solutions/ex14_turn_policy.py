#!/usr/bin/env python3
"""Solution · Exercise 14 — one correct turn policy (yours may be shaped differently and still pass)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import referee as R  # noqa: E402

import ex14_turn_policy as ex  # noqa: E402


def next_state(state, answers, heard):
    n = lambda k: answers[k]["noul"]
    actions = []
    emergency = n("caller_emergency") >= R.EMERGENCY_AT or bool(R.DANGER.search(heard or ""))
    if emergency and state != "escalated":                     # rule 1: escalate once, stay escalated
        state = "escalated"
        actions += ["page on-call engineer + security", "say: " + R.SAFETY_LINE]
    elif state == "calm" and n("needs_human") >= R.HUMAN_AT:   # rule 2: only a calm call hands off
        state = "handoff"
        actions.append("queue a maintenance visit + offer a human agent")
    if n("assistant_promised_action") >= R.PROMISE_AT:         # rule 3: one correction, promise first
        actions.append("say: " + R.CORRECTION_PROMISE)
    elif n("assistant_invented_fact") >= R.INVENTED_AT:
        actions.append("say: " + R.CORRECTION_FACT)
    return state, actions


ex.next_state = next_state

if __name__ == "__main__":
    ex.main()
