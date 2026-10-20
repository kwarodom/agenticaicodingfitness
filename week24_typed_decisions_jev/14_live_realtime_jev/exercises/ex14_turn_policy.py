#!/usr/bin/env python3
"""Exercise 14 · Write the per-call turn policy — the code that decides, not the live model.

Fill in `next_state()`. It receives the call's current state, Jev's five answers for
one turn, and the caller's transcript, and returns (new_state, proposed_actions).

Rules to implement (thresholds are in referee.py):
  1. EMERGENCY — caller_emergency ≥ EMERGENCY_AT (0.30) OR the DANGER regex matches the
     transcript → state "escalated" and propose BOTH "page on-call engineer + security"
     and "say: " + SAFETY_LINE. Escalated is sticky: never leave it, never re-page.
  2. HANDOFF — only from "calm": needs_human ≥ HUMAN_AT (0.60) → state "handoff" and
     propose "queue a maintenance visit + offer a human agent".
  3. CORRECTIONS (any state), at most ONE per turn — assistant_promised_action ≥ PROMISE_AT
     → propose "say: " + CORRECTION_PROMISE; otherwise assistant_invented_fact ≥ INVENTED_AT
     → propose "say: " + CORRECTION_FACT.

Run:  .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/exercises/ex14_turn_policy.py
The offline tests run first and cost nothing. Only when all pass does one live turn run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import referee as R  # noqa: E402
from jevkit import banner, check, step  # noqa: E402


def next_state(state: str, answers: dict, heard: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    # TODO 1 — emergency (Jev probability OR the R.DANGER regex) → "escalated", sticky, page once
    # TODO 2 — handoff: only from "calm", needs_human ≥ R.HUMAN_AT
    # TODO 3 — corrections: promised action / invented fact → the matching R.CORRECTION_* line
    raise NotImplementedError("TODO: implement next_state")


# ─────────────────────────── checker — no need to edit below ────────────────
def fx(emergency=0.0, human=0.0, promised=0.0, invented=0.0, lang="en"):
    """A fixed, fake Jev answer set — the offline tests never call a model."""
    n = lambda v: {"type": "noul", "noul": v}
    return {"caller_emergency": n(emergency), "needs_human": n(human), "assistant_promised_action": n(promised),
            "assistant_invented_fact": n(invented),
            "caller_language": {"type": "choice", "choice": lang, "confidence": 1.0, "probabilities": {lang: 1.0}}}


TESTS = [
    ("calm, nothing to do", ("calm", fx(), "What time is checkout?"), "calm", set()),
    ("emergency by Jev", ("calm", fx(emergency=0.9), "There is a strange noise."), "escalated", {"page", "safety"}),
    ("emergency by keyword only", ("calm", fx(emergency=0.05), "No, I said smoke, not smell!"), "escalated", {"page", "safety"}),
    ("escalated is sticky, no re-page", ("escalated", fx(emergency=0.95), "The smoke is getting worse"), "escalated", set()),
    ("handoff from calm", ("calm", fx(human=0.8), "Can someone come fix the TV?"), "handoff", {"visit"}),
    ("no handoff once escalated", ("escalated", fx(human=0.9), "Please send someone"), "escalated", set()),
    ("correct a promise", ("calm", fx(promised=0.8), "The AC is clicking."), "calm", {"promise"}),
    ("correct an invented fact", ("handoff", fx(invented=0.7), "When is breakfast?"), "handoff", {"fact"}),
    ("only one correction per turn", ("calm", fx(promised=0.9, invented=0.8), "The AC is clicking."), "calm", {"promise"}),
]
KIND = {"page": "page on-call", "safety": R.SAFETY_LINE, "visit": "queue a maintenance visit",
        "promise": R.CORRECTION_PROMISE, "fact": R.CORRECTION_FACT}


def kinds(actions):
    return {k for k, needle in KIND.items() if any(needle in a for a in actions)}


def main() -> None:
    banner("Exercise 14 · the per-call turn policy")
    step(1, "offline tests — fixed Jev answers, no API call")
    ok = True
    for name, args, want_state, want_kinds in TESTS:
        try:
            got_state, actions = next_state(*args)
        except NotImplementedError as e:
            print(f"✕ {e}")
            ok = False
            break
        good = got_state == want_state and kinds(actions) == want_kinds
        ok &= check(good, f"{name}: → {got_state} {sorted(kinds(actions))}",
                    f"{name}: expected {want_state} {sorted(want_kinds)}, got {got_state} {sorted(kinds(actions))}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        sys.exit(1)

    step(2, "one live turn — your policy on a real live model + a real Jev verdict")
    text = "Help, there is smoke coming out of the air conditioner vent in room 1507!"
    t = R.turn("gemini", text=text, system=R.system_for(R.NAIVE_SYSTEM, "EX14"))
    if t.get("error"):
        print(f"✕ gemini: {t['error']}")
        sys.exit(1)
    print(f"← caller: {text}")
    print(f"→ said  : {t['said']}")
    a = R.referee(text, t["said"])["answers"]
    state, actions = next_state("calm", a, text)
    print(f"» Jev emergency {a['caller_emergency']['noul']:.2f} · promised {a['assistant_promised_action']['noul']:.2f}")
    print(f"→ your policy: calm → {state}")
    for act in actions:
        print(f"→ proposed: {act}")
    print("═ execute: False — your policy proposed; nothing was paged.")


if __name__ == "__main__":
    main()
