#!/usr/bin/env python3
"""Exercise 04 · Add a new route: `iot_connectivity`.

Alto Copilot gets many "gateway offline / sensor not reporting" questions.
Today they land in `hvac` or `facility_ops`. You will add a new topic.

Part A (offline, $0): write the confidence gate `accepted(answer)` —
          tested against fixed answer dicts, no API call.
Part B (live): write the description for the new option and three test
          messages of your own. The checker also runs hidden gateway messages
          (must route to iot_connectivity) and control messages (must NOT).

Run: .venv/bin/python week24/04_intent_routing/exercises/ex04_add_a_route.py
"""
import sys
from pathlib import Path

WEEK24 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK24 / "common"))
sys.path.insert(0, str(WEEK24 / "jev_lab"))
import jev_lab as ref  # noqa: E402
from jevkit import ask, banner, check, step, table, top2, validate  # noqa: E402


# ── PART A ── TODO 1: return True only if ALL hold for a choice/score answer `a`:
#   a["confidence"] >= 0.75   ·   top probability >= 0.80   ·   (top − second) >= 0.20
#   Tip: top2(a["probabilities"]) returns (top, second).
def accepted(a: dict) -> bool:
    return False   # ← replace this line


# ── PART B ── TODO 2: describe the new option like a rubric line:
#   what belongs here, and what does NOT (so it does not steal HVAC problems).
NEW_OPTION = "iot_connectivity"
NEW_OPTION_DESCRIPTION = None

# ── TODO 3: three messages of your own that SHOULD route to iot_connectivity.
MY_TEST_MESSAGES = []


# ─────────────────────────── checker — no need to edit below ────────────────
GATE_CASES = [   # (answer, expected) — fixed fixtures, NOT model output
    ({"confidence": 0.95, "probabilities": {"a": 0.96, "b": 0.03, "c": 0.01}}, True),
    ({"confidence": 0.70, "probabilities": {"a": 0.80, "b": 0.15, "c": 0.05}}, False),   # confidence < .75
    ({"confidence": 0.80, "probabilities": {"a": 0.79, "b": 0.20, "c": 0.01}}, False),   # peak < .80
    ({"confidence": 0.60, "probabilities": {"a": 0.80, "b": 0.20}}, False),              # confidence < .75
    ({"confidence": 0.75, "probabilities": {"a": 0.80, "b": 0.20}}, True),               # boundaries are inclusive
]
HIDDEN_IOT = ["The LoRaWAN gateway at Siam Tower has been offline since 3am — no sensor data is arriving.",
              "Room 1204's temperature sensor stopped reporting yesterday; the dashboard shows it as stale."]
CONTROLS = [("Why is AHU-3 not cooling the hotel lobby?", "hvac"),
            ("Write a Python function to validate a JSON payload.", "coding")]


def build_questions() -> dict:
    base = ref.QUESTIONS["intent"]["intent"]
    criteria = dict(base["criteria"])
    criteria[NEW_OPTION] = NEW_OPTION_DESCRIPTION
    return {"intent": {"type": "choice", "instructions": base["instructions"], "criteria": criteria}}


def main() -> None:
    banner("Exercise 04 · add a route")
    step("A", "your accepted() gate against 5 fixed fixtures (offline, $0)")
    gate_ok = True
    for i, (ans, want) in enumerate(GATE_CASES, 1):
        got = bool(accepted(ans))
        gate_ok &= check(got == want, f"case {i}: accepted → {got}", f"case {i}: got {got}, expected {want}")

    print("◆ did you notice? if probabilities sum to 1 and peak ≥ .80, the second is ≤ .20,")
    print("  so the margin is ≥ .60 — the margin rule only bites when you lower the peak rule.")

    step("B", "shape checks (offline, $0)")
    shape_ok = check(isinstance(NEW_OPTION_DESCRIPTION, str) and len(NEW_OPTION_DESCRIPTION) >= 25,
                     "iot_connectivity has a real description",
                     "TODO 2: write NEW_OPTION_DESCRIPTION (a full sentence: what belongs, what does not)")
    shape_ok &= check(len(MY_TEST_MESSAGES) >= 3 and all(isinstance(m, str) and m.strip() for m in MY_TEST_MESSAGES),
                      f"{len(MY_TEST_MESSAGES)} test messages", "TODO 3: add 3 messages to MY_TEST_MESSAGES")
    if not (gate_ok and shape_ok):
        print("\n⚠ fix the ✕ lines above and run again — no API call was made.")
        sys.exit(1)

    questions = build_questions()
    step("C", f"route {len(MY_TEST_MESSAGES) + len(HIDDEN_IOT) + len(CONTROLS)} messages (one call each)")
    rows, passed = [], True
    cases = ([(m, NEW_OPTION, "yours") for m in MY_TEST_MESSAGES]
             + [(m, NEW_OPTION, "hidden") for m in HIDDEN_IOT]
             + [(m, want, "control") for m, want in CONTROLS])
    for msg, want, kind in cases:
        a = validate(ask({"message": msg}, questions, quiet=True), questions)["intent"]
        good = a["choice"] == want
        passed &= good or kind == "yours"
        rows.append([kind, msg, want, a["choice"], f"{a['confidence']:.2f}",
                     "yes" if accepted(a) else "no", "✓" if good else "✕"])
    table(rows, ["kind", "message", "want", "jev", "conf", "accepted?", ""])
    check(passed, "hidden gateway messages route to iot_connectivity, controls stay put",
          "a hidden or control case failed — sharpen your description (say what is NOT connectivity)")
    print("═ execute: False — in production, a new route also needs its own prompt, tools,")
    print("  owner and a shadow-mode evaluation before it receives real traffic.")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
