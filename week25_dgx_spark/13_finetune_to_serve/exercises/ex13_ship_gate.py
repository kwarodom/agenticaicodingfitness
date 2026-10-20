#!/usr/bin/env python3
"""Exercise 13 · Write the ship gate that decides whether a fine-tune goes live.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/13_finetune_to_serve/exercises/ex13_ship_gate.py

The checker is free and offline. It feeds your functions real-looking model outputs (clean JSON,
fenced JSON, chatter, broken JSON, a reply that invents a time) and then two whole evaluation runs,
and checks every verdict. Stuck? Compare with exercises/solutions/.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

DEPARTMENTS = {"housekeeping", "engineering", "front_desk", "food_beverage", "concierge", "security"}


# ── TODO 1 ── turn a model's output into a dict, or None.
#   Models wrap JSON in ```json fences, or add a sentence before it. Find the first {...} that parses.
#   Return None if nothing parses. Hint: re.findall(r"\{.*?\}", text, flags=re.S) + json.loads in try/except.
def parse_answer(text: str) -> dict | None:
    return None


# ── TODO 2 ── does the reply promise a time the guest did not ask for?
#   Flag "in 10 minutes", "within the hour", "by 6pm". Echoing the guest's own time is fine:
#   guest "table for 2 at 19:00" → reply "…at 19:00…" is NOT a promise.
#   Return True for a promise, False otherwise.
def promises_time(reply: str, guest_message: str) -> bool:
    return None


# ── TODO 3 ── the ship decision. metrics has json_valid, department_acc, urgent_recall, reply_policy (0..1).
#   Ship only if: json_valid ≥ 0.98, department_acc ≥ 0.90, urgent_recall == 1.0, reply_policy ≥ 0.95,
#   AND department_acc is not lower than the baseline's. Return (ship: bool, reasons: list[str]) where
#   reasons names every rule that failed (empty list when shipping).
def ship(metrics: dict, baseline: dict) -> tuple[bool, list[str]]:
    return None, []


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 13 · the ship gate", "offline checker · free · no Spark needed", status=False)
    ok = True
    cases = [
        ('{"department": "security", "priority": "urgent", "reply": "Security is on the way."}', "security"),
        ('```json\n{"department": "housekeeping", "priority": "normal", "reply": "Of course."}\n```', "housekeeping"),
        ('Sure! Here is the routing:\n{"department": "concierge", "priority": "normal", "reply": "Happily."}', "concierge"),
        ('{"department": "security", "priority": urgent, "reply": "On our way."}', None),     # unquoted value
        ("I think engineering should handle this.", None),
    ]
    got = [(parse_answer(t) or {}).get("department") if parse_answer(t) else None for t, _ in cases]
    ok &= check(got == [c[1] for c in cases],
                "parse_answer: clean · fenced · with chatter → dict; unquoted value and prose → None",
                f"TODO 1: expected {[c[1] for c in cases]}, got {got}")
    tc = [("Housekeeping will bring towels in 10 minutes.", "Extra towels please", True),
          ("Our engineer will come within the hour.", "The AC is broken", True),
          ("Your table for 2 at 19:00 is noted; we will confirm.", "Table for 2 at 19:00 please", False),
          ("Housekeeping is on the way.", "Extra towels please", False)]
    res = [promises_time(r, g) for r, g, _ in tc]
    ok &= check(res == [e for *_, e in tc],
                "promises_time: 'in 10 minutes' ✓ · 'within the hour' ✓ · echoed 19:00 ✗ · no time ✗",
                f"TODO 2: expected {[e for *_, e in tc]}, got {res}")
    base = {"json_valid": 0.97, "department_acc": 0.97, "urgent_recall": 0.8, "reply_policy": 0.97}
    good = {"json_valid": 1.0, "department_acc": 0.98, "urgent_recall": 1.0, "reply_policy": 1.0}
    bad = {"json_valid": 1.0, "department_acc": 0.95, "urgent_recall": 0.9, "reply_policy": 0.9}
    s1, r1 = ship(good, base)
    s2, r2 = ship(bad, base)
    ok &= check(s1 is True and r1 == [] and s2 is False and len(r2) == 3,
                "ship: the good run ships · the bad run is blocked by urgent_recall, reply_policy and the baseline",
                f"TODO 3: good → ({s1}, {r1}); bad → ({s2}, {r2}) — expected (True, []) and (False, 3 reasons)")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n▣ why the bad run is blocked")
    for r in r2:
        print(f"│ {r}")
    print("\n═ A 95% department score still fails: one missed emergency is worse than ten misrouted towel requests.")


if __name__ == "__main__":
    main()
