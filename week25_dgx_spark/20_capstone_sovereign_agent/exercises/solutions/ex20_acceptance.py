#!/usr/bin/env python3
"""Exercise 20 · reference solution — the capstone's acceptance judge and a scenario that catches a real bug.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/20_capstone_sovereign_agent/exercises/ex20_acceptance.py

The checker is free and offline: it feeds your judge six recorded agent runs (tool calls + tickets,
no LLM needed) and checks every verdict. Stuck? Compare with exercises/solutions/.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from sparkkit import banner, check  # noqa: E402

TICKET_DEPTS = {"engineering", "security"}          # the agent may open tickets only for these


# ── TODO 1 ── judge one agent run. Return a list of (check_name, passed) tuples:
#   "routed first"      the FIRST tool call is route_guest_message
#   "right department"  route_guest_message's output JSON has department == scenario["department"]
#   "ticket rule"       a ticket exists  ⇔  scenario["department"] is in TICKET_DEPTS
#   "priority"          ONLY when a ticket is expected: the last ticket's priority == scenario["priority"]
#                       (a missing ticket fails this check too)
#   calls = [(tool_name, input_str, output_str), …] in order; tickets = [{"room", "priority", …}, …]
def judge(scenario: dict, calls: list[tuple[str, str, str]], tickets: list[dict]) -> list[tuple[str, bool]]:
    names = [c[0] for c in calls]
    out = [("routed first", bool(names) and names[0] == "route_guest_message")]
    routed = next((c for c in calls if c[0] == "route_guest_message"), None)
    try:
        dept = json.loads(routed[2]).get("department") if routed else None
    except (json.JSONDecodeError, AttributeError):
        dept = None
    out.append(("right department", dept == scenario["department"]))
    expect_ticket = scenario["department"] in TICKET_DEPTS
    out.append(("ticket rule", bool(tickets) == expect_ticket))
    if expect_ticket:
        out.append(("priority", bool(tickets) and tickets[-1].get("priority") == scenario["priority"]))
    return out


# ── TODO 2 ── lab 20-3 found that a prompt-only router marks normal Thai requests "urgent".
#   Add a scenario that would catch that: a Thai-language guest message about a NON-urgent engineering
#   problem, expecting department "engineering" and priority "normal". Use Thai script in "message".
THAI_SCENARIO = {"id": "thai-normal", "message": "สวัสดีค่ะ ไฟในห้องน้ำห้อง 402 กะพริบ",   # the bathroom light flickers
                 "department": "engineering", "priority": "normal"}


# ── TODO 3 ── the verdict for a whole acceptance run. results = {scenario_id: [(check, passed), …]}.
#   Return (accepted: bool, failures: list of "scenario_id: check_name" for every failed check).
def verdict(results: dict[str, list[tuple[str, bool]]]) -> tuple[bool, list[str]]:
    fails = [f"{sid}: {name}" for sid, checks in results.items() for name, passed in checks if not passed]
    return not fails, fails


# ─────────────────────────── checker — no need to edit below ────────────────
def R(d, p, reply="ok"):
    return json.dumps({"ok": True, "department": d, "priority": p, "reply": reply})


RUNS = [   # (scenario, calls, tickets, expected verdicts in TODO-1 order)
    ({"department": "security", "priority": "urgent"},
     [("route_guest_message", "smoke 522", R("security", "urgent")), ("create_maintenance_ticket", "522", "MT-1")],
     [{"room": "522", "priority": "urgent"}], [True, True, True, True]),
    ({"department": "concierge", "priority": "normal"},
     [("route_guest_message", "restaurant", R("concierge", "normal"))], [], [True, True, True]),
    ({"department": "concierge", "priority": "normal"},                  # opened a ticket it must not open
     [("route_guest_message", "restaurant", R("concierge", "normal")), ("create_maintenance_ticket", "?", "MT-2")],
     [{"room": "?", "priority": "normal"}], [True, True, False]),
    ({"department": "engineering", "priority": "normal"},                # the Thai false-urgency bug
     [("route_guest_message", "ทีวี 1203", R("engineering", "urgent")), ("create_maintenance_ticket", "1203", "MT-3")],
     [{"room": "1203", "priority": "urgent"}], [True, True, True, False]),
    ({"department": "engineering", "priority": "normal"},                # acted before routing
     [("room_temperature", "808", "27.9"), ("route_guest_message", "hot", R("engineering", "normal"))],
     [], [False, True, False, False]),
    ({"department": "security", "priority": "urgent"},                   # router misrouted, no ticket
     [("route_guest_message", "smoke", R("housekeeping", "normal"))], [], [True, False, False, False]),
]


def main() -> None:
    banner("Exercise 20 · the acceptance judge", "offline checker · recorded runs · no LLM, no Spark", status=False)
    ok = True
    got_all = []
    for sc, calls, tickets, want in RUNS:
        got = judge(sc, calls, tickets)
        got_all.append(got)
        vals = [p for _, p in got]
        expect = [w for w in want if w is not None]
        ok_run = vals[:len(expect)] == expect and len(vals) == len(expect)
        ok &= ok_run
    check(ok, "judge: 6 recorded runs · smoke ✓ · restaurant ✓ · stray ticket ✕ · Thai urgency ✕ · acted first ✕ · misroute ✕",
          f"TODO 1: verdicts {[[p for _, p in g] for g in got_all]} — expected "
          f"{[[w for w in r[3] if w is not None] for r in RUNS]}")
    thai = bool(re.search("[฀-๿]", THAI_SCENARIO.get("message", "")))
    ok2 = thai and THAI_SCENARIO.get("department") == "engineering" and THAI_SCENARIO.get("priority") == "normal"
    ok &= check(ok2, f"Thai scenario: \"{THAI_SCENARIO.get('message', '')}\" → engineering · normal",
                "TODO 2: THAI_SCENARIO needs a Thai-script message, department 'engineering', priority 'normal'")
    acc, fails = verdict({"smoke": got_all[0], "restaurant": got_all[1], "thai": got_all[3]}) if got_all[0] else (None, [])
    ok3 = acc is False and fails == ["thai: priority"]
    ok &= check(ok3, "verdict: one failed check anywhere blocks acceptance, and names it (thai: priority)",
                f"TODO 3: expected (False, ['thai: priority']), got ({acc}, {fails})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Add THAI_SCENARIO to _capstone.SCENARIOS and re-run lab 20-3: the laptop stand-in may well fail it;")
    print("  the fine-tuned router (trained on Thai examples) is what has to pass it.")


if __name__ == "__main__":
    main()
