#!/usr/bin/env python3
"""Exercise 12 · CAPSTONE — build the smart-hotel guest-request copilot yourself.

Four TODOs, each checked OFFLINE (free) before a single API call:

  TODO 1  QUESTIONS        the five typed questions (one batched Jev call)
  TODO 2  extract_room()   a regex — a room number is copied, never guessed
  TODO 3  policy()         safety override → uncertainty gate → priority
  TODO 4  build_ticket()   the JSON proposal (execute False!)

When every check is ✓, the copilot runs live on five guest messages.
Reference answer: ../copilot_reference.py (try first — peek later).

Run: .venv/bin/python week24/12_capstone_hotel_copilot/exercises/ex12_build_the_copilot.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, check, choice, guarded, noul, score, top2, validate  # noqa: E402,F401

# ── TODO 1 ── the questions. Keys MUST be exactly these five:
#   department    choice  hvac, housekeeping, maintenance, front_desk, food_beverage + "unknown"
#   severity      score   ≥ 3 ordered levels, from "no problem" up to "possible danger"
#   safety        noul    explicitly mentions smoke/fire/burning/sparks/gas/flooding/shock/injury?
#   needs_human   noul    asks for a manager/person, strong anger, refund or compensation?
#   mentions_room noul    does the message state the guest's room number?
#   Tip: wrap the dict in guarded({...}) so every instruction carries the injection guard.
QUESTIONS = {}


# ── TODO 2 ── return the room number as a string, or None.
#   Must handle: "room 1203", "Room 402.", "ห้อง 815", and a lone "— 610."
#   Must NOT return: "40 minutes", "29 degrees", "1500 baht", "11:30".
def extract_room(message: str):
    return None


# ── TODO 3 ── the policy. `answers` is Jev's validated answers dict.
#   Return a dict with: route, priority, escalate_to_human, requires_human_approval, reason
#   Rules (in this order):
#     a) safety noul ≥ 0.20        → route "duty_manager", priority "P0",
#                                     escalate_to_human True, requires_human_approval True
#     b) department is "unknown", OR top probability < 0.70, OR top − second < 0.20
#                                   → route "front_desk_review"
#        otherwise                  → route = the chosen department
#     c) priority from severity score: ≥ 1.5 → "P1", ≥ 0.5 → "P2", else "P3"
#     d) escalate_to_human = needs_human noul ≥ 0.5 ; requires_human_approval False
#   (top2(probabilities) from jevkit gives (top, second).)
def policy(answers: dict) -> dict:
    return {}


# ── TODO 4 ── the ticket. Must contain at least:
#   room, route, priority, escalate_to_human, requires_human_approval,
#   follow_ups (list; add "ask_room_number" when room is None), model, execute (False!)
def build_ticket(message: str, answers: dict, decision: dict, room, resp: dict) -> dict:
    return {}


# ─────────────────────────── checker — no need to edit below ────────────────
def _choice(probs):
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "probabilities": probs, "confidence": 0.9}


def _answers(dept, sev, safety, human, room=0.9):
    return {"department": _choice(dept), "severity": {"type": "score", "score": sev},
            "safety": {"type": "noul", "noul": safety}, "needs_human": {"type": "noul", "noul": human},
            "mentions_room": {"type": "noul", "noul": room}}


POLICY_FIXTURES = [
    ("clear HVAC fault", _answers({"hvac": .95, "maintenance": .05}, 2.0, .01, .1), "hvac", "P1", False),
    ("burning smell overrides everything", _answers({"hvac": .9, "maintenance": .1}, 2.9, .9, .1), "duty_manager", "P0", True),
    ("low safety still triggers (0.25)", _answers({"housekeeping": .99, "hvac": .01}, 0.2, .25, .0), "duty_manager", "P0", True),
    ("torn between two teams", _answers({"hvac": .55, "maintenance": .45}, 1.0, .02, .1), "front_desk_review", "P2", False),
    ("confident but unknown", _answers({"unknown": .98, "front_desk": .02}, 0.0, .01, .0), "front_desk_review", "P3", False),
    ("angry guest escalates", _answers({"food_beverage": .95, "front_desk": .05}, 1.9, .01, .97), "food_beverage", "P1", False),
]
ROOM_FIXTURES = [
    ("The AC in room 1203 is warm", "1203"), ("Room 402.", "402"), ("แอร์ห้อง 815 ไม่เย็น", "815"),
    ("No need to clean today — 610.", "610"), ("waited 40 minutes", None), ("it is 29 degrees", None),
    ("we paid 1500 baht", None), ("checkout at 11:30", None), ("hello?", None),
]
LIVE_MESSAGES = [
    "The AC in room 1203 is blowing warm air.",
    "There's smoke coming out of the bathroom fan in 1507!",
    "ขอผ้าเช็ดตัวเพิ่มสองผืนค่ะ ห้อง 1402",
    "I want to speak to a manager about my bill. This is unacceptable.",
    "hello?",
]


def offline_checks() -> bool:
    ok = True
    print("▣ TODO 1 · questions")
    want = {"department": "choice", "severity": "score", "safety": "noul", "needs_human": "noul", "mentions_room": "noul"}
    ok &= check(set(QUESTIONS) == set(want), "exactly the five question keys", f"TODO 1: keys must be {sorted(want)}")
    for k, t in want.items():
        q = QUESTIONS.get(k) or {}
        ok &= check(q.get("type") == t and bool(q.get("instructions")), f"{k} is a {t}", f"TODO 1: {k} must be a {t} with instructions")
    crit = (QUESTIONS.get("department") or {}).get("criteria") or {}
    ok &= check({"hvac", "housekeeping", "maintenance", "front_desk", "food_beverage", "unknown"} <= set(crit),
                "department has the five teams + unknown", "TODO 1: department options are incomplete")
    ok &= check(len((QUESTIONS.get("severity") or {}).get("criteria") or []) >= 3, "severity has ≥ 3 levels",
                "TODO 1: severity needs at least 3 ordered levels")

    print("\n▣ TODO 2 · extract_room()")
    for text, want_room in ROOM_FIXTURES:
        got = extract_room(text)
        ok &= check(got == want_room, f"{text!r} → {want_room!r}", f"TODO 2: {text!r} should give {want_room!r}, got {got!r}")

    print("\n▣ TODO 3 · policy()")
    for name, answers, route, prio, approval in POLICY_FIXTURES:
        d = policy(answers) or {}
        good = d.get("route") == route and d.get("priority") == prio and d.get("requires_human_approval") is approval
        ok &= check(good, f"{name} → {route} {prio}",
                    f"TODO 3: {name}: want {route}/{prio}/approval={approval}, got "
                    f"{d.get('route')}/{d.get('priority')}/approval={d.get('requires_human_approval')}")
    angry = policy(POLICY_FIXTURES[5][1]) or {}
    ok &= check(angry.get("escalate_to_human") is True, "angry guest → escalate_to_human True",
                "TODO 3: needs_human ≥ 0.5 must set escalate_to_human True")

    print("\n▣ TODO 4 · build_ticket()")
    fake_resp = {"model": "jev-1.13.0", "_request_id": "0123456789abcdef", "usage": {"input_tokens": 800}}
    ans = _answers({"hvac": .9, "maintenance": .1}, 2.9, .9, .1, room=.05)   # "smoke!" — no room stated
    d = policy(ans) or {"route": "duty_manager", "priority": "P0", "escalate_to_human": True,
                        "requires_human_approval": True, "reason": ""}
    t = build_ticket("smoke!", ans, d, None, fake_resp) or {}
    need = {"room", "route", "priority", "escalate_to_human", "requires_human_approval", "follow_ups", "model", "execute"}
    ok &= check(need <= set(t), "ticket has all required fields", f"TODO 4: ticket is missing {sorted(need - set(t))}")
    ok &= check(t.get("execute") is False, "execute is False", "TODO 4: execute must be exactly False — a ticket is a proposal")
    ok &= check("ask_room_number" in (t.get("follow_ups") or []), "no room → ask_room_number follow-up",
                "TODO 4: when room is None, follow_ups must include 'ask_room_number'")
    return ok


def main() -> None:
    print("━" * 72)
    print("━━ Exercise 12 · CAPSTONE — build the guest-request copilot")
    print("━" * 72)
    if not offline_checks():
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        sys.exit(1)
    print(f"\n▣ all offline checks passed — running your copilot LIVE on {len(LIVE_MESSAGES)} messages")
    for msg in LIVE_MESSAGES:
        resp = ask({"message": msg}, QUESTIONS, quiet=True)
        answers = validate(resp, QUESTIONS)
        ticket = build_ticket(msg, answers, policy(answers), extract_room(msg), resp)
        print(f"\n━━ “{msg}”")
        print(json.dumps({k: ticket[k] for k in ("room", "route", "priority", "escalate_to_human",
                                                  "requires_human_approval", "follow_ups", "execute")},
                         ensure_ascii=False))
    print("\n═ You built it: code for exact facts, Jev for judgments, code for policy, a human for actions.")


if __name__ == "__main__":
    main()
