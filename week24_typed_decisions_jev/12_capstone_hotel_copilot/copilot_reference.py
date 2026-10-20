#!/usr/bin/env python3
"""copilot_reference — the finished smart-hotel guest-request copilot.

This is the reference answer the capstone labs run (and the exercise solution
reuses). It is ~150 lines because the design keeps each job where it belongs:

    guest message
      ├─ code : detect_language()  exact — Unicode ranges, free
      ├─ code : extract_room()     exact — regex, never guessed by a model
      ├─ Jev  : ONE batched call   department · severity · safety · needs_human · mentions_room
      ├─ code : policy()           safety override → uncertainty gate → priority
      └─ code : build_ticket()     JSON ticket + a templated reply DRAFT · execute False

Nothing is sent to a guest and no work order is filed: every ticket is a proposal.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from jevkit import ask, choice, guarded, noul, score, top2, validate  # noqa: E402

# ── the questions — ONE request, five independent judgments ─────────────────
DEPARTMENTS = {
    "hvac": "Room too hot or too cold, air-conditioning, ventilation, a noisy or leaking AC unit",
    "housekeeping": "Cleaning, making up the room, towels, linen, pillows, toiletries, amenities",
    "maintenance": "Plumbing, toilet, shower, hot water, electrical, lights, TV, door lock, broken furniture",
    "front_desk": "Bookings, billing, check-out, room keys, general information",
    "food_beverage": "Room service, restaurant, breakfast, minibar, food orders",
    "unknown": "No identifiable request, or none of the teams above",
}

QUESTIONS = guarded({
    "department": choice("Which hotel team should handle the request in `message`?", DEPARTMENTS),
    "severity": score("How much does the problem in `message` affect the guest's stay?", [
        "No problem: a question or a simple request",
        "Minor inconvenience",
        "A service failure that disrupts the stay",
        "Possible danger to people or property",
    ]),
    "safety": noul("Does `message` explicitly mention smoke, fire, a burning smell, sparks, gas, "
                   "water flooding the floor, electric shock or an injury?"),
    "needs_human": noul("Does `message` ask for a manager or a person, express strong anger, "
                        "or ask for a refund or compensation?"),
    "mentions_room": noul("Does `message` state the guest's room number?"),
})

# ── exact jobs stay in code ─────────────────────────────────────────────────
THAI = re.compile("[\u0E00-\u0E7F]")          # the Thai Unicode block
LATIN = re.compile(r"[A-Za-z]")
ROOM_KEYWORD = re.compile(r"(?:\broom|\brm\.?|ห้อง)\s*(?:no\.?|number|#)?\s*(\d{3,4})\b", re.IGNORECASE)
BARE_NUMBER = re.compile(r"(?<![\d.:])(\d{3,4})(?!\d)(?![.:]\d)(?!\s*(?:min|minutes|degrees|°|%|baht|฿|am|pm))",
                         re.IGNORECASE)


def detect_language(message: str) -> str:
    """Exact and free: which scripts appear? (Why ask a model what a regex knows?)"""
    th, en = bool(THAI.search(message)), bool(LATIN.search(message))
    return "mixed" if th and en else "th" if th else "en" if en else "other"


def extract_room(message: str) -> str | None:
    """A room number is a VALUE to copy, not a judgment: regex, never a model.
    Keyword form first ("room 1203", "ห้อง 815"); else a lone 3-4 digit number
    that is not a time, temperature, amount or duration."""
    m = ROOM_KEYWORD.search(message)
    if m:
        return m.group(1)
    m = BARE_NUMBER.search(message)
    return m.group(1) if m else None


# ── policy: YOUR rules, readable and testable, no model inside ──────────────
SAFETY_BAR = 0.20        # deliberately low: a false alarm costs a phone call, a miss costs far more
DEPT_TOP = 0.70          # illustrative, uncalibrated — tune on your own labeled data
DEPT_MARGIN = 0.20


def policy(answers: dict) -> dict:
    dept = answers["department"]
    top, second = top2(dept["probabilities"])
    sev = answers["severity"]["score"]
    if answers["safety"]["noul"] >= SAFETY_BAR:
        return {"route": "duty_manager", "priority": "P0", "escalate_to_human": True,
                "requires_human_approval": True, "reason": "possible safety issue — the hotel's own "
                "emergency procedure applies regardless of this classifier"}
    if dept["choice"] == "unknown":
        route, reason = "front_desk_review", f"no identifiable request (unknown p={top:.2f}) — ask the guest"
    elif top < DEPT_TOP or top - second < DEPT_MARGIN:
        route, reason = "front_desk_review", f"department unclear (top {top:.2f}, margin {top - second:.2f})"
    else:
        route, reason = dept["choice"], f"{dept['choice']} (p={top:.2f})"
    priority = "P1" if sev >= 1.5 else "P2" if sev >= 0.5 else "P3"
    return {"route": route, "priority": priority, "escalate_to_human": answers["needs_human"]["noul"] >= 0.5,
            "requires_human_approval": False, "reason": reason}


# ── the ticket + a templated reply draft ────────────────────────────────────
TEAM_NAMES = {
    "en": {"hvac": "engineering (air-conditioning)", "housekeeping": "housekeeping", "maintenance": "maintenance",
           "front_desk": "front desk", "food_beverage": "food & beverage", "front_desk_review": "front desk",
           "duty_manager": "duty manager"},
    "th": {"hvac": "ช่างแอร์", "housekeeping": "แม่บ้าน", "maintenance": "ช่างซ่อมบำรุง", "front_desk": "พนักงานต้อนรับ",
           "food_beverage": "ฝ่ายอาหารและเครื่องดื่ม", "front_desk_review": "พนักงานต้อนรับ", "duty_manager": "ผู้จัดการเวร"},
}


def reply_draft(language: str, decision: dict, room: str | None) -> str:
    """Filled by CODE from approved templates. This is where a generative LLM could
    personalise the wording later — still as a draft a person approves."""
    lang = "th" if language in ("th", "mixed") else "en"
    team = TEAM_NAMES[lang][decision["route"]]
    where = (f" for room {room}" if lang == "en" else f" ห้อง {room}") if room else ""
    if decision["route"] == "duty_manager":
        return ("Our duty manager has been alerted. If you feel unsafe, leave the room and dial 0 for reception."
                if lang == "en" else "เราแจ้งผู้จัดการเวรแล้ว หากรู้สึกไม่ปลอดภัย กรุณาออกจากห้องและกด 0 ติดต่อแผนกต้อนรับ")
    if lang == "th":
        return f"ขอบคุณค่ะ เราแจ้ง{team}{where}แล้ว" + ("" if room else " รบกวนแจ้งหมายเลขห้องด้วยค่ะ")
    return f"Thank you — our {team} team has been notified{where}." + ("" if room else " Could you tell us your room number?")


def build_ticket(message: str, answers: dict, decision: dict, room: str | None,
                 language: str, resp: dict) -> dict:
    follow_ups = []
    mentions = answers["mentions_room"]["noul"]
    if room is None:
        follow_ups.append("confirm_room_number" if mentions >= 0.5 else "ask_room_number")
    elif mentions < 0.5:
        follow_ups.append("check_room_number")          # regex found a number Jev doesn't read as a room
    return {
        "ticket_id": "T-" + resp["_request_id"][:8],
        "room": room,
        "language": language,
        **decision,
        "follow_ups": follow_ups,
        "signals": {"department": answers["department"]["choice"],
                    "department_p": round(max(answers["department"]["probabilities"].values()), 2),
                    "severity": round(answers["severity"]["score"], 2),
                    "safety": answers["safety"]["noul"], "needs_human": answers["needs_human"]["noul"],
                    "mentions_room": mentions},
        "reply_draft": reply_draft(language, decision, room),
        "jev": {"model": resp["model"], "request_id": resp["_request_id"],
                "input_tokens": resp["usage"]["input_tokens"]},
        "execute": False,                                # a proposal — a person or a governed system acts
    }


def handle(message: str) -> dict:
    """The whole pipeline for one guest message."""
    language = detect_language(message)
    room = extract_room(message)
    resp = ask({"message": message}, QUESTIONS, quiet=True)
    answers = validate(resp, QUESTIONS)
    decision = policy(answers)
    return build_ticket(message, answers, decision, room, language, resp)


# Sample guest messages — fictional, EN / TH / mixed, including hard cases.
MESSAGES = [
    "The AC in room 1203 is blowing warm air and it's 29 degrees in here.",
    "แอร์ห้อง 815 ไม่เย็นเลยค่ะ",
    "Could we get two extra towels and more coffee pods? Room 402.",
    "There's a burning smell coming from the air conditioner in 1507!",
    "No need to clean my room today, thanks — 610.",
    "I've been waiting 40 minutes for room service. I want to speak to the manager and get a refund.",
    "ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้นแล้ว",
    "Room 1110 the shower ไม่มีน้ำร้อน",
    "hello?",
]
