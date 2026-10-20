#!/usr/bin/env python3
"""Module 11 shared pieces — Jev decides, your chosen LLM writes, Jev checks the draft.

    guest message ──► Jev: department · urgency · wants compensation   (typed, ~0.5 s)
                  ──► code: language, room number, APPROVED prompt, facts
                  ──► LLM (your pick): writes the reply draft            (prose, seconds)
                  ──► Jev: checks the DRAFT (promises? invents? language? on-topic?)
                  ──► code: review policy → staff approval or human review   execute: False

Labs 11-2, 11-3, 11-4 and the exercise import this file. Read it top to bottom.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from jevkit import ask, choice, guarded, noul, score, top2, validate  # noqa: E402
import llmkit  # noqa: E402

# ── 1 · Jev routes: three typed questions about the GUEST message ─────────────
ROUTE_QUESTIONS = guarded({
    "department": choice("Which hotel team should handle `message`?", {
        "hvac": "Air conditioning, heating, ventilation, room too hot or too cold",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "maintenance": "Plumbing, electrical, broken furniture, doors, lights, TV",
        "front_desk": "Bookings, billing, check-in/out, general questions",
        "unknown": "No clear request, or none of the above",
    }),
    "urgency": score("How urgent is the request in `message`?", [
        "Routine: no time pressure stated",
        "Should be handled today",
        "Needs attention within the hour",
        "Possible danger or a total outage right now",
    ]),
    "wants_compensation": noul("Does `message` ask for a refund, discount, free night or other compensation?"),
})

# ── 2 · code: the APPROVED system prompts — stored in code, never invented at runtime ──
HOUSE_RULES = (
    "You write short replies to hotel guests for staff to review before sending. "
    "Rules: reply in {language}; 2-4 sentences; warm and professional; "
    "use ONLY the facts provided — never invent names, times, prices or causes; "
    "never promise refunds, discounts or compensation (a manager decides that); "
    "if a room number is provided, mention it; sign off as 'Guest Services'."
)
APPROVED_PROMPTS = {
    "hvac": HOUSE_RULES + " Say an engineer has been asked to check the air conditioning.",
    "housekeeping": HOUSE_RULES + " Say housekeeping has been asked to help.",
    "maintenance": HOUSE_RULES + " Say the maintenance team has been asked to take a look.",
    "front_desk": HOUSE_RULES + " Say the front desk will follow up.",
}
LANG_NAME = {"th": "Thai", "en": "English"}

THAI = re.compile(r"[฀-๿]")
ROOM = re.compile(r"(?:room|rm\.?|ห้อง(?:พัก)?(?:หมายเลข)?)\s*#?\s*(\d{3,4})\b", re.I)

GUESTS = [
    "Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.",
    "ห้อง 815 ยังไม่มีผ้าเช็ดตัวเลยค่ะ รบกวนเอามาให้หน่อยนะคะ",
    "I was charged twice for my stay. I want a refund today or I'm leaving a review.",
]


def language(message: str) -> str:
    """An exact fact → code, not a model: does the text contain Thai script?"""
    return "th" if THAI.search(message) else "en"


def room_number(message: str):
    m = ROOM.search(message)
    return m.group(1) if m else None


def route(message: str) -> dict:
    """One Jev call + code facts. Returns everything the writer and the checker need."""
    resp = ask({"message": message}, ROUTE_QUESTIONS, quiet=True)
    a = validate(resp, ROUTE_QUESTIONS)
    dept = a["department"]
    p1, p2 = top2(dept["probabilities"])
    confident = dept["choice"] != "unknown" and p1 >= 0.70 and p1 - p2 >= 0.20
    team = dept["choice"] if confident else "front_desk"
    facts = {"room": room_number(message), "team_asked": team,
             "urgency": ["routine", "today", "within the hour", "urgent"][min(3, round(a["urgency"]["score"]))]}
    return {"answers": a, "resp": resp, "team": team, "confident": confident,
            "language": language(message), "facts": facts}


def write_reply(message: str, routed: dict, provider=None, model=None, max_tokens: int = 600) -> dict:
    """The LLM step. The prompt is picked by CODE from the approved registry."""
    system = APPROVED_PROMPTS[routed["team"]].format(language=LANG_NAME[routed["language"]])
    user = (f"Guest message:\n{message}\n\n"
            f"Facts you may use (nothing else):\n"
            f"- room: {routed['facts']['room'] or 'not given'}\n"
            f"- team asked to help: {routed['facts']['team_asked']}\n"
            f"- urgency: {routed['facts']['urgency']}\n\n"
            f"Write the reply draft now.")
    return llmkit.generate(provider, system=system, user=user, model=model, max_tokens=max_tokens)


# ── 3 · Jev checks the DRAFT (verify-and-escalate) ────────────────────────────
CHECK_QUESTIONS = guarded({
    "promises_compensation": noul(
        "Does `draft` promise or offer a refund, discount, free night, upgrade or other compensation?"),
    "invents_facts": noul(
        "Does `draft` state a specific fact — a person's name, a time or deadline, a price, or a cause of the "
        "problem — that appears in neither `guest_message` nor `facts`?"),
    "reply_language": choice("Which language is `draft` mainly written in?", {
        "en": "English", "th": "Thai", "mixed": "A real mix of Thai and English sentences",
        "other": "Any other language"}),
    "addresses_request": noul("Does `draft` respond to what the guest actually asked for in `guest_message`?"),
})


def review(answers: dict, facts: dict, draft: str, guest_lang: str) -> dict:
    """Deterministic policy over Jev's judgments + exact code checks. Never sends anything."""
    reasons = []
    if answers["promises_compensation"]["noul"] >= 0.5:
        reasons.append("promises compensation — only a manager may offer that")
    if answers["invents_facts"]["noul"] >= 0.5:
        reasons.append("states facts that are not in the message or the facts list")
    if answers["reply_language"]["choice"] != guest_lang:
        reasons.append(f"written in {answers['reply_language']['choice']}, guest wrote in {guest_lang}")
    if answers["addresses_request"]["noul"] < 0.5:
        reasons.append("does not answer what the guest asked")
    if facts.get("room") and facts["room"] not in draft:        # exact fact → plain string check
        reasons.append(f"room {facts['room']} is missing from the draft")
    return {"route": "human_review" if reasons else "staff_approval", "reasons": reasons, "execute": False}


def check_draft(message: str, routed: dict, draft: str) -> dict:
    state = {"guest_message": message, "facts": routed["facts"], "draft": draft}
    resp = ask(state, CHECK_QUESTIONS, quiet=True)
    a = validate(resp, CHECK_QUESTIONS)
    return {"answers": a, "resp": resp, "decision": review(a, routed["facts"], draft, routed["language"])}
