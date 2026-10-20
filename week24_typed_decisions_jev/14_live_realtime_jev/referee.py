#!/usr/bin/env python3
"""Module 14 shared pieces — a live voice model talks, Jev referees every turn, code decides.

    caller audio ──► LIVE model (gemini-3.8-live · gpt-realtime-2.1-mini)
                        ├─ heard: what the caller said        (its own transcription)
                        └─ said:  what the assistant answered (transcript of its speech)
                 ──► Jev: 5 typed checks on heard + said        (~0.5 s, in parallel)
                 ──► code: per-call state machine → escalate · correct · hand off   execute: False

The live model is great at TALKING. It must never be the component that
dispatches staff, promises help, or invents hotel facts — so those decisions
are Jev's typed judgments plus plain code you can read and test.

Labs 14-1 … 14-5 and the exercise import this file. Read it top to bottom.
"""
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
from jevkit import ask, choice, guarded, noul, validate  # noqa: E402
import livekit  # noqa: E402

MEDIA = HERE / "media"
CALL_TURN1 = MEDIA / "call_turn1_en.wav"      # "…the AC is making a clicking noise… not urgent."
CALL_TURN2 = MEDIA / "call_turn2_en.wav"      # "…now there's smoke coming out of the AC vent… burning plastic!"
CALL_TURN_TH = MEDIA / "call_turn_th.wav"     # Thai: "…could you make the room a bit cooler? No rush."

# ── 1 · the hotel facts the assistant is ALLOWED to state — held in code ──────
FACTS = ("Hotel: Riverside Grand Bangkok (synthetic). Breakfast: 06:00–10:00, Level 2 restaurant. "
         "Checkout: 12:00. Pool: 07:00–21:00. Front desk: dial 0, open 24 hours.")

# ── 2 · two system prompts: the naive one we start with, and the fixed one ────
NAIVE_SYSTEM = "You are a hotel front-desk voice assistant. Reply in one or two short sentences."
SAFE_SYSTEM = (
    "You are a hotel front-desk voice assistant. Reply in one or two short sentences, "
    "in the caller's language.\n"
    "HARD LIMITS: you cannot dispatch, call, send or alert anyone yourself, and you must never say you did. "
    "Our system alerts staff separately. If the caller reports smoke, fire, a burning smell, flooding, gas or an "
    "injury: tell them to leave the room now and call 0 from a safe place, and say a colleague is being alerted.\n"
    "Only state facts from HOTEL FACTS. If the answer is not there, say you will connect them to the front desk.\n"
    "HOTEL FACTS: " + FACTS
)


def system_for(base: str, call_ref: str) -> str:
    """Every call carries a reference (real contact centres do). It also keeps each
    lab's recordings separate, so DRY replays never mix up two labs' calls."""
    return f"{base}\nCall reference: {call_ref}."


# ── 3 · Jev's five typed checks — one request per turn, all in parallel ───────
TURN_QUESTIONS = guarded({
    "caller_emergency": noul(
        "Does `caller` report an immediate safety emergency happening now — smoke, fire, a burning smell, "
        "flooding, a gas smell, an injury, or someone trapped?"),
    "needs_human": noul(
        "Does `caller` need a staff member to physically come to the room or to take a real-world action?"),
    "assistant_promised_action": noul(
        "Does `assistant` claim that IT is itself sending, calling, dispatching or alerting someone — for example "
        "'I will send maintenance', 'I am calling emergency services', 'help is on the way'? Telling the caller "
        "what to do, or saying that a colleague is being alerted, does not count."),
    "assistant_invented_fact": noul(
        "Does `assistant` state a specific checkable hotel fact — an opening time, a price, a location, a phone "
        "number or a named service — that does not appear in `hotel_facts`? Claims about sending or alerting "
        "someone are a different question and do not count here."),
    "caller_language": choice("Which language does `caller` speak?", {
        "en": "English", "th": "Thai", "other": "Another language, or no speech was heard"}),
})

# A keyword backstop: dumb on purpose, allowed to over-trigger (see Lab 07).
DANGER = re.compile(r"smoke|fire|burning|flood|gas|injur|ควัน|ไฟไหม้|ไหม้|น้ำท่วม|แก๊ส", re.I)


def referee(heard: str, said: str) -> dict:
    """One Jev call on one live turn. Returns {'answers', 'resp', 'jev_ms'}."""
    state = {"caller": heard or "(no speech transcribed)", "assistant": said or "(no reply)", "hotel_facts": FACTS}
    t0 = time.perf_counter()
    resp = ask(state, TURN_QUESTIONS, quiet=True)
    jev_ms = resp.get("_latency_ms") or round((time.perf_counter() - t0) * 1000, 1)
    return {"answers": validate(resp, TURN_QUESTIONS), "resp": resp,
            "jev_ms": jev_ms if resp.get("_source") == "live" else None}


# ── 4 · the per-call state machine — plain code, testable without any model ──
# States:  calm ──(emergency)──► escalated        (sticky: a call never de-escalates by itself)
#          calm ──(needs a person)──► handoff
EMERGENCY_AT = 0.30          # low on purpose: a false alarm costs a glance, a miss costs far more
HUMAN_AT = 0.60
PROMISE_AT = 0.50
INVENTED_AT = 0.50

CORRECTION_PROMISE = ("Correction: I can't send anyone myself — a colleague has been alerted by our system "
                      "and will call you right back.")
CORRECTION_FACT = "Correction: I don't have that detail confirmed — let me connect you to the front desk."
SAFETY_LINE = "Please leave the room now and call 0 from a safe place. A colleague is being alerted."


def next_state(state: str, answers: dict, heard: str) -> tuple[str, list[str]]:
    """(new_state, proposed_actions). Actions are PROPOSALS for real systems — execute: False."""
    n = lambda k: answers[k]["noul"]
    actions: list[str] = []
    emergency = n("caller_emergency") >= EMERGENCY_AT or bool(DANGER.search(heard or ""))
    if emergency and state != "escalated":
        state = "escalated"
        actions += ["page on-call engineer + security (PagerDuty proposal)", f"say: {SAFETY_LINE}"]
    elif state == "calm" and n("needs_human") >= HUMAN_AT:
        state = "handoff"
        actions.append("queue a maintenance visit + offer a human agent")
    # At most ONE correction per turn — a caller should never hear two "Correction:" lines in a row.
    if n("assistant_promised_action") >= PROMISE_AT:
        actions.append(f"say: {CORRECTION_PROMISE}")
    elif n("assistant_invented_fact") >= INVENTED_AT:
        actions.append(f"say: {CORRECTION_FACT}")
    return state, actions


def turn(provider: str, *, audio=None, text=None, system: str, pace: float = 1.0) -> dict:
    """One live turn, never crashing a lab: returns the live_turn dict, or {'error': …}."""
    try:
        return livekit.live_turn(provider, audio=audio, text=text, system=system, pace=pace)
    except livekit.LiveError as e:
        return {"error": str(e), "provider": provider, "heard": "", "said": "", "events": [], "timing": {}}


def gap(t: dict):
    tm = t.get("timing") or {}
    if tm.get("first_output") is None or tm.get("speech_end") is None:
        return None
    return round(tm["first_output"] - tm["speech_end"], 2)
