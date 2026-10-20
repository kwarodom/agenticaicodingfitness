#!/usr/bin/env python3
"""Module 13 shared pieces — the model PERCEIVES, Jev JUDGES, code DECIDES.

    image / audio / video ──► multimodal model (mediakit): pixels & sound → TEXT   (2–13 s)
                          ──► code: exact facts from that text (regex numbers, arithmetic, caller-ID)
                          ──► Jev: typed judgments on the TEXT (queue · safety · suspicious)  (~0.5 s)
                          ──► code: one policy for every modality → proposal     execute: False

Jev never sees a pixel or a waveform. It reads what a perception model wrote,
which is exactly why the same Jev questions and the same policy keep working
when you swap Gemini for Claude, or a photo for a voicemail.

Labs 13-1 … 13-5 and the exercise import this file. Read it top to bottom.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
RUNS = HERE / ".runs"                                   # lab outputs (gitignored)
sys.path.insert(0, str(HERE.parent / "common"))
from jevkit import ask, choice, guarded, noul, score, validate  # noqa: E402,F401
import mediakit as mk  # noqa: E402,F401

# ── 1 · IMAGES: what we ask the vision model to write down ────────────────────
SEE_THERMOSTAT = ("Read this thermostat display exactly. Reply on ONE line in this format: "
                  "room=<room>; measured_c=<number>; setpoint_c=<number>; mode=<mode>. "
                  "If any value is unreadable, write unreadable for it.")
SEE_SCENE = ("You are helping a hotel engineering team. Describe what is visible in 2-3 sentences, focusing on any "
             "safety or maintenance issue. Then add one sentence starting 'Unsure:' saying what you cannot tell from the image.")
SEE_INVOICE = ("Transcribe this invoice exactly: every line item with its amount, the VAT line, the TOTAL, and the full "
               "text of any notice or box. Plain text, one item per line.")

# code reads the numbers — never Jev
NUM = r"(-?\d+(?:\.\d+)?)"


def parse_thermostat(text: str) -> dict:
    """room / measured / setpoint / mode from the vision model's one-liner (None when missing)."""
    g = lambda pat: (m.group(1) if (m := re.search(pat, text, re.I)) else None)
    room = g(r"room\s*[=:]?\s*(\d{3,4})")
    meas = g(r"measured(?:_c|\s+temp(?:erature)?)?\s*[=:]?\s*" + NUM)
    setp = g(r"set\s*point(?:_c)?\s*[=:]?\s*" + NUM) or g(r"setpoint(?:_c)?\s*[=:]?\s*" + NUM)
    mode = g(r"mode\s*[=:]?\s*([A-Za-z]+)")
    out = {"room": room, "measured_c": float(meas) if meas else None,
           "setpoint_c": float(setp) if setp else None, "mode": mode.upper() if mode else None}
    if out["measured_c"] is not None and out["setpoint_c"] is not None:
        out["deviation_c"] = round(out["measured_c"] - out["setpoint_c"], 1)
        out["warm_deviation"] = out["deviation_c"] > 2.0            # teaching threshold, as in Lab 07
    return out


MONEY = re.compile(r"(\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2})")


def check_invoice_math(text: str) -> dict:
    """Arithmetic in code: line items + 7% VAT must equal the TOTAL (all read from the transcription)."""
    lines = [l for l in text.splitlines() if MONEY.search(l)]
    amt = lambda l: float(MONEY.findall(l)[-1].replace(",", ""))
    vat = [amt(l) for l in lines if re.search(r"\bVAT\b", l, re.I)]
    total = [amt(l) for l in lines if re.search(r"\btotal\b", l, re.I)]
    items = [amt(l) for l in lines if not re.search(r"\b(VAT|total)\b", l, re.I)]
    if not (vat and total and items):
        return {"ok": None, "why": "could not find items, VAT and total in the transcription"}
    sub = round(sum(items), 2)
    ok = abs(round(sub * 0.07, 2) - vat[0]) < 0.02 and abs(sub + vat[0] - total[0]) < 0.02
    return {"ok": ok, "subtotal": sub, "vat": vat[0], "total": total[0],
            "why": "items + 7% VAT = total" if ok else "the numbers do not add up"}


# ── 2 · Jev judges the TEXT a perception model produced ───────────────────────
SCENE_QUESTIONS = guarded({
    "queue": choice("Which engineering queue should handle the issue in `description`?", {
        "water_leak": "Water or another liquid leaking, dripping or pooling",
        "smoke_or_fire": "Smoke, steam, haze, burning, fire or sparks",
        "comfort": "Temperature or comfort complaint with no visible damage",
        "none": "Nothing wrong is described",
        "unclear": "The description is too vague or contradictory to choose",
    }),
    "safety": noul("Does `description` describe something that could injure a person or damage electrical "
                   "equipment — smoke, steam, haze, fire, sparks, or water near electrical equipment?"),
    "severity": score("How severe is the issue in `description`?", [
        "No issue", "Minor: cosmetic or slow to worsen", "Significant: needs a visit today",
        "Serious: possible safety risk now"]),
    "hedged": noul("Does `description` say the image is unclear, or that the model cannot tell something important?"),
})

INVOICE_QUESTIONS = guarded({
    "suspicious": noul("Does `invoice_text` ask to pay a new or changed bank account, demand unusually urgent "
                       "payment, or discourage the reader from verifying by phone?"),
    "category": choice("What kind of document is `invoice_text`?", {
        "supplier_invoice": "A supplier bill for goods or services",
        "receipt": "A receipt for something already paid",
        "other": "Anything else"}),
})

THERMO_QUESTIONS = guarded({
    "readable": noul("Does `vision_text` give a clear value for room, measured temperature, setpoint and mode "
                     "without saying anything is unreadable?"),
})

# ── 3 · AUDIO: transcript → code facts → Jev ──────────────────────────────────
# The phone system knows which room called. That is a TRUSTED fact; a transcript is not.
CALLER_ID = {"voicemail_flood_en.wav": "1507", "voicemail_burning_th.wav": "1203", "request_towels_en.wav": "812"}

THAI = re.compile(r"[฀-๿]")
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
THAI_WORDS = {"ศูนย์": "0", "หนึ่ง": "1", "สอง": "2", "สาม": "3", "สี่": "4", "ห้า": "5",
              "หก": "6", "เจ็ด": "7", "แปด": "8", "เก้า": "9"}
_THAI_RUN = re.compile("(?:(?:" + "|".join(THAI_WORDS) + r")\s*){3,4}")


def room_from_transcript(text: str) -> str | None:
    """Digits, Thai numerals (๑๒๐๓) or spelled Thai digits (หนึ่งสองศูนย์สาม) → '1203'."""
    t = text.translate(THAI_DIGITS)
    t = _THAI_RUN.sub(lambda m: "".join(THAI_WORDS[w] for w in re.findall("|".join(THAI_WORDS), m.group(0))), t)
    m = re.search(r"(?:room|ห้อง)\D{0,12}(\d{3,4})", t, re.I) or re.search(r"\b(\d{3,4})\b", t)
    return m.group(1) if m else None


def clean_transcript(text: str) -> str:
    """Strip timestamps like '00:00:' some transcribers prepend."""
    return re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*:?\s*", "", text).strip()


VOICE_QUESTIONS = guarded({
    "department": choice("Which hotel team should handle the request in `transcript`?", {
        "hvac": "Air conditioning, heating, ventilation, room temperature",
        "plumbing": "Water leaks, flooding, toilets, showers, drains",
        "housekeeping": "Towels, linen, cleaning, amenities",
        "front_desk": "Bookings, billing and general questions",
        "unknown": "No clear request",
    }),
    "urgency": score("How urgent is the request in `transcript`?", [
        "Routine: no time pressure", "Should be handled today", "Needs attention within the hour",
        "Possible danger right now"]),
    "safety": noul("Does `transcript` mention smoke, a burning smell, fire, sparks, flooding, or anything that "
                   "could injure a person?"),
})

REPLY = {  # approved reply templates — code fills them; no model invents promises
    "en": "Thank you for your message about room {room}. Our {team} team has been notified and a staff member "
          "will contact you shortly.",
    "th": "ขอบคุณสำหรับข้อความเกี่ยวกับห้อง {room} ทีม{team}ได้รับแจ้งแล้ว เจ้าหน้าที่จะติดต่อกลับโดยเร็วที่สุด",
}
TEAM_TH = {"hvac": "ช่างแอร์", "plumbing": "ช่างประปา", "housekeeping": "แม่บ้าน", "front_desk": "ต้อนรับ", "unknown": "ต้อนรับ"}

# ── 4 · VIDEO: the vision model lists events; code parses; Jev judges each one ─
WATCH_PLAIN = ('List what happens in this CCTV clip as a JSON array of objects '
               '{"t_seconds": number, "event": string}. Only the JSON.')
WATCH_CAREFUL = ('List what happens in this CCTV clip as a JSON array of objects {"t_seconds": number, "event": string, '
                 '"could_also_be": string}. In could_also_be, name the most likely alternative interpretation '
                 '(for example smoke vs steam) or "none". Only the JSON.')


def parse_events(text: str) -> list[dict]:
    """The perception model returns JSON text — parse it in code, tolerate a ```json fence."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", t, re.S)
        data = json.loads(m.group(0)) if m else []
    return [e for e in data if isinstance(e, dict) and "event" in e]


def event_questions(n: int) -> dict:
    """One Noul + one Choice PER EVENT, all in one Jev call (speculative fan-out, Lab 03)."""
    qs = {}
    for i in range(n):
        qs[f"hazard_{i}"] = noul(f"Does `events[{i}]` describe smoke, steam, haze, fire, sparks or leaking fluid "
                                 "that could be dangerous in a plant room?")
        qs[f"kind_{i}"] = choice(f"What is `events[{i}]` about?", {
            "smoke_or_fire": "Smoke, burning or fire", "steam_or_vapour": "Steam, vapour or mist",
            "leak": "A liquid leak", "normal": "Normal operation, nothing wrong", "unclear": "Cannot tell"})
    return guarded(qs)


# ── 5 · ONE policy for every modality ─────────────────────────────────────────
def policy(signal: dict) -> dict:
    """signal = {modality, safety, queue, suspicious?, facts, perception_ok}. Jev supplied the judgments;
    these rules are plain code you can test offline (see the exercise)."""
    reasons, route = [], signal.get("queue") or "front_desk"
    if not signal.get("perception_ok", True):
        return {"route": "human_review", "alert": False, "reasons": ["perception failed or was empty"], "execute": False}
    if signal.get("safety", 0) >= 0.3:                              # low bar: a missed hazard costs more
        reasons.append(f"safety signal {signal['safety']:.2f} ≥ 0.30")
    if signal.get("suspicious", 0) >= 0.5:
        reasons.append("suspicious payment request → finance review; never pay from this document")
        route = "finance_review"
    f = signal.get("facts") or {}
    if f.get("room_mismatch"):
        reasons.append(f"transcript says room {f.get('room_heard')} but caller-ID says {f.get('room_trusted')} → confirm")
    alert = any(r.startswith("safety") for r in reasons)
    return {"route": "safety_review" if alert else route, "alert": alert, "reasons": reasons or ["routine"],
            "execute": False}
