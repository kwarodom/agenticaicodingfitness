#!/usr/bin/env python3
"""Lab 12-2 · Shadow mode — run the new router BESIDE the old one, change nothing.

Before a new router touches live traffic, you run it in the shadow: every
message still goes where the CURRENT system sends it, and you only log what the
new one WOULD have done. Disagreements are the interesting rows.

Here the "current system" is a typical keyword router. The candidate is the
Jev copilot from Lab 12-1. The human-labeled "right team" lets us score both.

Run: .venv/bin/python week24/12_capstone_hotel_copilot/labs/lab02_shadow_mode.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copilot_reference as copilot          # noqa: E402
from jevkit import banner, step, table       # noqa: E402

# The incumbent: first keyword wins. Cheap, fast, and brittle.
KEYWORDS = [
    ("hvac", ["air con", "aircon", " ac ", "the ac", "warm", "hot", "cold", "แอร์"]),
    ("housekeeping", ["towel", "clean", "pillow", "coffee", "sheet", "แม่บ้าน", "ผ้าเช็ดตัว"]),
    ("maintenance", ["toilet", "shower", "light", "tv", "leak", "broken", "ชักโครก", "น้ำร้อน"]),
    ("food_beverage", ["room service", "breakfast", "food", "menu", "อาหาร"]),
    ("front_desk", ["bill", "checkout", "check-out", "key", "invoice"]),
]


def keyword_route(message: str) -> str:
    text = f" {message.lower()} "
    for team, words in KEYWORDS:
        if any(w in text for w in words):
            return team
    return "front_desk_review"


# (message, the team a human dispatcher says is right)
CASES = [(m, gold) for m, gold in zip(copilot.MESSAGES, [
    "hvac", "hvac", "housekeeping", "duty_manager", "housekeeping",
    "food_beverage", "duty_manager", "maintenance", "front_desk_review"])] + [
    ("It's freezing cold in here and the room is too dark — the lights don't work. Room 312.", "maintenance"),
    ("Is breakfast included in my booking?", "front_desk"),
    ("ขอผ้าเช็ดตัวเพิ่มสองผืนค่ะ ห้อง 1402", "housekeeping"),
    ("The shower is fine now, thanks. But my room key stopped working. 705", "front_desk"),
]

banner("Lab 12-2 · shadow mode", f"{len(CASES)} messages · incumbent keyword router vs Jev copilot")

step(1, "route every message both ways — the incumbent's answer is the one that 'ships'")
rows, kw_ok, jev_ok, agree = [], 0, 0, 0
for msg, gold in CASES:
    kw = keyword_route(msg)
    jev = copilot.handle(msg)["route"]
    kw_ok += kw == gold
    jev_ok += jev == gold
    agree += kw == jev
    rows.append([msg if len(msg) <= 40 else msg[:39] + "…", gold, kw, jev,
                 "=" if kw == jev else "≠", "✓" if jev == gold else "✕"])
table(rows, ["message", "human", "keyword (live)", "Jev (shadow)", "", "Jev ok"])

step(2, "the shadow report")
n = len(CASES)
print(f"◆ agreement keyword vs Jev : {agree}/{n}")
print(f"◆ keyword router correct   : {kw_ok}/{n}")
print(f"◆ Jev copilot correct      : {jev_ok}/{n}")
print("→ Look at every ≠ row: is it a keyword false match ('room service' ≠ room problem,")
print("  'fine now' ≠ broken shower), a Thai phrasing, or a safety case only one router saw?")
print("→ Where Jev is ✕, fix the questions or the policy — then re-run on the SAME rows.")
print("═ execute: False — in shadow mode nothing changes for guests; you only collect evidence.")
print("  Promote the new router only when per-class EN/TH results, safety misses and review")
print("  load meet targets on a larger held-out set (Lab 09), not on these 13 rows.")
