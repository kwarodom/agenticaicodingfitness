#!/usr/bin/env python3
"""Solution · Exercise 11 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import choice, guarded, noul  # noqa: E402

import ex11_guarded_reply as ex  # noqa: E402

ex.MY_CHECKS = guarded({
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


def my_review(answers, facts, draft, guest_lang):
    reasons = []
    if answers["promises_compensation"]["noul"] >= 0.5:
        reasons.append("promises compensation")
    if answers["invents_facts"]["noul"] >= 0.5:
        reasons.append("invents facts")
    if answers["reply_language"]["choice"] != guest_lang:
        reasons.append(f"wrong language ({answers['reply_language']['choice']} ≠ {guest_lang})")
    if answers["addresses_request"]["noul"] < 0.5:
        reasons.append("does not answer the request")
    if facts.get("room") and facts["room"] not in draft:
        reasons.append(f"room {facts['room']} missing")
    return {"route": "human_review" if reasons else "staff_approval", "reasons": reasons, "execute": False}


ex.my_review = my_review

if __name__ == "__main__":
    ex.main()
