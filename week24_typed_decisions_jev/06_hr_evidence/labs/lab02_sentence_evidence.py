#!/usr/bin/env python3
"""Lab 06-2 · Select, don't generate — point to the exact sentence.

A label like "evidenced" is hard to audit. So:
  1. CODE splits the text into numbered sentences (s1, s2, …).
  2. Jev CHOOSES which sentence id is the evidence — or "not_stated".
  3. CODE prints that sentence VERBATIM from the original text.

Jev never writes a quotation, so it can never invent one. The recruiter sees
real words they can check in seconds.

Run: .venv/bin/python week24/06_hr_evidence/labs/lab02_sentence_evidence.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parent))
from jevkit import GUARD, ask, banner, step, table, validate  # noqa: E402

from lab01_evidence_checklist import load_profiles  # noqa: E402

CRITERIA = {
    "python": "hands-on Python implementation work",
    "integration": "hands-on REST API, BACnet or Modbus integration work",
    "building": "work with HVAC, BMS or building energy systems",
}


def split_sentences(text: str) -> dict[str, str]:
    """Deterministic sentence split, done in code (good enough for short summaries)."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return {f"s{i}": s for i, s in enumerate(parts, 1)}


def evidence_questions(sentences: dict[str, str]) -> dict:
    """One Choice per criterion. The OPTIONS are the sentence ids, plus not_stated."""
    qs = {}
    for key, what in CRITERIA.items():
        options = {sid: f"Sentence {sid}: {text}" for sid, text in sentences.items()}
        options["not_stated"] = f"No sentence describes {what}."
        qs[key] = {"type": "choice",
                   "instructions": GUARD + f"Which ONE sentence in `sentences` is the clearest explicit "
                                           f"evidence of {what}? Mentioning an interest or a course is not "
                                           f"hands-on work.",
                   "criteria": options}
    return qs


def main() -> None:
    banner("Lab 06-2 · sentence-level evidence", "code splits → Jev selects an id → code quotes verbatim")
    rows = []
    profiles = load_profiles()
    for n, p in enumerate(profiles, 1):
        sentences = split_sentences(p["professional_text"])
        qs = evidence_questions(sentences)
        step(n, f"{p['candidate_id']} — {len(sentences)} sentences")
        for sid, s in sentences.items():
            print(f"   {sid}: {s}")
        a = validate(ask({"sentences": sentences}, qs, quiet=True), qs)
        for key in CRITERIA:
            pick, conf = a[key]["choice"], a[key]["confidence"]
            quote = sentences.get(pick, "—")
            print(f"» {key:<12} → {pick:<10} conf {conf:.2f}   “{quote}”" if pick != "not_stated"
                  else f"» {key:<12} → not_stated conf {conf:.2f}   (ask the candidate)")
            rows.append([p["candidate_id"], key, pick, f"{conf:.2f}"])

    step(len(profiles) + 1, "summary for the recruiter")
    table(rows, ["candidate", "criterion", "evidence", "conf"])
    print("◆ every quoted sentence above was copied from the original text by code — Jev only chose an id")
    print("═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True")


if __name__ == "__main__":
    main()
