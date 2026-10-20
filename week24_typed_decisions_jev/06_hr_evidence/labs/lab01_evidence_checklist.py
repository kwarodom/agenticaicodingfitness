#!/usr/bin/env python3
"""Lab 06-1 · A recruiter's evidence checklist — not a ranking.

Four FICTIONAL professional summaries (no names, contacts, photos, age, gender
or nationality) → for each job criterion Jev answers evidenced / not_stated /
unclear → every record goes to a recruiter. No score, no ranking, no rejection.

Run: .venv/bin/python week24/06_hr_evidence/labs/lab01_evidence_checklist.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, guarded, show, step, table, validate  # noqa: E402

PROFILES = Path(__file__).resolve().parents[1] / "data" / "profiles.csv"

# The same three labels for every criterion. "not_stated" means the TEXT does
# not show it — never that the person lacks the skill.
LEVELS = {
    "evidenced": "A specific project, implementation or work responsibility is described.",
    "not_stated": "No such work is described in the text.",
    "unclear": "The technology or domain is named, but hands-on work is ambiguous.",
}

# Example role: AI/IoT integration engineer. Narrow, job-related criteria only.
QUESTIONS = guarded({
    "python_evidence": choice(
        "Does `professional_text` provide explicit evidence of hands-on Python implementation work?", LEVELS),
    "integration_evidence": choice(
        "Does `professional_text` show hands-on REST API, BACnet or Modbus integration work?", LEVELS),
    "building_evidence": choice(
        "Does `professional_text` state work with HVAC, BMS or building energy systems?", LEVELS),
})


def load_profiles() -> list[dict]:
    with PROFILES.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:                                   # fail loudly on bad input
        if not r.get("candidate_id", "").strip() or not r.get("professional_text", "").strip():
            raise ValueError("every row needs candidate_id and professional_text")
    return rows


def main() -> None:
    banner("Lab 06-1 · evidence checklist", "3 criteria × evidenced / not_stated / unclear — for a recruiter to verify")

    step(1, "one profile in detail")
    profiles = load_profiles()
    p = profiles[2]
    print(f"   {p['candidate_id']}: {p['professional_text']}")
    resp = ask({"professional_text": p["professional_text"]}, QUESTIONS)
    validate(resp, QUESTIONS)
    show(resp, top=3)

    step(2, f"the checklist for all {len(profiles)} profiles")
    rows = []
    for p in profiles:
        a = validate(ask({"professional_text": p["professional_text"]}, QUESTIONS, quiet=True), QUESTIONS)
        cell = lambda k: f"{a[k]['choice']} ({a[k]['confidence']:.2f})"
        rows.append([p["candidate_id"], cell("python_evidence"), cell("integration_evidence"),
                     cell("building_evidence"), "recruiter_review"])
    table(rows, ["id", "python", "integration", "building", "route"])

    step(3, "what the recruiter does next")
    print("→ open the ORIGINAL text beside each row and mark the supporting sentence yourself")
    print("→ 'not_stated' = missing from this text. Ask the candidate; do not assume they lack it")
    print("⚠ Jev returned labels, not quotations — lab 06-2 shows how to get a verifiable sentence")
    print("═ execute: False · route: recruiter_review · no_ranking: True · no_auto_rejection: True")


if __name__ == "__main__":
    main()
