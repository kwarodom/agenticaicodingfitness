#!/usr/bin/env python3
"""Exercise 06 · Add a new job criterion — and a fair checklist policy.

The role now also needs cloud deployment experience.

TODO 1 — write `cloud_evidence`: a CHOICE with exactly the three labels
         evidenced / not_stated / unclear, asking whether `professional_text`
         shows hands-on deployment of services to a cloud platform
         (AWS, Azure, GCP …). Say in the instruction that an interest or a
         course is NOT hands-on work.

TODO 2 — write `checklist_row(answers)`: return a dict
             {"cloud": <label>, "follow_up": [<criteria whose label is not "evidenced">],
              "route": "recruiter_review"}
         Rules: every record goes to "recruiter_review" (never "reject"),
         and the dict must NOT contain any score, rank or total.

Run: .venv/bin/python week24/06_hr_evidence/exercises/ex06_new_criterion.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
from jevkit import ask, banner, check, choice, guarded, step, table, validate  # noqa: E402,F401

# ── TODO 1 ──
cloud_evidence = None


# ── TODO 2 ──
def checklist_row(answers: dict) -> dict:
    """answers = {"cloud_evidence": {"choice": ...}, "python_evidence": {...}, ...}"""
    return {}


# ─────────────────────────── checker — no need to edit below ────────────────
TEST_TEXTS = [
    ("Deployed Python microservices to AWS ECS with Terraform and ran them in production.", "evidenced"),
    ("Interested in learning Azure. Completed a cloud fundamentals course.", "not_stated|unclear"),
    ("Built Excel reports for the finance team.", "not_stated"),
]


def fake(**labels) -> dict:
    return {k: {"type": "choice", "choice": v, "confidence": 0.9, "probabilities": {v: 1.0}}
            for k, v in labels.items()}


def main() -> None:
    banner("Exercise 06 · a new criterion")
    step(1, "shape checks — free")
    ok = True
    crit = (cloud_evidence or {}).get("criteria") or {}
    instr = str((cloud_evidence or {}).get("instructions", "")).lower()
    ok &= check(isinstance(cloud_evidence, dict) and cloud_evidence.get("type") == "choice",
                "cloud_evidence is a choice", "TODO 1: cloud_evidence should be choice('…', {…})")
    ok &= check(set(crit) == {"evidenced", "not_stated", "unclear"},
                "labels are exactly evidenced / not_stated / unclear",
                f"TODO 1: labels must be evidenced, not_stated, unclear (got {sorted(crit)})")
    ok &= check("cloud" in instr and ("course" in instr or "interest" in instr),
                "instruction names the cloud and excludes interests/courses",
                "TODO 1: mention cloud deployment and that interests/courses are not hands-on work")

    step(2, "policy checks — free, fixed answer dicts")
    row = checklist_row(fake(cloud_evidence="unclear", python_evidence="evidenced",
                             integration_evidence="not_stated"))
    ok &= check(isinstance(row, dict) and row.get("route") == "recruiter_review",
                "every record goes to recruiter_review", "TODO 2: route must always be 'recruiter_review'")
    ok &= check(row.get("cloud") == "unclear", "cloud label copied", "TODO 2: row['cloud'] should be the cloud label")
    ok &= check(sorted(row.get("follow_up", [])) == ["cloud_evidence", "integration_evidence"],
                "follow_up lists the criteria that are not evidenced",
                f"TODO 2: follow_up should be ['cloud_evidence', 'integration_evidence'] (got {row.get('follow_up')})")
    banned = {"score", "rank", "total", "reject", "points"}
    ok &= check(not any(b in str(k).lower() for k in row for b in banned) and "reject" not in str(row.values()),
                "no score, rank or rejection in the output", "TODO 2: remove any score / rank / reject field")
    if not ok:
        print("\n⚠ fix the ✕ lines, save, run again — no API call was made.")
        sys.exit(1)

    step(3, "live — does Jev agree with the expected labels?")
    qs = guarded({"cloud_evidence": cloud_evidence})
    rows = []
    for text, want in TEST_TEXTS:
        a = validate(ask({"professional_text": text}, qs, quiet=True), qs)
        got = a["cloud_evidence"]["choice"]
        rows.append(["✓" if got in want.split("|") else "⚠", text[:52], want, got,
                     f"{a['cloud_evidence']['confidence']:.2f}"])
    table(rows, ["", "professional_text", "expected", "jev", "conf"])
    print("◆ a ⚠ row is a rubric problem to discuss, not a test failure — sharpen the wording and rerun")
    print("═ execute: False · no_ranking: True · no_auto_rejection: True")


if __name__ == "__main__":
    main()
