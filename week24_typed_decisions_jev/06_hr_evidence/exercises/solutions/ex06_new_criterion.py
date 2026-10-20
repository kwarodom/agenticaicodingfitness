#!/usr/bin/env python3
"""Solution · Exercise 06 — a cloud criterion and a fair checklist row."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import choice  # noqa: E402

import ex06_new_criterion as ex  # noqa: E402

ex.cloud_evidence = choice(
    "Does `professional_text` show hands-on deployment or operation of services on a cloud platform "
    "such as AWS, Azure or GCP? An interest, a course or a certificate alone is not hands-on work.", {
        "evidenced": "A specific cloud deployment, migration or production operation is described.",
        "not_stated": "No cloud deployment work is described.",
        "unclear": "A cloud platform is named, but hands-on work is ambiguous.",
    })


def checklist_row(answers: dict) -> dict:
    return {
        "cloud": answers["cloud_evidence"]["choice"],
        # "not evidenced" means: ask the candidate — it is a follow-up list, not a penalty
        "follow_up": sorted(k for k, a in answers.items() if a["choice"] != "evidenced"),
        "route": "recruiter_review",
    }


ex.checklist_row = checklist_row

if __name__ == "__main__":
    ex.main()
