#!/usr/bin/env python3
"""Exercise 10 · The capstone grader: is the Hotel Ops Ensemble ready for production? Reference solution.

Points the same grader at the reference files next to this script (one of several configurations that pass):
    exercises/solutions/capstone.yaml       9 controls, 15 of 16 points, careful approver, HITL off
    exercises/solutions/threat_model.yaml   the 8 design-time threats mitigated, 4 added from the suite,
                                            1 accepted, 1 honestly open

It writes its own report to .runs/maestro_report_solution.md first (your .runs/maestro_report.md is untouched),
then grades. Run:
    .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/exercises/solutions/ex10_capstone_grade.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import capkit as ck  # noqa: E402
from maskit import banner, finish  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "capstone.yaml"
THREAT_MODEL = Path(__file__).resolve().parent / "threat_model.yaml"


# ─────────────────────────── grader — no need to edit below ─────────────────
def main() -> None:
    banner("Exercise 10 · the capstone grader", f"solutions/{CONFIG.name} + solutions/{THREAT_MODEL.name} · SIM brain · "
           "the course's rubric", brain=False)
    report = ck.write_report(CONFIG, THREAT_MODEL, ck.report_path(solution=True))
    good = ck.grade(CONFIG, THREAT_MODEL, report)
    finish(good, "Ready for the GM: 0/12 attacks, 8/8 tasks, within budget, every claim backed by evidence.")


if __name__ == "__main__":
    main()
