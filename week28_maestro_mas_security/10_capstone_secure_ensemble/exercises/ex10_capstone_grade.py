#!/usr/bin/env python3
"""Exercise 10 · The capstone grader: is the Hotel Ops Ensemble ready for production?

This file is the grader. Your work goes into two YAML files in this module's data/ folder:

  data/capstone.yaml      TODO 1 the switches you run · TODO 2 one reason each · TODO 3 the approver ·
                          TODO 4 stay within the budget
  data/threat_model.yaml  TODO 1 mitigations, status, residual scores · TODO 2 two independent controls for
                          every impact-5 threat · TODO 3 threats for the attacks nobody modelled ·
                          TODO 4 a rationale for anything open or accepted

Then write the report (lab 10-3) and run:
    .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_3_report.py
    .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/exercises/ex10_capstone_grade.py

The grader is free, offline and runs in about a second on the SIM brain. It prints one ✓/✕ line per rubric item
(the course's rubric, 100 points) and exits 1 until every line is ✓. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
from maskit import banner, finish  # noqa: E402

CONFIG = ck.CONFIG                 # data/capstone.yaml
THREAT_MODEL = ck.THREAT_MODEL     # data/threat_model.yaml


# ─────────────────────────── grader — no need to edit below ─────────────────
def main() -> None:
    banner("Exercise 10 · the capstone grader", f"{CONFIG.name} + {THREAT_MODEL.name} · SIM brain · the course's rubric",
           brain=False)
    good = ck.grade(CONFIG, THREAT_MODEL, ck.report_path())
    finish(good, "Ready for the GM: 0/12 attacks, 8/8 tasks, within budget, every claim backed by evidence.")


if __name__ == "__main__":
    main()
