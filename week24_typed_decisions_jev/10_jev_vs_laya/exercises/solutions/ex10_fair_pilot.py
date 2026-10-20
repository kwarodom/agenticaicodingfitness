#!/usr/bin/env python3
"""Solution · Exercise 10 — a fair-pilot plan and the seven-rule referee."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ex10_fair_pilot as ex  # noqa: E402

ex.PILOT_PLAN.update({
    "splits": ["train", "calibration", "test"],
    "languages": ["en", "th", "mixed"],
    "metrics": ["per_class_f1", "macro_f1", "wrong_accepted", "urgent_false_negatives",
                "calibration", "p50_latency", "p95_latency", "cost_per_handled_task"],
    "thresholds_tuned_on": "calibration",
    "shadow_mode_first": True,
    "cloud_fallback_when_local_uncertain": False,
})


def is_fair_comparison(run_a, run_b):
    problems = []
    if run_a["corpus_sha256"] != run_b["corpus_sha256"]:
        problems.append("R1 different corpus")
    if run_a["split"] != "test" or run_b["split"] != "test":
        problems.append("R2 not both on the held-out test split")
    if run_a["schema_sha256"] != run_b["schema_sha256"]:
        problems.append("R3 different questions/prompts")
    if "test" in (run_a["thresholds_tuned_on"], run_b["thresholds_tuned_on"]):
        problems.append("R4 thresholds tuned on test")
    if run_a["fine_tuned"] != run_b["fine_tuned"]:
        problems.append("R5 fine-tuned vs zero-shot")
    if run_a["truncated_rows"] or run_b["truncated_rows"]:
        problems.append("R6 truncated input")
    if not (run_a["failures_reported"] and run_b["failures_reported"]):
        problems.append("R7 failures not reported")
    return problems


ex.is_fair_comparison = is_fair_comparison

if __name__ == "__main__":
    ex.main()
