#!/usr/bin/env python3
"""Solution · Exercise 05 — the review policy, rules in priority order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ex05_review_policy as ex  # noqa: E402


def route(answers: dict) -> str:
    cat = answers["category"]
    if answers["suspicious"]["noul"] >= 0.5:      # 1. fraud first — beats any confident label
        return "fraud_desk"
    if cat["choice"] == "other":                  # 2. no dominant intent
        return "human_review"
    if cat["confidence"] < 0.75:                  # 3. torn between labels
        return "human_review"
    if answers["urgent"]["noul"] >= 0.5:          # 4. outages must reach a person quickly
        return "human_review"
    return cat["choice"]                          # 5. safe to propose the label


ex.route = route

if __name__ == "__main__":
    ex.main()
