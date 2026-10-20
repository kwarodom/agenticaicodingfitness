#!/usr/bin/env python3
"""Exercise 10 · Is this a fair Jev-vs-Laya comparison?  (offline, $0)

Two published articles "disagree" about Jev vs Laya mostly because they are
not the same experiment. Your job: write the referee.

TODO 1 — fill PILOT_PLAN: the fair-pilot plan for an AltoTech EN/TH router.
TODO 2 — implement is_fair_comparison(run_a, run_b) → list of problems
         (an EMPTY list means "fair to compare").

Rules the referee must enforce (one problem string per broken rule):
  R1  same corpus            run_a["corpus_sha256"] == run_b["corpus_sha256"]
  R2  held-out test split    both runs have split == "test"
  R3  same questions         run_a["schema_sha256"] == run_b["schema_sha256"]
  R4  thresholds not tuned on the test split   (thresholds_tuned_on != "test", both runs)
  R5  like with like         run_a["fine_tuned"] == run_b["fine_tuned"]
  R6  no hidden truncation   truncated_rows == 0 in both runs
  R7  failures reported      failures_reported is True in both runs

No API call is ever made — this is pure reasoning about experiments.

Run: .venv/bin/python week24/10_jev_vs_laya/exercises/ex10_fair_pilot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import check, table  # noqa: E402

# ── TODO 1 ── the pilot plan. Replace every None.
PILOT_PLAN = {
    "splits": None,            # list — must include "train" or "calibration" AND "test"
    "languages": None,         # list — must include "en" and "th"
    "metrics": None,           # list — include "per_class_f1", "wrong_accepted", "calibration", "p95_latency"
    "thresholds_tuned_on": None,  # which split chooses thresholds? (never "test")
    "shadow_mode_first": None,    # True/False — run beside the current router before switching?
    "cloud_fallback_when_local_uncertain": None,  # True/False — see the tutorial's hybrid rule!
}


# ── TODO 2 ── the referee.
def is_fair_comparison(run_a: dict, run_b: dict) -> list:
    problems = []
    # R1 … R7 — append one short string per broken rule, e.g. problems.append("R1 different corpus")
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
BASE = {"corpus_sha256": "c0ffee", "split": "test", "schema_sha256": "5c4e", "thresholds_tuned_on": "calibration",
        "fine_tuned": False, "truncated_rows": 0, "failures_reported": True}


def variant(**changes):
    return {**BASE, **changes}


FIXTURES = [
    ("identical fair setup", BASE, variant(), 0),
    ("different corpora (the two articles!)", BASE, variant(corpus_sha256="beef"), 1),
    ("fine-tuned Laya vs zero-shot Jev", BASE, variant(fine_tuned=True), 1),
    ("threshold chosen on the test set", BASE, variant(thresholds_tuned_on="test"), 1),
    ("Laya silently truncated 3 rows", BASE, variant(truncated_rows=3), 1),
    ("smoke split + different prompts + failures hidden", variant(split="smoke"),
     variant(split="smoke", schema_sha256="aaaa", failures_reported=False), 3),
]


def main() -> None:
    print("━" * 72)
    print("━━ Exercise 10 · the fair-pilot referee (offline)")
    print("━" * 72)
    ok = True
    p = PILOT_PLAN
    ok &= check(isinstance(p["splits"], list) and "test" in p["splits"]
                and bool({"train", "calibration"} & set(p["splits"])),
                "splits separate tuning data from a held-out test set",
                "TODO 1: splits needs 'test' plus 'train' or 'calibration'")
    ok &= check(isinstance(p["languages"], list) and {"en", "th"} <= set(p["languages"]),
                "languages cover English and Thai", "TODO 1: languages must include 'en' and 'th'")
    need = {"per_class_f1", "wrong_accepted", "calibration", "p95_latency"}
    ok &= check(isinstance(p["metrics"], list) and need <= set(p["metrics"]),
                "metrics include per-class F1, wrong-accepted, calibration, p95 latency",
                f"TODO 1: metrics must include {sorted(need)}")
    ok &= check(p["thresholds_tuned_on"] in ("calibration", "train", "development"),
                "thresholds are tuned off the test set", "TODO 1: thresholds_tuned_on must not be 'test' or None")
    ok &= check(p["shadow_mode_first"] is True, "shadow mode before switching routing",
                "TODO 1: shadow_mode_first — should the pilot touch live routing on day one?")
    ok &= check(p["cloud_fallback_when_local_uncertain"] is False,
                "no automatic cloud fallback just because the local model was unsure",
                "TODO 1: cloud eligibility is a POLICY decided before escalation, not a reaction to uncertainty")

    rows = []
    for name, a, b, want in FIXTURES:
        got = is_fair_comparison(a, b)
        good = isinstance(got, list) and len(got) == want
        ok &= good
        rows.append(["✓" if good else "✕", name, want, "—" if not isinstance(got, list) else len(got),
                     "; ".join(got)[:60] if isinstance(got, list) else f"returned {got!r}"])
    table(rows, ["", "fixture", "expected problems", "yours", "your messages"])
    if not ok:
        print("\n⚠ fix the ✕ lines, save, and run again.")
        sys.exit(1)
    print("\n═ Your referee would have flagged the published Jev-vs-Laya comparison: different")
    print("  corpora, a fine-tuned checkpoint, different prompts. Neither article is 'wrong' —")
    print("  they are simply not the same experiment.")


if __name__ == "__main__":
    main()
