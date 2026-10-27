#!/usr/bin/env python3
"""Solution 07 · Diagnose six harnessing incidents, name the control, and write the egress rule.

Three TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/exercises/ex07_diagnose_the_incident.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from h2out import check, finish        # noqa: E402

MODES = ("blind selection", "premise caps the pool", "attribution noise", "rubric leak")
CONTROLS = ("selector audit and human review before Serve", "re-verify facts and probe the shared assumption",
            "repeat the probes", "block the hosts at the network level",
            "measurement-word ban and grader-verdict check")

INCIDENTS = [
    "1. At k=5 the delivered digest scored 60 on the after-the-fact judge; two other finals of the same step scored 100.",
    "2. At the new hotel every candidate harness carried 'approach limit 3.0 K' from the library, and all five finals "
    "made the same wrong diagnosis. The selector had nothing better to choose.",
    "3. An edit row says 'helped' for a new tool_guidance line. One probe stands behind it, and the same candidate "
    "also changed a skill.",
    "4. During a Harness Bench run the executor's trajectory shows web.fetch(huggingface.co/…/RUBRICS.json), and its "
    "judge scores jump.",
    "5. A candidate SKILL.md proposed for the library says: 'The grader expects a CSV, so always produce a CSV.'",
    "6. A failure no edit targeted appeared in one probe, and the composer dropped a working skill as 'hurt'.",
]

# TODO 1 — the failure mode each incident shows (one of MODES), in order.
DIAGNOSIS = ["blind selection", "premise caps the pool", "attribution noise", "rubric leak", "rubric leak",
             "attribution noise"]

# TODO 2 — the control that addresses each incident (one of CONTROLS), in order.
CONTROL = [CONTROLS[0],    # the better final was in the pool: audit the selector, a human reads before Serve
           CONTROLS[1],    # every candidate shares the premise: re-verify it against this task's sources
           CONTROLS[2],    # one rollout, two components changed: repeat the probes (and say confounded until then)
           CONTROLS[3],    # the rubric was reachable: block the hosts at the network level
           CONTROLS[4],    # grader language in a harness file: ban the words, refuse the verdict
           CONTROLS[2]]    # variance blamed on an edit: repeated executions separate the edit from noise


# TODO 3 — the egress rule. Return True if the executor may NOT reach `host`. Block exactly the hosts
#          docs/EXPERIMENTS.md lists: huggingface.co, hf.co, any subdomain of hf.co (*.hf.co), github.com and
#          raw.githubusercontent.com. Everything else is allowed. (fnmatch from the standard library helps.)
def blocked(host: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(host, pat) for pat in
               ("huggingface.co", "hf.co", "*.hf.co", "github.com", "raw.githubusercontent.com"))


# ── checker (do not edit below) ────────────────────────────────────────────────
good = True
KEY_MODE = ["blind selection", "premise caps the pool", "attribution noise", "rubric leak", "rubric leak",
            "attribution noise"]
KEY_CTRL = [CONTROLS[0], CONTROLS[1], CONTROLS[2], CONTROLS[3], CONTROLS[4], CONTROLS[2]]
bad = [str(i + 1) for i, (a, b) in enumerate(zip(DIAGNOSIS, KEY_MODE)) if a != b]
good &= check(not bad, "six incidents diagnosed: 1 selection, 2 premise, 3 and 6 noise, 4 and 5 leak",
              f"diagnosis wrong for incident(s) {', '.join(bad)} — ask: was the better answer in the pool? did "
              "every candidate share it? does one rollout stand behind the verdict? did measurement reach the work?")
bad = [str(i + 1) for i, (a, b) in enumerate(zip(CONTROL, KEY_CTRL)) if a != b]
good &= check(not bad, "six controls named — none of them shows the loop a score",
              f"control wrong for incident(s) {', '.join(bad)}")
CASES = {"huggingface.co": True, "hf.co": True, "cdn-lfs.hf.co": True, "github.com": True,
         "raw.githubusercontent.com": True, "docs.example-hotel.test": False, "pypi.org": False,
         "nothf.co": False, "huggingface.co.example.test": False}
wrong = [h for h, want in CASES.items() if bool(blocked(h)) != want]
good &= check(not wrong, f"egress rule right on all {len(CASES)} hosts (cdn-lfs.hf.co blocked, nothf.co allowed)",
              f"egress rule wrong for: {', '.join(wrong)}")
finish(good, "Done. Module 08 puts these controls into Ops Assistant's nightly harness review, behind a human gate.")
