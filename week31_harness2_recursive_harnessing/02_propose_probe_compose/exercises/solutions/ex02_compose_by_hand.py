#!/usr/bin/env python3
"""Solution 02 · Be the composer: keep, drop or KEEP_BASE, with the reason, for five contrast tables.

Each case below is one k = 1 step: the failures its reference showed, and for each candidate the failure its
edit targeted, whether its probe delivered, and the behavioural contrast (mitigated · unchanged · regressed).
You decide what the composer does. No score is given anywhere — exactly like the real composer.

Three TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/exercises/ex02_compose_by_hand.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from h2out import check, finish        # noqa: E402

# The composer's rule in this course's loop (common/h2kit.py · compose):
#   keep an edit   ⇔ its probe delivered, the failure it targeted is "mitigated", and the candidate regressed nothing
#   "regressed"    counts when it hits a targeted failure, no_deliverable, or a DETERMINISTIC failure kind
#                  (hand_arithmetic · no_reopen · blind_retry · no_deliverable)
#   a STOCHASTIC failure kind that no edit touched and that flips on in one probe is noted as possible
#   execution variance — it does not count against the edit (repeated executions would tell)
#   no edit kept at all ⇒ the step's outcome is KEEP_BASE
CASES = {
    "A": {"task": "F1-01 fault review (reviewer: CH-2 was already on a PM)",
          "candidates": {"inv1": ("no_cmms", True, {"no_cmms": "mitigated", "hand_arithmetic": "unchanged"}),
                         "dom1": ("no_analytic_move", True, {"no_analytic_move": "mitigated", "no_cmms": "mitigated"})}},
    "B": {"task": "F7-00 30-day energy helper (stopped after three timeouts)",
          "candidates": {"inv1": ("blind_retry", False, {"blind_retry": "mitigated", "no_deliverable": "regressed"})}},
    "C": {"task": "F2-01 morning digest (reviewer: missed today's PMs)",
          "candidates": {"inv1": ("no_reopen", True, {"no_reopen": "mitigated", "digest_contract": "unchanged"}),
                         "dom1": ("digest_contract", True, {"digest_contract": "unchanged", "no_reopen": "unchanged"})}},
    "D": {"task": "F1-04 fault review",
          "candidates": {"inv1": ("no_cmms", True, {"no_cmms": "mitigated", "no_evidence": "regressed"}),
                         "dom1": ("no_analytic_move", True, {"no_analytic_move": "mitigated", "hand_arithmetic": "regressed"})}},
    "E": {"task": "F1-10 quiet trend check (an editor added a skill anyway)",
          "candidates": {"dom1": ("no_analytic_move", True, {})}},
}
for cid, case in CASES.items():
    print(f"case {cid} · {case['task']}")
    for label, (target, delivered, con) in case["candidates"].items():
        print(f"   {label}: targets {target} · probe delivered={delivered} · contrast: "
              + (", ".join(f"{k} {v}" for k, v in con.items()) or "no difference"))
print()

# TODO 1 — for every candidate: ("keep" | "drop", reason). Reasons:
#          "mitigated" · "no observed change" · "regressed" · "not delivered"
DECISIONS = {
    "A": {"inv1": ("keep", "mitigated"), "dom1": ("keep", "mitigated")},
    "B": {"inv1": ("drop", "not delivered")},
    "C": {"inv1": ("keep", "mitigated"), "dom1": ("drop", "no observed change")},
    "D": {"inv1": ("keep", "mitigated"), "dom1": ("drop", "regressed")},
    "E": {"dom1": ("drop", "no observed change")},
}

# TODO 2 — the step's outcome per case: "COMPOSED" or "KEEP_BASE"
OUTCOME = {"A": "COMPOSED", "B": "KEEP_BASE", "C": "COMPOSED", "D": "COMPOSED", "E": "KEEP_BASE"}

# TODO 3 — in case D one "regressed" entry is the kind the composer only NOTES as possible execution variance.
#          Which failure kind is it?
NOISE_IN_D = "no_evidence"


# ── checker (do not edit below) ────────────────────────────────────────────────
KEY = {
    "A": {"inv1": ("keep", "mitigated"), "dom1": ("keep", "mitigated")},
    "B": {"inv1": ("drop", "not delivered")},
    "C": {"inv1": ("keep", "mitigated"), "dom1": ("drop", "no observed change")},
    "D": {"inv1": ("keep", "mitigated"), "dom1": ("drop", "regressed")},
    "E": {"dom1": ("drop", "no observed change")},
}
KEY_OUT = {"A": "COMPOSED", "B": "KEEP_BASE", "C": "COMPOSED", "D": "COMPOSED", "E": "KEEP_BASE"}
good = True
wrong = [f"{c}/{lab}" for c in KEY for lab in KEY[c] if tuple(DECISIONS.get(c, {}).get(lab, ())) != KEY[c][lab]]
good &= check(not wrong, "every keep/drop decision and its reason match the composer's rule",
              f"decision or reason wrong for: {', '.join(wrong)}")
bad_out = [c for c in KEY_OUT if OUTCOME.get(c) != KEY_OUT[c]]
good &= check(not bad_out, "outcomes right: B and E are KEEP_BASE — no edit earned its place",
              f"outcome wrong for case(s): {', '.join(bad_out)} (KEEP_BASE ⇔ no edit kept)")
good &= check(NOISE_IN_D == "no_evidence",
              "no_evidence in D is noted as possible variance: stochastic, and no edit touched it",
              "which regressed kind in D is stochastic AND untargeted? (hand_arithmetic is deterministic)")
finish(good, "Done. Module 03 asks WHERE each kept edit should live: the cheapest effective component.")
