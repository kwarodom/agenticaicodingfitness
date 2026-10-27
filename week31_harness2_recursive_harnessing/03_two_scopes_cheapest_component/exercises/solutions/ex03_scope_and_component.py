#!/usr/bin/env python3
"""Solution 03 · Map nine agent failures to a scope and the cheapest effective component.

The nine failures are EXAMPLE failure classes of Ops Assistant, a fictional always-on operations assistant (the
course's own design). For each, decide which editor owns it (INV: how the agent works · DOM: what a practitioner knows), which
line of that editor's template it falls under, and the cheapest component that fixes it.

Three TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/exercises/ex03_scope_and_component.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from h2out import check, finish        # noqa: E402

FAILURES = [
    "Analytics MCP times out on a 30-day query; the agent retries blindly",
    "Agent re-derives unit conversions or degree-day arithmetic in every turn",
    "Chiller trip diagnosis follows the same six steps each time",
    "Agent keeps emitting proposals without evidence links despite instructions",
    "Agent declares a digest done without reopening the produced file",
    "Reconcile BMS alarm list with CMMS work orders before proposing",
    "A chief-engineer digest must list open jobs, overnight alarms and today's PMs with IDs",
    "When to escalate a high condenser pressure alarm versus wait",
    "Thai digest phrasing conventions for hotel engineering teams",
]

# TODO 1 — the scope of each failure, in the same order: "INV" or "DOM"
SCOPE = ["INV", "INV", "INV", "INV", "INV", "DOM", "DOM", "DOM", "DOM"]

# TODO 2 — the template line each one falls under, in the same order. Choose from:
#          INV: "tool error" · "mechanical repetition" · "triggered workflow" · "ignored invariant" · "completion"
#          DOM: "analytic move" · "artifact contract" · "decision rule" · "encoding"
CLASS = ["tool error", "mechanical repetition", "triggered workflow", "ignored invariant", "completion",
         "analytic move", "artifact contract", "decision rule", "encoding"]

# TODO 3 — the cheapest effective component for each, in the same order. Choose from:
#          "C4 tool guidance (one bounded fallback)" · "C6 skill" · "C7 script" · "C8 plugin"
COMPONENT = ["C4 tool guidance (one bounded fallback)", "C7 script", "C6 skill", "C8 plugin", "C7 script",
             "C6 skill", "C6 skill", "C6 skill", "C6 skill"]


# ── checker (do not edit below) ────────────────────────────────────────────────
KEY_SCOPE = ["INV", "INV", "INV", "INV", "INV", "DOM", "DOM", "DOM", "DOM"]
KEY_CLASS = ["tool error", "mechanical repetition", "triggered workflow", "ignored invariant", "completion",
             "analytic move", "artifact contract", "decision rule", "encoding"]
KEY_COMP = ["C4 tool guidance (one bounded fallback)", "C7 script", "C6 skill", "C8 plugin", "C7 script",
            "C6 skill", "C6 skill", "C6 skill", "C6 skill"]


def wrong(got, key):
    return [str(i + 1) for i, (g, k) in enumerate(zip(list(got) + ["??"] * 9, key)) if g != k]


good = True
w = wrong(SCOPE, KEY_SCOPE)
good &= check(not w, "scopes right: five INV (how the agent works), four DOM (what a practitioner knows)",
              f"scope wrong for failure(s) {', '.join(w)} — would the edit still make sense if the subject changed?")
w = wrong(CLASS, KEY_CLASS)
good &= check(not w, "every failure sits on the right line of its editor's template",
              f"template line wrong for failure(s) {', '.join(w)}")
w = wrong(COMPONENT, KEY_COMP)
good &= check(not w, "cheapest components right: one bounded fallback, two scripts, five skills, one plugin",
              f"component wrong for failure(s) {', '.join(w)} — skill > script > subagent > plugin > always-on prose")
good &= check(COMPONENT[3:4] == ["C8 plugin"] and COMPONENT.count("C8 plugin") == 1,
              "only failure 4 earns a plugin, and only because clear prose was already skipped repeatedly",
              "a plugin needs repeated evidence that clear prose is skipped — exactly one failure here says so")
finish(good, "Done. Module 04 asks how the composer knows an edit helped: contrasts, verdicts and the evidence boundary.")
