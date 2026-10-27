#!/usr/bin/env python3
"""Lab 03-1 · Two scopes and the ladder: which editor owns a failure, and which component it reaches for.

Prints the key lines of the repository's INV and DOM editor templates and its "prefer, in this order" ladder
(verbatim, from common/sources/harness2_prompts.py.txt). Then it runs the base harness on every task of the
refine half (three seeds, both executors), lists every failure kind an editor could SEE, and shows which scope
owns it and which component the course's SIM editors write for it (h2kit.inv_edits / DOM_EDITS), with the
always-on cost of each. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/labs/lab03_1_two_scopes_and_the_ladder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                      # noqa: E402
from ops_sim import execute, tasks                                   # noqa: E402
from h2kit import DOM_EDITS, Edit, apply, inv_edits, observe             # noqa: E402
from h2out import banner, dim, note, ok, result, show_text, step, table  # noqa: E402

QUOTE_INV = [
    "# YOUR DIMENSION: INV -- how the agent WORKS",
    "You own procedure, tool use and completion -- not subject matter. Trace the workflow, fix its earliest "
    "observed divergence, and match each failure to the cheapest effective component:",
    "- TOOL ERROR: distinguish bad input from timeout, missing capability, permission, or unread output. Give one "
    "bounded fallback and stop condition; never prescribe blind repetition.",
    "- MECHANICAL REPETITION: use `scripts/<name>.py` for repeated deterministic extraction, conversion or checking, "
    "with clear inputs, outputs and failure behavior.",
    "- TRIGGERED WORKFLOW: use an on-demand skill for a narrow multi-step procedure, name its trigger in always-on "
    "text, and carry any helper it requires.",
    "- IGNORED INVARIANT: only repeated evidence that clear prose is skipped can justify a defensive",
    "- ORDINARY SEQUENCING: prefer concise always-on guidance. Do not add code for a one-off mistake or automate "
    "professional judgment.",
    "Do not add professional rules, required domain sections or matter facts. A parallel candidate owns them; stay "
    "focused on how the agent works.",
]
QUOTE_DOM = [
    "# YOUR DIMENSION: DOM -- what a PRACTITIONER knows",
    "You own the practitioner model the agent lacks:",
    "- ANALYTIC MOVE: name the operation -- compare, reconcile, calculate, classify, trace, draft or explain -- its "
    "inputs and required result.",
    "- ARTIFACT CONTRACT: name required parts, their dependencies, and domain checks for consistency and practical use.",
    "- DECISION RULE: state the trigger, supporting evidence, conflicts to resolve, and uncertainty that must remain "
    "explicit.",
    "- ENCODING: choose the FORM as deliberately as the content. A skill is the default home for domain knowledge that "
    "has a recognisable trigger",
    "Do not diagnose commands, retries, reading order, output paths, budgeting, scripts or hooks. A parallel candidate "
    "owns procedure; stay on practitioner knowledge.",
]
QUOTE_LADDER = [
    "So prefer, in this order:",
    "- a SKILL for anything with a recognisable trigger -- a procedure for a document type, a format, a recurring "
    "situation. This is the default form for new instruction.",
    "- a SCRIPT for a deterministic operation the runs hand-rolled and got wrong, or did slowly.",
    "- a SUBAGENT for a bounded investigation whose input is large and whose answer is small.",
    "- a PLUGIN when a recurring, mechanical problem in the tool loop is better handled there than in prose.",
    "- ALWAYS-ON TEXT last, and only for a short rule that genuinely applies to every run. If you add always-on text, "
    "cut something else to pay for it.",
]

# The course's reading: which template line each observable failure falls under (our table, not the repository's).
CLASS = {
    "blind_retry": "TOOL ERROR", "hand_arithmetic": "MECHANICAL REPETITION",
    "no_reopen": "completion: reopen or check the deliverable", "no_cmms": "ORDINARY SEQUENCING",
    "no_evidence": "ORDINARY SEQUENCING → IGNORED INVARIANT", "no_deliverable": "completion: reserve room for every artifact",
    "no_analytic_move": "ANALYTIC MOVE", "storm_spam": "DECISION RULE", "nuisance_notify": "DECISION RULE",
    "digest_contract": "ARTIFACT CONTRACT", "thai_register": "ENCODING",
}
RUNG = {"C6": "1 skill", "C7": "2 script", "C5": "3 subagent", "C8": "4 plugin"}

banner("Lab 03-1 · Two scopes and the ladder", "INV vs DOM · failure → scope → cheapest effective component", brain=False)
dim("SIM editors = rule tables after the templates below (common/h2kit.py); a teaching model, not the repository")

step(1, "The INV template: how the agent WORKS (harness2/core/prompts.py)")
show_text("_INV_TEMPLATE (key lines)", "\n".join(QUOTE_INV))
step(2, "The DOM template: what a PRACTITIONER knows (harness2/core/prompts.py)")
show_text("_DOM_TEMPLATE (key lines)", "\n".join(QUOTE_DOM))
step(3, "The ladder both editors are given (harness2/core/prompts.py · # WHAT YOU CAN CHANGE)")
show_text("prefer, in this order", "\n".join(QUOTE_LADDER))

step(4, "Every failure an editor could SEE in base runs of the refine half (3 seeds × 2 executors)")
base = H.base_harness()
seen: dict[str, set] = {}
example = {}
for t in tasks("refine"):
    for ex in ("sim-cloud", "sim-local"):
        for s in range(3):
            for o in observe(execute(base, t, seed=s, executor=ex), t, human_reason=t.human_reason):
                seen.setdefault(o.kind, set()).add(t.id)
                example.setdefault(o.kind, (t, o.scope))
rows = []
for kind in sorted(seen, key=lambda k: (example[k][1], k)):
    t, scope = example[kind]
    if scope == "inv":
        files, line = inv_edits(kind, base)
        files = {**files, **({"tool_guidance.md": line} if line else {})}
    else:
        files, line = DOM_EDITS[kind](t)
        files = {**files, "systemprompt.md": line}
    tree = apply(base, [Edit(kind, scope, files, "")]) if files else base
    comps = sorted({H.component_of(p) for p in files})
    on_demand = [c for c in comps if c in RUNG]
    rung = RUNG[on_demand[0]] if on_demand else ("5 always-on" if comps else "— (no edit)")
    rows.append([kind, scope.upper(), CLASS[kind], " + ".join(comps) or "—", rung,
                 f"{H.always_on_tokens(tree) - H.always_on_tokens(base):+d}", len(seen[kind])])
table(rows, ["failure kind", "scope", "template line (our reading)", "kit writes", "ladder rung", "Δ always-on", "tasks"], maxw=44)
note("every DOM row lands on rung 1 (a skill) plus one trigger line in systemprompt.md — the ENCODING rule")
note("INV rows split: a script (rung 2) when the runs hand-rolled something; one short always-on line for ordinary "
     "sequencing and for a tool error's bounded fallback — the 'short rule that genuinely applies' exception")
note("no_deliverable gets no edit of its own: the kit fixes its cause (the blind retry), never 'try harder' prose")
note("no_evidence starts as one always-on line; Lab 03-3 shows when it may climb to a plugin")

step(5, "Scope discipline: what each editor may NOT write")
inv_paths = set()
dom_paths = set()
for kind, (t, scope) in example.items():
    if scope == "inv":
        f, line = inv_edits(kind, base)
        inv_paths |= set(f) | ({"tool_guidance.md"} if line else set())
    else:
        f, _ = DOM_EDITS[kind](t)
        dom_paths |= set(f) | {"systemprompt.md"}
table([["INV", ", ".join(sorted({H.component_of(p) for p in inv_paths})), "domain rules, required sections, matter facts"],
       ["DOM", ", ".join(sorted({H.component_of(p) for p in dom_paths})), "commands, retries, budgeting, scripts, hooks"]],
      ["editor", "components it wrote here", "never writes (its template's last line)"])
ok("on these tasks the kit's two editors never wrote the same component: INV → C4 / C7 (C8 in Lab 03-3), "
   "DOM → C6 + its C1 trigger line") \
    if not ({H.component_of(p) for p in inv_paths} & {H.component_of(p) for p in dom_paths}) else \
    note("the editors overlapped on a component — the composer must resolve it, not concatenate")
note("the split is by OWNERSHIP, not by file type: the INV template may also write a skill for a TRIGGERED WORKFLOW "
     "(a procedure that holds whatever the subject); the kit's INV editor never needed one on these tasks")
result(f"{len(rows)} failure kinds seen · {sum(1 for r in rows if r[1] == 'DOM')} DOM → skills · "
       f"{sum(1 for r in rows if r[1] == 'INV')} INV → scripts, guidance lines or nothing")
