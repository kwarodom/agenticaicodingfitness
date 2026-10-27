#!/usr/bin/env python3
"""Lab 03-3 · A plugin only after repeated evidence: prose first, a hook when clear prose is still skipped.

The INV template allows a defensive plugin only when there is repeated evidence that clear prose is skipped. The
course's SIM editor encodes that as: for an empty evidence list, write one always-on line; if that line is already
in the always-on text and the failure shows up again, write plugin/require_evidence.ts instead. This lab shows the
rule with and without the line present, measures the mechanism (how often evidence is empty under base, prose and
plugin), shows a two-round sequential step where round 2 escalates to the plugin, and checks that the plugin never
blocks a valid completion. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/labs/lab03_3_plugin_after_repeated_evidence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                        # noqa: E402
from ops_sim import execute, task                                      # noqa: E402
from h2kit import inv_edits, step as h2step                                # noqa: E402
from h2out import banner, dim, edit, note, ok, result, show_text, step, table, warn  # noqa: E402

QUOTE_IGNORED = ["- IGNORED INVARIANT: only repeated evidence that clear prose is skipped can justify a defensive",
                 "It must not block valid completion."]
QUOTE_HOOK = "Prose can be stopped reading; a hook cannot."

banner("Lab 03-3 · A plugin only after repeated evidence", "no_evidence: prose line → still skipped → plugin hook", brain=False)
dim("SIM executor = a teaching model: its prose-following and plugin numbers are assumptions that show the mechanism")

step(1, "The rule (harness2/core/prompts.py)")
show_text("_INV_TEMPLATE · IGNORED INVARIANT", "\n".join(QUOTE_IGNORED))
show_text("# WHAT YOU CAN CHANGE · plugin/", QUOTE_HOOK)

step(2, "inv_edits('no_evidence', tree) — without and with the prose line already present")
base = H.base_harness()
files0, line0 = inv_edits("no_evidence", base)
edit(f"tree = v1 base       → files {sorted(files0) or '—'} · always-on line: {line0!r}")
prose = {**base, "tool_guidance.md": base["tool_guidance.md"] + line0 + "\n"}
files1, line1 = inv_edits("no_evidence", prose)
edit(f"tree = v1 + that line → files {sorted(files1) or '—'} · always-on line: {line1!r}")
plugin = {**prose, **files1}
show_text("plugin/require_evidence.ts  (C8)", plugin["plugin/require_evidence.ts"])
note("the kit's stand-in for 'repeated evidence': the line is already in the always-on text (an earlier edit put it "
     "there) and THIS run still sent an empty evidence list — two observations, not one")
table([["v1 base", H.always_on_tokens(base), "—", H.validate_tree(base) == []],
       ["v1 + prose line", H.always_on_tokens(prose), "asks", H.validate_tree(prose) == []],
       ["v1 + prose line + plugin", H.always_on_tokens(plugin), "enforces", H.validate_tree(plugin) == []]],
      ["harness", "always-on ≈tokens", "evidence rule", "valid tree"])
note("the plugin adds 0 always-on tokens: it lives in the tool loop, not in the prompt")

step(3, "The mechanism: how often is the evidence list empty? (F1-02 and F1-05 × 20 seeds, proposals only)")
pick = [task("F1-02"), task("F1-05")]
rows = []
for ex in ("sim-cloud", "sim-local"):
    for name, tree in (("v1 base", base), ("+ prose line", prose), ("+ prose + plugin", plugin)):
        n = empty = blocked = 0
        for t in pick:
            for s in range(20):
                r = execute(tree, t, seed=s, executor=ex)
                p = r.deliverable.get("proposal")
                if isinstance(p, dict):
                    n += 1
                    empty += not p.get("evidence")
                    blocked += bool(r.signals.get("plugin_events"))
        rows.append([ex, name, n, empty, blocked])
table(rows, ["executor", "harness", "proposals", "evidence empty", "plugin blocked + agent fixed"])
note("prose cuts the empty-evidence rate but cannot remove it — the weaker executor skips prose more often. "
     "The hook catches every remaining case")
dim("these rates are the SIM's assumptions (prose followed 90% / 65%), not measurements of a real model")

step(4, "Repeated evidence inside one step: sequential k = 2 on F1-02 (sim-local, seed 19)")
rec = h2step(task("F1-02"), base, k=2, mode="sequential", seed=19, executor="sim-local")
rec.show()
r1 = [sorted(e.files) for c in rec.candidates if c.label == "inv1.r1" for e in c.edits if e.obs == "no_evidence"]
r2 = [sorted(e.files) for c in rec.candidates if c.label == "inv1.r2" for e in c.edits if e.obs == "no_evidence"]
note(f"round 1 (editing v1): no_evidence → {r1[0] if r1 else '—'} · round 2 (editing the round-1 checkpoint, which "
     f"has the line): no_evidence → {r2[0] if r2 else '—'}")
ok("the plugin appears only in round 2, after the prose line was in place and the failure came back") \
    if r1 and r2 and "plugin/require_evidence.ts" in r2[0] and "plugin/require_evidence.ts" not in r1[0] else \
    warn("the escalation did not happen as described — read the trajectory")

step(5, "'It must not block valid completion': the hook on deliverables that are not proposals")
rows = []
for tid in ("F1-00", "F2-00", "F3-02", "F7-01"):
    t = task(tid)
    r = execute(plugin, t, seed=0)
    rows.append([tid, t.family, r.delivered, type(r.deliverable.get("proposal")).__name__ if "proposal" in r.deliverable
                 else "—", len(r.signals.get("plugin_events", []))])
table(rows, ["task", "family", "delivered", "proposal type", "plugin events"])
ok("no non-proposal deliverable was blocked") if all(r[2] and r[4] == 0 for r in rows) else \
    warn("the plugin fired on a deliverable that was not a proposal")
note("in Ops Assistant's EXAMPLE design this hook is the send_to_user auto-review — it still never replaces the "
     "named-human approval, which stays in platform code (Module 01)")
result("prose first (+25 always-on tokens, asks) → repeated evidence → plugin (+0 always-on, enforces) · "
       f"final tree of the sequential step: {', '.join(c for c in H.changed_components(base, rec.final_tree))}")
