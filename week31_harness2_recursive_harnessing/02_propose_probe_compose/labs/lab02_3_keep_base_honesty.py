#!/usr/bin/env python3
"""Lab 02-3 · KEEP_BASE honesty: when the evidence argues for no change, the loop changes nothing.

Runs one k = 1 step on every quiet task (ops_sim.quiet: no trap, the base harness handles it) and every trap
task of the refine half, for three seeds and both executors, and reports how often the step answered KEEP_BASE.
Then it forces an unsupported edit onto a quiet task to show what the composer does with it, and what the edit
would have cost. Scores appear only at the end, measured after the steps by bench.judge. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/labs/lab02_3_keep_base_honesty.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                         # noqa: E402
from bench import judge                                                     # noqa: E402
from ops_sim import execute, quiet, task, tasks                         # noqa: E402
from h2kit import DOM_EDITS, Candidate, Edit, apply, compose, contrast, observe, step as h2step  # noqa: E402
from h2out import banner, dim, note, ok, result, show_text, step, table, warn  # noqa: E402

# The rule, verbatim from the repository's prompts (common/sources/harness2_prompts.py.txt).
QUOTE_KEEP_BASE = ("KEEP_BASE is a real answer, not a failure. Adaptation that the evidence does not argue for is a "
                   "regression: it costs context and buys nothing.")
QUOTE_SURFACE_KEEP_BASE = ("If the evidence does not argue for any change, leave everything as it is and say KEEP_BASE "
                           "in your notes. That is a real answer, not a failure: adaptation the evidence does not support "
                           "costs context and buys nothing.")

banner("Lab 02-3 · KEEP_BASE honesty", "quiet tasks vs trap tasks · k = 1 · three seeds · two executors", brain=False)
dim("SIM executor and SIM editors = deterministic teaching models; quiet() is kit metadata the loop never reads")

step(1, "The rule, as the repository's composer and editors are told it")
show_text("harness2/core/prompts.py · _KEEP_BASE", QUOTE_KEEP_BASE)
show_text("harness2/core/prompts.py · # HOW TO MAKE A CHANGE", QUOTE_SURFACE_KEEP_BASE)

refine = tasks("refine")
QUIET = [t for t in refine if quiet(t)]
TRAP = [t for t in refine if not quiet(t)]
print(f"│ quiet ({len(QUIET)}): {', '.join(t.id for t in QUIET)}")
print(f"│ trap  ({len(TRAP)}): {', '.join(t.id for t in TRAP)}")

step(2, "One step on every quiet and every trap task: how often is the answer KEEP_BASE?")
base = H.base_harness()
rows, recs = [], {}
for ex in ("sim-cloud", "sim-local"):
    for s in range(3):
        kb = {"quiet": 0, "trap": 0}
        for t in refine:
            rec = h2step(t, base, k=1, seed=s, executor=ex, editor=f"{ex}-editor")
            recs[(ex, s, t.id)] = rec
            kb["quiet" if quiet(t) else "trap"] += rec.keep_base
        rows.append([ex, s, f"{kb['quiet']}/{len(QUIET)}", f"{kb['trap']}/{len(TRAP)}"])
table(rows, ["executor (+ its editor)", "seed", "KEEP_BASE on quiet", "KEEP_BASE on trap"])
qk = sum(recs[k].keep_base for k in recs if quiet(task(k[2])))
tk = sum(recs[k].keep_base for k in recs if not quiet(task(k[2])))
nq, nt = len(QUIET) * 6, len(TRAP) * 6
note(f"KEEP_BASE rate: quiet {qk}/{nq} = {100 * qk / nq:.0f}% · trap {tk}/{nt} = {100 * tk / nt:.0f}%")
ok("on quiet tasks the loop never edits: no observation, no proposal, no new version") if qk == nq else \
    warn("a quiet task was edited — read its notes")

step(3, "Why does a TRAP task sometimes get KEEP_BASE? Read the reference, not the label")
rows = []
for (ex, s, tid), rec in recs.items():
    if ex != "sim-cloud" or s != 0 or quiet(task(tid)) or not rec.keep_base:
        continue
    t = task(tid)
    obs = [o.kind for o in observe(rec.refs[0], t, human_reason=t.human_reason)]
    proposed = sum(len(c.edits) for c in rec.candidates)
    rows.append([tid, ", ".join(t.traps), rec.refs[0].signals.get("cmms_checked"), ", ".join(obs) or "nothing wrong",
                 proposed, rec.notes[-1][:44]])
table(rows, ["task", "trap (kit-only)", "cmms checked", "reference observed", "edits", "composer"])
note("in these reference runs the agent happened to call cmms.work_orders (sim-cloud does so 45% of the time on "
     "v1): the trajectory shows no failure, so no editor has anything to fix")
why = {}
for (ex, s, tid), rec in recs.items():
    if quiet(task(tid)) or not rec.keep_base:
        continue
    obs = observe(rec.refs[0], task(tid), human_reason=task(tid).human_reason)
    n = sum(len(c.edits) for c in rec.candidates)
    r = "reference showed nothing" if not obs else "editors proposed nothing" if not n else "edits proposed, none supported"
    why.setdefault(r, {"sim-cloud": 0, "sim-local": 0})[ex] += 1
table([[r, v["sim-cloud"], v["sim-local"]] for r, v in why.items()], ["why a trap task got KEEP_BASE (3 seeds)", "sim-cloud", "sim-local"])
note("'editors proposed nothing' = the editor missed what the trajectory showed (sim-local-editor misses 40%, "
     "sim-cloud-editor 25%) · 'none supported' = a probe changed nothing or regressed something, so the composer dropped it")
note("KEEP_BASE answers 'did THIS evidence argue for a change?' — not 'is the harness perfect?'. The weakness is "
     "still there; more references (k > 1) or repeated executions would surface it (Modules 04 and 06)")

step(4, "What if the loop edited anyway? Force a DOM skill onto quiet task F1-10 and let the composer judge it")
tq = task("F1-10")
ref = execute(base, tq, seed=0)
files, line = DOM_EDITS["no_analytic_move"](tq)
e = Edit("no_analytic_move", "dom", {**files, "systemprompt.md": line}, "forced, no observation behind it")
c = Candidate("dom1", "dom", apply(base, [e]), [e])
probe = execute(c.tree, tq, seed=1000)
c.probes.append(probe)
c.contrasts.append(contrast(ref, probe, tq))
note(f"reference observed: {', '.join(o.kind for o in observe(ref, tq)) or 'nothing wrong'} · probe contrast: "
     + (", ".join(f"{k} {v}" for k, v in c.contrasts[0].items()) or "no difference"))
tree, notes, kb = compose(base, [c])
for n in notes:
    print(f"→ compose: {n}")
table([["KEEP_BASE (composer)", H.always_on_tokens(tree), H.prefix_hash(tree), len(tree)],
       ["if the forced edit had shipped", H.always_on_tokens(c.tree), H.prefix_hash(c.tree), len(c.tree)]],
      ["outcome", "always-on ≈tokens", "prefix hash", "files"])
ok("the composer dropped it: an edit whose probe changed nothing buys nothing, and would still cost "
   f"+{H.always_on_tokens(c.tree) - H.always_on_tokens(base)} always-on tokens and one cold prompt cache") if kb else \
    warn("the forced edit was merged?!")

step(5, "Measured AFTER the steps by bench.judge (the loop never saw these)")
rows = []
for name, group in (("quiet", QUIET), ("trap", TRAP)):
    for ex in ("sim-cloud", "sim-local"):
        kept = [recs[(ex, s, t.id)] for s in range(3) for t in group if recs[(ex, s, t.id)].keep_base]
        edited = [recs[(ex, s, t.id)] for s in range(3) for t in group if not recs[(ex, s, t.id)].keep_base]
        avg = lambda rs: f"{sum(judge(r.task, r.final_run) for r in rs) / len(rs):.1f}" if rs else "—"   # noqa: E731
        rows.append([name, ex, len(kept), avg(kept), len(edited), avg(edited)])
table(rows, ["tasks", "executor", "KEEP_BASE steps", "mean score*", "composed steps", "mean score*"])
note("* measured after the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it")
dim("SIM numbers show the mechanism (no evidence → no edit); they are not a measurement of Harness² or of a production agent")
result(f"KEEP_BASE on quiet {qk}/{nq} · on trap {tk}/{nt} · a forced edit on a quiet task was "
       + ("dropped by the composer" if kb else "merged"))
