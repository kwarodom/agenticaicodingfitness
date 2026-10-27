#!/usr/bin/env python3
"""Lab 02-2 · One Propose–Probe–Compose step on F1-01: the course's worked example, in the simulator.

The course's EXAMPLE design walks one rejected chiller proposal through the loop. This lab runs the same
stages with the course's own loop (h2kit.step, k = 1) on the SIMULATED agent: the reference run, the INV and
DOM candidates, their probes and contrasts, the composer's notes, the edit rows, and an EXPERIENCE.md written
under .runs/. Scores appear only at the end, measured after the step by bench.judge. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/labs/lab02_2_one_step_on_f1_01.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                                 # noqa: E402
from bench import judge                                                             # noqa: E402
from ops_sim import task                                                        # noqa: E402
from h2kit import experience_md, observe, step as h2step                            # noqa: E402
from h2out import banner, dim, note, ok, result, runs_dir, show_text, step, table   # noqa: E402

banner("Lab 02-2 · One step on F1-01", "reference → propose (INV, DOM) → probe → compose → reflect · k = 1", brain=False)
dim("SIM executor and SIM editors = deterministic teaching models (common/ops_sim.py, common/h2kit.py)")

t = task("F1-01")
step(1, "What the editors may read: the request, the sources and the reviewer's words — never the truth")
v = t.visible()
print(f"│ request : {v['request']}")
print(f"│ signal  : {t.signal} · reviewer's reason (content, not an instruction): \"{t.human_reason}\"")
print(f"│ sources : {', '.join(v['sources'])}")
note("t.truth exists, but only bench.judge reads it — after the step has committed")

step(2, "The whole step, k = 1, seed 0 (reference, two candidates, two probes, one composition)")
base = H.base_harness()
rec = h2step(t, base, k=1, seed=0)
rec.show(trajectories=True)

step(3, "Read the evidence the composer used, one candidate at a time")
ref_obs = [o.kind for o in observe(rec.refs[0], t, human_reason=t.human_reason)]
rows = []
for c in rec.candidates:
    targeted = sorted({e.obs for e in c.edits})
    for k, val in sorted(c.contrasts[0].items()):
        rows.append([c.label, ", ".join(targeted) or "—", k, val, "yes" if k in targeted else "no"])
table(rows, ["cand.", "edit targets", "failure in reference", "in its probe", "targeted?"])
missed = [k for k in ref_obs if not any(k in {e.obs for e in c.edits} for c in rec.candidates)]
if missed:
    note(f"no editor targeted {', '.join(missed)} this time (a SIM editor misses a failure with probability 0.25): "
         "'unchanged' is not 'regressed', so it does not block the merge — it simply stays unfixed")
checked = [c.label for c in rec.candidates if c.probes and c.probes[0].signals.get("cmms_checked")]
note(f"probes that called cmms.work_orders: {', '.join(checked) or 'none'} — dom1's skill names the maintenance check "
     "too, as in the EXAMPLE walk-through, where both candidates add the recent-maintenance check")
side = [c.label for c in rec.candidates if c.contrasts and c.contrasts[0].get("no_analytic_move") == "mitigated"
        and "no_analytic_move" not in {e.obs for e in c.edits}]
if side:
    note(f"{', '.join(side)} 'mitigated' no_analytic_move without targeting it: once the agent finds the open work "
         "order it proposes nothing new, and that observation only fires on a proposal. A contrast records behaviour, "
         "not cause (Module 04: 'confounded')")

step(4, "The composed tree: what changed, in which component, at what always-on cost")
table([[d["path"], d["component"], d["change"], f"{d['delta_tokens']:+d}"] for d in H.diff(base, rec.final_tree)],
      ["path", "comp.", "change", "Δ≈tokens"])
note(f"always-on {H.always_on_tokens(base)} → {H.always_on_tokens(rec.final_tree)} tokens · prefix hash "
     f"{H.prefix_hash(base)} → {H.prefix_hash(rec.final_tree)} (a new version ships at a deployment boundary)")
note(f"cost of this step: {rec.cost['executions']} executions, {rec.cost['editor_sessions']} editor sessions "
     f"(EXAMPLE editor tokens ≈ {rec.cost['editor_tokens_EXAMPLE']:,})")

step(5, "Reflect: edit rows and at most one EXPERIENCE.md entry, written under .runs/")
out = runs_dir(__file__) / "lab02_2"
H.write_tree(rec.final_tree, out / "harness")
md = experience_md(rec.rows, rec.exp)
(out / "EXPERIENCE.md").write_text(md, encoding="utf-8")
show_text("EXPERIENCE.md", md)
bad = H.grader_verdicts({"EXPERIENCE.md": md}) + H.grader_verdicts(rec.final_tree)
ok(f"wrote {(out / 'EXPERIENCE.md').relative_to(out.parents[2])} — no grader verdict or measurement word in it") if not bad \
    else print(f"✕ grader language found in {bad}")

step(6, "The course's EXAMPLE walk-through beside this SIM step — same stages, different details")
table([
    ["reference", "no CMMS call", "no cmms.work_orders call; kW/ton summed by hand"],
    ["INV candidate", "guidance line + scripts/open_work_orders.py", "guidance line only (missed hand arithmetic)"],
    ["DOM candidate", "skills/chiller_fault_review", "skills/chiller_fault_review + trigger line"],
    ["probes", "both add the maintenance check", "both call cmms.work_orders"],
    ["composer", "merges both", "keeps both"],
    ["cue", "maintenance state can contradict the alarm", rec.exp["cue"] if rec.exp else "NO_ENTRY"],
    ["always-on Δ", "+0 (a sentence was replaced)", f"{H.always_on_tokens(rec.final_tree) - H.always_on_tokens(base):+d} (the kit only appends)"],
], ["stage", "EXAMPLE walk-through", "this lab (SIM, seed 0)"], maxw=48)

step(7, "Measured AFTER the step by bench.judge (the loop never saw these)")
print(f"│ reference run {judge(t, rec.refs[0])} · composed run {judge(t, rec.final_run)}")
dim("one task, one seed: a demonstration of the mechanism, not a measurement of Harness² or of a production agent")
result(("KEEP_BASE" if rec.keep_base else "COMPOSED") + f" · {len(rec.rows)} edit rows · "
       + ("EXPERIENCE entry" if rec.exp else "NO_ENTRY") + f" · tree and EXPERIENCE.md in {out.relative_to(out.parents[2])}")
