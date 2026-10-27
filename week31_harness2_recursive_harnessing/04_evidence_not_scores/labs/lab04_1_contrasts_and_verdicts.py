#!/usr/bin/env python3
"""Lab 04-1 · Contrasts, verdicts and repeats: what the loop may conclude from one task without a score.

A behavioural contrast says what a probe did differently from the reference (mitigated · unchanged · regressed).
A verdict says what an edit did (helped · hurt · no observed effect · confounded). Neither is correctness.
This lab runs one Propose–Probe–Compose step on F1-01 with one probe per candidate and with three, and shows
the verdicts move in both directions. Then it shows Candidate.noise(): a failure that flipped on in a probe
although no edit touched it. Scores appear once, measured after the step by bench.judge, to show they are a
different question. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/04_evidence_not_scores/labs/lab04_1_contrasts_and_verdicts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                    # noqa: E402
from bench import checks, judge                                        # noqa: E402
from ops_sim import execute, task                                  # noqa: E402
from h2kit import STOCHASTIC, contrast, observe, step                  # noqa: E402
from h2out import banner, dim, note, ok, result, step as stage, table, warn  # noqa: E402

banner("Lab 04-1 · Contrasts, verdicts and repeats", "evidence of behaviour, not of correctness", brain=False)
dim("SIM executor and SIM editors = teaching models (common/ops_sim.py, common/h2kit.py)")
base = H.base_harness()
t = task("F1-01")

stage(1, "A contrast compares two trajectories — it never asks whether the answer is right")
rec = step(t, base, seed=1, repeats=1)
ref = rec.refs[0]
print(f"│ task {t.id}: {t.request}")
print(f"│ reference run (base harness) observed: {', '.join(o.kind for o in observe(ref, t, human_reason=t.human_reason))}")
for c in rec.candidates:
    for i, (p, con) in enumerate(zip(c.probes, c.contrasts)):
        print(f"│ probe {c.label}#{i} vs reference: " + ", ".join(f"{k} {v}" for k, v in sorted(con.items())))
note("mitigated = the failure was in the reference and is gone in the probe · unchanged = still there · "
     "regressed = new in the probe")
rows = []
for name, run in [("reference", ref)] + [(f"probe {c.label}#0", c.probes[0]) for c in rec.candidates if c.probes]:
    failed = [n for n, v in checks(t, run) if not v]
    rows.append([name, judge(t, run), ", ".join(failed) or "—"])
table(rows, ["run", "score*", "failed checks*"], maxw=44)
note("* measured AFTER the step by bench.judge from hidden truth. The contrast above was computed without it.")
p0 = rec.candidates[0].probes[0]
dim(f"inv1's probe mitigated every failure the editor could see — even no_analytic_move, which no edit "
    f"targeted (chance) — and still failed: {', '.join(n for n, v in checks(t, p0) if not v)}. "
    "'mitigated' is a claim about behaviour, not about a correct deliverable")

stage(2, "Four verdicts, four real edit rows (sim-cloud and sim-local, one probe each)")
picks = [("F1-01", "sim-cloud", 1, "scripts/energy_units.py"), ("F1-01", "sim-cloud", 1, "tool_guidance.md"),
         ("F1-02", "sim-local", 0, "skills/chiller_fault_review"), ("F2-01", "sim-cloud", 2, "skills/engineer_digest")]
rows = []
for tid, ex, seed, comp in picks:
    rr = step(task(tid), base, seed=seed, executor=ex, editor=ex + "-editor")
    row = next(x for x in rr.rows if x["component"].startswith(comp))
    cand = next(c for c in rr.candidates if c.label == row["candidates"])
    rows.append([f"{tid} {ex} s{seed}", row["component"][:40], row["observed"],
                 "; ".join(", ".join(f"{k} {v}" for k, v in sorted(con.items()) if k != row["obs"]) for con in cand.contrasts) or "—",
                 row["verdict"]])
table(rows, ["step", "edit row (component)", "its own failure", "other kinds in the probe", "verdict"], maxw=44)
note("the kit's rules (h2kit.edit_rows): hurt = the candidate regressed · no observed effect = its failure never "
     "went away · confounded = several failures edited at once on one rollout with no script or plugin trace, or "
     "fewer than 2 of 3 repeats mitigated · helped = otherwise")
dim("the repository's rule is stricter than the kit: a single rollout behind a difference is already "
    "'confounded' there. Step 3 shows why that is the wiser default.")
warn("the 'hurt' row is no_reopen flipping on in a digest probe — the digest skill never touches reopening. In "
     "this SIM no_reopen is a chance roll that the kit counts as an edit's regression: a mis-attribution the "
     "record keeps (Module 07 calls this attribution noise)")

stage(3, "Repeated executions: one probe per candidate (repeats=1) versus three (repeats=3)")
rows = []
for seed in (1, 0):
    for reps in (1, 3):
        r = step(t, base, seed=seed, repeats=reps)          # repeats = probe executions per candidate
        for row in r.rows:
            rows.append([f"seed {seed}", reps, row["component"][:34], row["observed"], row["verdict"],
                         r.cost["executions"]])
table(rows, ["step", "repeats", "edit row (component)", "observed in each probe", "verdict", "executions"], maxw=44)
ok("seed 1: the no_cmms line was 'confounded' on one probe and 'helped' when three probes all mitigated it")
warn("seed 0: one probe said 'helped' twice; three probes showed 'mitigated, unchanged, mitigated' — the single "
     "rollout had over-claimed, and both rows fall back to 'confounded'")
note("repeats do not only confirm: they also withdraw confidence. They cost executions (see the last column).")

stage(4, "Candidate.noise(): a failure no edit touched that flipped on in a probe")
t2 = task("F1-02")
r2 = step(t2, base, seed=0, repeats=1)
for c in r2.candidates:
    print(f"│ {c.label}: edits target {sorted({e.obs for e in c.edits})} · contrast "
          + "; ".join(", ".join(f"{k} {v}" for k, v in sorted(con.items())) for con in c.contrasts) + " · "
          f"regressed() {sorted(c.regressed()) or '—'} · noise() {sorted(c.noise()) or '—'}")
for n in r2.notes:
    if "variance" in n:
        print(f"→ compose: {n}")
dim(f"failure kinds the kit treats as chance (h2kit.STOCHASTIC): {', '.join(sorted(STOCHASTIC))}")
note("noise() keeps a chance flip from vetoing an edit; repeated executions are how you tell it from a real "
     "regression")
result("a contrast is behaviour, a verdict is attribution, a score is correctness — the loop only ever has the first two")
