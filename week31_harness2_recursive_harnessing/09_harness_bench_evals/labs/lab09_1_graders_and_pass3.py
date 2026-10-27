#!/usr/bin/env python3
"""Lab 09-1 · The Harness Bench: two frozen halves, three grader types, and pass^3.

The course's evaluation suite (EXAMPLE design) on the SIM agent. The refine half may enter the
nightly loop; the test half never does. Code graders check the product (and, for safety only, the transcript).
A SIM model-based judge grades one dimension at a time and may answer "Unknown"; it is accepted only if it agrees
with the "human" labels at least as well as two humans agree with each other. Then pass@k versus pass^3.
No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/09_harness_bench_evals/labs/lab09_1_graders_and_pass3.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchkit as B                                                           # noqa: E402
import harness as H                                                            # noqa: E402
from bench import checks, judge, mean_sd, pass_hat_k, pass_k                   # noqa: E402
from ops_sim import execute, task, tasks                                   # noqa: E402
from h2out import banner, dim, note, ok, result, step, table, warn             # noqa: E402

banner("Lab 09-1 · Graders and pass^3", "refine/test halves · code, model and human graders · pass@k vs pass^k", brain=False)
dim("EXAMPLE design for this course, on the SIM agent; 'human' labels are the SIM's hidden truth standing in for chief engineers")
v1 = H.base_harness()

step(1, "Two frozen halves: the loop may touch refine, never test")
refine, test = tasks("refine"), tasks("test")
rows = [[f, name, sum(B.fam(t) == f for t in refine), sum(B.fam(t) == f for t in test)] for f, name in B.FAMILIES.items()]
table(rows + [["", "total", len(refine), len(test)]], ["family", "what", "refine", "test"])
ok(f"no task id in both halves: {not set(t.id for t in refine) & set(t.id for t in test)} · an EXAMPLE start size is "
   "20 / 20 for F1; the SIM has 8 / 8")

step(2, "Code graders: grade the product; the safety graders are the one legitimate transcript check")
picks = ["F1-21", "F2-20", "F3-20", "F7-20"]
rows = []
for tid in picks:
    t = task(tid)
    r = execute(v1, t, seed=1)
    cs = checks(t, r)
    sg = B.safety_graders(t, r)
    rows.append([tid, t.family, ", ".join(n for n, okv in cs if not okv) or "—",
                 ", ".join(g for g, okv in sg.items() if not okv) or "all pass", judge(t, r)])
table(rows, ["task", "family", "failed product checks", "safety graders", "score*"], maxw=48)
note("* bench.judge = share of checks passed, measured AFTER the run; the safety graders are binary and blocking")

step(3, "The SIM model-based judge: one isolated judge per dimension, allowed to say Unknown")
careful = B.SimJudge("careful judge", err=0.02, unknown=0.05)
t = task("F1-21")
r = execute(v1, t, seed=1)
ji = B.judge_input(t, r)
note(f"judge input keys: {sorted(ji)} — no harness, no EXPERIENCE.md, no human reason")
ok("the reviewer's reason is absent from the judge's input") if t.human_reason not in str(ji) else warn("leak!")
for trial in range(3):
    print(f"│ {t.id} trial view {trial}: {careful.grade(t, r, trial)} → verdict {careful.verdict(t, r, trial)}")
note("an Unknown dimension routes the item to a human; it is never scored as a pass or a fail")

step(4, "Calibration: accept the judge only if it agrees with the humans as well as the humans agree")
items = B.calibration_set(v1)
rows = []
for j in (careful, B.SimJudge("sloppy judge", err=0.12, unknown=0.05)):
    c = B.calibrate(j, items)
    rows.append([j.name, c["n"], c["unknown"], f"{c['judge_vs_A']:.1%}", f"{c['B_vs_A']:.1%}",
                 "ACCEPT" if c["judge_vs_A"] >= c["B_vs_A"] else "REJECT"])
table(rows, ["judge", "items", "Unknown → human", "judge vs engineer A", "engineer B vs A (floor)", "decision"])
note(f"calibration set = refine-half deliverables, both executors, 3 seeds; the test half is never used to tune a grader")

step(5, "A judge that fails is missing, not zero (the spec.py rule)")
fl = {"F1-21", "F2-20"}
sc = [judge(x, execute(v1, x, seed=1), flaky=fl) for x in test]
kept = [s for s in sc if s is not None]
as_zero = [0 if s is None else s for s in sc]
print(f"│ {sum(s is None for s in sc)} failed judge calls · mean of the scored {len(kept)}: {statistics.mean(kept):.1f} · "
      f"if counted as zero: {statistics.mean(as_zero):.1f}")
retry = [judge(x, execute(v1, x, seed=1), flaky=fl, attempt=1) for x in test]
ok(f"re-run the failed calls (attempt 1): every task scored · mean {statistics.mean(retry):.1f}")

step(6, "pass@k versus pass^k: one success is enough, or every trial must succeed")
p = 0.75
note(f"worked example: p = {p} per trial → pass@3 = {pass_k(p, 3):.0%} · pass^3 = {pass_hat_k(p, 3):.0%}")
ts = B.trial_scores(v1, test, seed=0)
rows = []
for f, name in B.FAMILIES.items():
    res = [[x >= B.PASS for x in v] for k, v in ts["per_task"].items() if B.fam(k) == f]
    pr = sum(sum(x) for x in res) / sum(len(x) for x in res)
    emp_k = sum(any(x) for x in res) / len(res)
    emp_hat = sum(all(x) for x in res) / len(res)
    rows.append([f, name, f"{pr:.2f}", f"{emp_k:.2f}", f"{emp_hat:.2f}", f"{pass_hat_k(pr, 3):.2f}"])
table(rows, ["family", "what", "per-trial pass", "pass@3 (any)", "pass^3 (all)", "p³ model"])
warn("pass@3 flatters a customer-facing family: one good digest in three is not a digest the chief engineer can trust")
m, sd = mean_sd([B.family_view(B.trial_scores(v1, test, seed=s))["all"]["mean"] for s in range(3)])
result(f"v1 on the test half: {m} ± {sd} over three seeds (measured after the runs by bench.judge) · promote "
       "customer-facing families on pass^3, report pass@k only as a diagnostic")
