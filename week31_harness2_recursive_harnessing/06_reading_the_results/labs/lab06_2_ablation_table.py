#!/usr/bin/env python3
"""Lab 06-2 · The ablation table: which parts of the method show up beyond noise, and what each costs.

First prints the paper's ablations and component removals (as reported in the paper, arXiv 2610.14224). Then
builds the SIM's own ablation table for both executors, each arm as three shuffled streams over
the 17 refine tasks: Harness² k=1 (the reference arm), library off, scoping off (one generic editor writing
prose), composition off (adopt one candidate whole), parallel k=3 and k=5, and sequential k=3 and k=5. Every
arm is read against k=1 stream by stream, with its cost (executions and editor sessions per task) and the
always-on tokens of the harnesses it adopted. Scores are measured after each step by bench.judge.

    .venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/labs/lab06_2_ablation_table.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import streams                                             # noqa: E402
from ops_sim import tasks                                         # noqa: E402
from h2out import banner, dim, note, result, step, table, warn        # noqa: E402
from readkit import fmt, reading, streams_str                         # noqa: E402

banner("Lab 06-2 · The ablation table", "library · scoping · composition · k · sequential — beyond noise or not?", brain=False)
dim("SIM numbers are a teaching model of the method (common/), not measurements of Harness², a real model or a "
    "production agent")

step(1, "What the paper reports (as reported in the paper, arXiv 2610.14224)")
table([
    ["without the library", "−3.1", "−1.8"],
    ["without scoping (generic editors)", "−2.9", "−3.7"],
    ["without composition", "−1.6", "−1.9"],
], ["ablation (k = 1)", "WorkBuddy-Bench", "JobBench"])
table([
    ["remove skills", "−5.3 average (6.9 on Web)"],
    ["remove scripts", "−3.7 average (7.4 on Office)"],
    ["remove prompt text", "−1.5"],
], ["from the improved harness", "points lost"])
note("recursion: parallel k 1 → 5 raises Code 76.5 → 80.8 and Office 79.5 → 83.4; the paper reports both modes "
     "improving monotonically")
note("against JobBench base SDs of 1.1 and 2.6, the ablations of 1.6–1.9 are the ones to read most carefully")

R = tasks("refine")
EDITOR = "sim-cloud-editor"
ARMS = [
    ("Harness² k=1", dict()),
    ("no library", dict(library_on=False)),
    ("generic editor", dict(scoped=False)),
    ("no composition", dict(compose_on=False)),
    ("parallel k=3", dict(k=3)),
    ("parallel k=5", dict(k=5)),
    ("sequential k=3", dict(k=3, recursion="sequential")),
    ("sequential k=5", dict(k=5, recursion="sequential")),
]
S = {}
for ex in ("sim-cloud", "sim-local"):
    step(2 if ex == "sim-cloud" else 3, f"The SIM's ablation table on {ex} (3 shuffled streams each, read against k=1)")
    rows = []
    for name, kw in ARMS:
        s = streams(R, n=3, mode="harness2", executor=ex, editor=EDITOR, **kw)
        S[ex, name] = s
        recs = [rec for run in s["runs"] for rec in run["records"]]
        execs = statistics.mean(rec.cost["executions"] for rec in recs)
        sess = statistics.mean(rec.cost["editor_sessions"] for rec in recs)
        aot = statistics.mean(H.always_on_tokens(rec.final_tree) for rec in recs)
        if name == "Harness² k=1":
            rows.append([name, streams_str(s), fmt(s), "—", "—", f"{execs:.1f}", f"{sess:.1f}", f"{aot:.0f}"])
            continue
        r = reading(S[ex, "Harness² k=1"]["means"], s["means"])
        rows.append([name, streams_str(s), fmt(s), f"{r['delta']:+.1f}", r["verdict"], f"{execs:.1f}", f"{sess:.1f}",
                     f"{aot:.0f}"])
    table(rows, ["arm", "stream 1 / 2 / 3", "mean ± SD*", "Δ vs k=1", "reading", "execs/task", "sessions/task",
                 "always-on"])
    note("* measured after each step by bench.judge · always-on = mean always-on tokens of the adopted harnesses "
         f"(base {H.always_on_tokens(H.base_harness())})")

step(4, "Two comparisons the table invites: k=5 against k=3, and sequential against parallel at k=5")
rows = []
for ex in ("sim-cloud", "sim-local"):
    for a, b in (("parallel k=3", "parallel k=5"), ("parallel k=5", "sequential k=5")):
        r = reading(S[ex, a]["means"], S[ex, b]["means"])
        rows.append([ex, f"{b} − {a}", " / ".join(f"{d:+.1f}" for d in r["diffs"]), f"{r['delta']:+.1f}", r["sd"],
                     r["verdict"]])
table(rows, ["executor", "comparison", "per-stream Δ", "Δ mean", "larger SD", "reading"])

step(5, "What an honest reader writes under this table")
for ex in ("sim-cloud", "sim-local"):
    shown = [n for n, _ in ARMS[1:] if reading(S[ex, "Harness² k=1"]["means"], S[ex, n]["means"])["verdict"] == "beyond noise"]
    not_shown = [n for n, _ in ARMS[1:] if n not in shown]
    print(f"│ {ex}: differs from k=1 beyond noise → {', '.join(shown) or 'nothing'}")
    print(f"│ {' ' * len(ex)}  not shown on 17 tasks         → {', '.join(not_shown)}")
for ex in ("sim-cloud", "sim-local"):
    aot = {n: statistics.mean(H.always_on_tokens(rec.final_tree) for run in S[ex, n]["runs"] for rec in run["records"])
           for n in ("Harness² k=1", "generic editor")}
    note(f"{ex}: the generic editor writes always-on prose — {aot['generic editor']:.0f} always-on tokens per adopted "
         f"harness vs {aot['Harness² k=1']:.0f} for the scoped editors at the same k, for a lower mean")
warn("the library and composition effects the paper reports (1.6 to 3.1 points) are not shown beyond noise here: 17 "
     "tasks and three streams cannot resolve them. 'Not shown' is not 'no effect' — and it is not 'shown' either")
warn("k=5 does not beat k=3 in the SIM, unlike the paper's monotone curve. One visible reason: with five finals the "
     "blind selector picks a weaker one more often (Module 07, Lab 07-1 counts it)")
result("an ablation row earns a sentence only when all three streams agree and the gap clears the SD")
