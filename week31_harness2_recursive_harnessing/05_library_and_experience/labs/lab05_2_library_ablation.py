#!/usr/bin/env python3
"""Lab 05-2 · Does the library help? One library per domain and executor, and the ablation read against the SD.

Runs bench.stream once with the library on (root under .runs/) and lists the libraries it created: one per task
family, one per executor. Then runs bench.streams — three streams, each task order shuffled — with the library
on and off, for both executors with their own editors, and reads the difference against the standard deviation
and stream by stream. Scores are measured after each step by bench.judge. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/05_library_and_experience/labs/lab05_2_library_ablation.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from bench import stream, streams                                       # noqa: E402
from ops_sim import tasks                                           # noqa: E402
from h2out import banner, bar, dim, note, result, runs_dir, step, table, warn  # noqa: E402

QUOTE_SEPARATE = "Libraries are separated by benchmark, executor, mode, budget, and seed."
QUOTE_SD = "SDs are 1–3 points: believe differences under ~3 only if all three streams agree."

banner("Lab 05-2 · Does the library help?", "one library per domain × executor · the ablation against the SD", brain=False)
dim("SIM executors and editors = teaching models; scores measured AFTER each step by bench.judge")
root = runs_dir(__file__) / "lab05_2_streams"
shutil.rmtree(root, ignore_errors=True)
refine = tasks("refine")

step(1, "One stream per executor with the library on — which libraries exist afterwards?")
for ex in ("sim-cloud", "sim-local"):
    stream(refine, seed=0, executor=ex, editor=ex + "-editor", library_on=True, root=root / "one")
libs = sorted((root / "one" / "stream0").iterdir())
rows = []
for d in libs:
    n = len(list((d / "harnesses").iterdir()))
    entries = sum(1 for e in (d / "harnesses").iterdir() if "NO_ENTRY" not in (e / "EXPERIENCE.md").read_text())
    rows.append([d.name.replace("__", "/"), n, entries])
table(rows, ["library (agent/family/tenant class/executor)", "tasks stored", "with a lesson"])
print(f"│ EXPERIMENTS.md: “{QUOTE_SEPARATE}”")
note("a sim-local task never reads a sim-cloud lesson: transfer between executors is measured, not assumed (Module 06)")

step(2, "The ablation: library on vs off, three shuffled streams, both executors")
res = {}
for ex in ("sim-cloud", "sim-local"):
    for on in (True, False):
        res[ex, on] = streams(refine, n=3, executor=ex, editor=ex + "-editor", library_on=on, root=root / f"abl_{ex}_{on}")
rows = []
for (ex, on), r in res.items():
    rows.append([ex, "on" if on else "off", " / ".join(f"{m:.1f}" for m in r["means"]), f"{r['mean']} ± {r['sd']}",
                 bar(r["mean"], 100, 20)])
table(rows, ["executor", "library", "stream means (s0 / s1 / s2)", "mean ± SD", ""])

step(3, "Read it honestly: the difference, the SD, and whether all three streams agree")
print(f"│ facilitator guide: “{QUOTE_SD}”")
rows = []
for ex in ("sim-cloud", "sim-local"):
    on, off = res[ex, True], res[ex, False]
    deltas = [round(a - b, 1) for a, b in zip(on["means"], off["means"])]
    agree = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    diff = round(on["mean"] - off["mean"], 1)
    rows.append([ex, f"{diff:+}", f"{on['sd']} / {off['sd']}", " / ".join(f"{d:+}" for d in deltas),
                 "yes" if agree else "no", "within one SD" if abs(diff) < min(on["sd"], off["sd"]) else "beyond one SD"])
table(rows, ["executor", "on − off", "SD on / off", "per stream (same order)", "all agree", "size"])
for r in rows:
    if r[4] == "no":
        warn(f"{r[0]}: the streams disagree in sign ({r[3]}) — no library effect is shown on 17 tasks")
    else:
        note(f"{r[0]}: all three streams agree ({r[3]}), but the gap of {r[1]} is {r[5]} on 17 tasks — "
             "a hint to be tested, not a result")
note("neither difference is a result you could act on. Module 06 reads tables like this; Module 09 pre-registers the test.")
dim("reported in the paper: without the library the method loses 3.1 points on "
    "WorkBuddy-Bench and 1.8 on JobBench. The SIM shows the mechanism, not that number.")
result("one library per domain and per executor; and a library effect is a claim you test, not one you assume")
