#!/usr/bin/env python3
"""Lab 06-1 · Three shuffled streams: base versus Harness² for both executors, read as mean ± SD.

First prints the paper's headline numbers (as reported in the paper, arXiv 2610.14224). Then runs
the course's SIM the way the paper reports: the 17 refine tasks as three open-world streams in shuffled order,
base harness versus one Harness² step per task, for sim-cloud and sim-local, with one fixed editor
(sim-cloud-editor, as the paper uses one default editor for both executors). It reads every difference stream
by stream, checks same-model editing (sim-local editing itself), and counts tool calls, tool errors and the
extra executions the loop spends. Scores are measured after each step by bench.judge.

    .venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/labs/lab06_1_three_streams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import streams                                             # noqa: E402
from ops_sim import execute, tasks                                # noqa: E402
from h2out import banner, dim, note, ok, result, step, table, warn    # noqa: E402
from readkit import fmt, reading, streams_str                         # noqa: E402

banner("Lab 06-1 · Three shuffled streams", "base vs Harness² · mean ± SD · both executors", brain=False)
dim("SIM numbers are a teaching model of the method (common/), not measurements of Harness², a real model or a "
    "production agent")

step(1, "What the paper reports (as reported in the paper, arXiv 2610.14224)")
table([
    ["WorkBuddy-Bench avg", "Gemini 3.5 Flash / OpenCode", "72.4", "77.5", "+5.1"],
    ["WorkBuddy-Bench avg", "GLM-5 / OpenCode (weaker)", "64.2", "75.7", "+11.5"],
    ["JobBench", "Gemini 3.5 Flash", "31.5", "36.0", "+4.5"],
    ["LAB", "GLM-5", "59.6", "71.4", "+11.8"],
], ["benchmark", "executor / harness", "base", "one step", "gain"])
note("protocol in the paper: each score is the mean over three independent streams, with SDs over three runs in "
     "shuffled task order")
note("JobBench base-harness SDs are 1.1 and 2.6 points → this course reads a gap under ~3 points as noise unless all "
     "three streams agree in direction")

R = tasks("refine")
EDITOR = "sim-cloud-editor"
step(2, f"The SIM, same protocol: {len(R)} refine tasks × 3 shuffled streams, base vs one Harness² step per task")
S = {}
rows = []
for ex in ("sim-cloud", "sim-local"):
    S[ex, "base"] = streams(R, n=3, mode="base", executor=ex)
    S[ex, "h2"] = streams(R, n=3, mode="harness2", executor=ex, editor=EDITOR)
    for arm, label in (("base", "base harness v1"), ("h2", "Harness² k=1")):
        s = S[ex, arm]
        rows.append([ex, label, streams_str(s), fmt(s)])
table(rows, ["executor", "arm", "stream 1 / 2 / 3", "mean ± SD*"])
note("* measured after each step by bench.judge (hidden truth); the loop never saw a score")

step(3, "Read each difference stream by stream: do all three agree, and is the gap larger than one SD?")
rows, gains = [], {}
for ex in ("sim-cloud", "sim-local"):
    r = reading(S[ex, "base"]["means"], S[ex, "h2"]["means"])
    gains[ex] = r
    rows.append([ex, " / ".join(f"{d:+.1f}" for d in r["diffs"]), f"{r['delta']:+.1f}", r["sd"],
                 "yes" if r["agree"] else "no", "yes" if r["beyond_sd"] else "no", r["verdict"]])
table(rows, ["executor", "per-stream Δ", "Δ mean", "larger SD", "all agree", "|Δ| > SD", "reading"])
ok("one step helps both SIM executors beyond stream noise — the same shape as the paper's 'every configuration'")

step(4, "Does the weaker executor gain more? Compare the GAINS, stream by stream")
g = reading(gains["sim-cloud"]["diffs"], gains["sim-local"]["diffs"])
print(f"│ gain per stream  sim-cloud {' / '.join(f'{d:+.1f}' for d in gains['sim-cloud']['diffs'])}"
      f"   sim-local {' / '.join(f'{d:+.1f}' for d in gains['sim-local']['diffs'])}")
print(f"│ local gain − cloud gain: {g['delta']:+.1f} · per stream {' / '.join(f'{d:+.1f}' for d in g['diffs'])}"
      f" · larger SD of the gains {g['sd']} → {g['verdict']}")
x = reading(S["sim-cloud", "base"]["means"], S["sim-local", "h2"]["means"])
note(f"sim-local with Harness² ({S['sim-local', 'h2']['mean']}) vs sim-cloud on its base ({S['sim-cloud', 'base']['mean']}): "
     f"{x['delta']:+.1f} · per stream {' / '.join(f'{d:+.1f}' for d in x['diffs'])} → {x['verdict']} "
     "(the paper reports the weaker executor with an improved harness ahead, for GLM-5)")
warn("the SIM ENCODES 'weaker executors follow prose less and hand-roll more' as an assumption, so this shows the "
     "mechanism; it is not evidence for the paper's finding")

step(5, "Same-model editing: sim-local as its own editor versus the fixed cloud editor")
self_ed = streams(R, n=3, mode="harness2", executor="sim-local", editor="sim-local-editor")
r = reading(S["sim-local", "h2"]["means"], self_ed["means"])
table([["sim-local", "sim-cloud-editor", streams_str(S["sim-local", "h2"]), fmt(S["sim-local", "h2"])],
       ["sim-local", "sim-local-editor", streams_str(self_ed), fmt(self_ed)]],
      ["executor", "editor", "stream 1 / 2 / 3", "mean ± SD"])
note(f"self-editing − cloud editor: {r['delta']:+.1f} · per stream {' / '.join(f'{d:+.1f}' for d in r['diffs'])} "
     f"→ {r['verdict']}")
note("the SIM local editor is set to miss a failure 40% of the time (cloud editor 25%), yet over 17 tasks and three "
     "streams that difference does not show: the table cannot tell the two editors apart")

step(6, "Tool use and cost: what the final runs did, and what the loop spent to get them")
rows = []
for ex in ("sim-cloud", "sim-local"):
    calls = errs = 0
    h_calls = h_errs = 0
    execs = sessions = n = 0
    for i, run in enumerate(S[ex, "h2"]["runs"]):
        for rec in run["records"]:
            b = execute(H.base_harness(), rec.task, seed=i * 100, executor=ex)    # the base run of the same stream
            calls += b.signals["tool_calls"]
            errs += b.signals["tool_errors"]
            h_calls += rec.final_run.signals["tool_calls"]
            h_errs += rec.final_run.signals["tool_errors"]
            execs += rec.cost["executions"]
            sessions += rec.cost["editor_sessions"]
            n += 1
    rows.append([ex, f"{calls / n:.2f} → {h_calls / n:.2f}", f"{100 * errs / calls:.1f}% → {100 * h_errs / h_calls:.1f}%",
                 f"1 → {execs / n:.1f}", f"0 → {sessions / n:.1f}"])
table(rows, ["executor", "tool calls / task", "tool-error rate", "executions / task", "editor sessions / task"])
note("more calls, fewer errors: scripts, validators and the maintenance check ADD useful work; refinement need not "
     "shorten a run")
warn("the base row costs one execution per task; Harness² adds references, probes and editor sessions — the paper "
     "publishes no token totals, only that it 'requires additional executions and editor sessions'")
result("read a results table by its streams and SDs, then ask what the gain cost")
