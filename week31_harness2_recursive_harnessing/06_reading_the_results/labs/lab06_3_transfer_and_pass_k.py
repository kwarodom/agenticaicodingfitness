#!/usr/bin/env python3
"""Lab 06-3 · Transfer between executors, and why pass^3 is the number that promotes a version.

Prints the paper's transfer numbers (as reported in the paper, arXiv 2610.14224). Then builds two
harness VERSIONS with the reviewer's merge (bench.promote, EXAMPLE): one from sim-cloud stream records, one from
sim-local stream records. Each version is frozen and evaluated on the 17-task TEST half — which never entered
the loop — on BOTH executors, three trials per task, under three evaluation seeds (mean ± SD). Finally it puts
pass@k next to pass^k: the formula for independent trials, and what the test half actually shows.

    .venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/labs/lab06_3_transfer_and_pass_k.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import evaluate, pass_hat_k, pass_k, promote, streams      # noqa: E402
from ops_sim import tasks                                         # noqa: E402
from h2out import banner, dim, note, ok, result, step, table, warn    # noqa: E402
from readkit import reading                                           # noqa: E402

PASS_K_ADVICE = "pass@k fits a tool where one success is enough; pass^k fits an agent that must be consistent"

banner("Lab 06-3 · Transfer and pass^k", "a version refined on one executor, evaluated on both · pass@k vs pass^k", brain=False)
dim("SIM numbers are a teaching model of the method (common/), not measurements of Harness², a real model or a "
    "production agent")

step(1, "What the paper reports on transfer (as reported in the paper, arXiv 2610.14224)")
table([
    ["GLM-5", "Code", "+17.3 (58.5 → 75.8)"],
    ["Gemini 3.7 Flash", "Code · Office", "+0.3 · +6.4"],
    ["Opus 4.8", "two domains", "+2.6 · +1.1"],
], ["executor given the Gemini-3.5-Flash-refined harness", "domain", "gain"])
note("the paper reports that the benefit varies a lot with the executor and the domain — so the course measures "
     "transfer per direction (pre-registered arm E, Module 09) instead of assuming it")

base = H.base_harness()
R, T = tasks("refine"), tasks("test")
step(2, "Promote one version per executor from its own stream records (reviewer's merge, min_support=2, EXAMPLE)")
versions = {"base v1": base}
for ex in ("sim-cloud", "sim-local"):
    s = streams(R, n=3, mode="harness2", executor=ex, editor="sim-cloud-editor")
    recs = [rec for run in s["runs"] for rec in run["records"]]
    v, notes = promote(base, recs, min_support=2)
    versions[f"refined on {ex}"] = v
    adopted = [n for n in notes if n.startswith("adopt")]
    print(f"│ refined on {ex}: {len(recs)} step records → {len(adopted)} adopted file edits · {len(v)} files · "
          f"always-on {H.always_on_tokens(base)} → {H.always_on_tokens(v)} tokens · new components "
          f"{', '.join(H.changed_components(base, v))}")
note("min_support=2: an edit enters the version only if at least two composed tasks adopted it with a 'helped' row "
     "— one task's outcome does not establish transfer")

step(3, f"Evaluate every version on the frozen TEST half ({len(T)} tasks × 3 trials) on both executors, 3 seeds")
E = {}
for name, v in versions.items():
    for ex in ("sim-cloud", "sim-local"):
        E[name, ex] = [evaluate(v, T, trials=3, executor=ex, seed=sd) for sd in range(3)]
rows = []
for ex in ("sim-cloud", "sim-local"):
    b = [e["mean"] for e in E["base v1", ex]]
    for name in versions:
        m = [e["mean"] for e in E[name, ex]]
        r = reading(b, m) if name != "base v1" else None
        rows.append([ex, name, " / ".join(f"{x:.1f}" for x in m),
                     f"{statistics.mean(m):.1f} ± {statistics.stdev(m):.1f}",
                     "—" if r is None else f"{r['delta']:+.1f} · {r['verdict']}",
                     f"{statistics.mean(e['pass_hat_3'] for e in E[name, ex]):.2f}",
                     sum(e["unsafe_trials"] for e in E[name, ex])])
table(rows, ["executor", "version", "seed 0 / 1 / 2", "mean ± SD*", "Δ vs base", "pass^3*", "unsafe"])
note("* measured on the test half by bench.evaluate; nothing in the loop ever read these tasks")

step(4, "Read transfer per direction")
for src, dst in (("sim-cloud", "sim-local"), ("sim-local", "sim-cloud")):
    own = [e["mean"] for e in E[f"refined on {dst}", dst]]
    moved = [e["mean"] for e in E[f"refined on {src}", dst]]
    r = reading(own, moved)
    print(f"│ {src} version on {dst}: {statistics.mean(moved):.1f} vs {dst}'s own version {statistics.mean(own):.1f} "
          f"→ {r['delta']:+.1f} · {r['verdict']}")
warn("in the SIM both executors share the same weaknesses and the same editors, so a version moves almost intact. "
     "The SIM cannot reproduce the paper's uneven transfer (+17.3 vs +0.3); it can only show how to measure it")

step(5, "pass@k versus pass^k")
print(f"◆ Anthropic's evals guide, in short: {PASS_K_ADVICE}")
p = 0.75
table([[f"{p:.2f}", 3, f"{pass_k(p, 3):.1%}", f"{pass_hat_k(p, 3):.1%}"]],
      ["per-trial success p", "k", "pass@k = 1 − (1 − p)^k", "pass^k = p^k"])
note("worked example: 75% per trial is about 42% across three trials")
rows = []
for name in versions:
    for ex in ("sim-cloud", "sim-local"):
        e = E[name, ex][0]
        rows.append([name, ex, f"{e['pass_rate']:.2f}", f"{pass_k(e['pass_rate'], 3):.2f}",
                     f"{pass_hat_k(e['pass_rate'], 3):.2f}", f"{e['pass_hat_3']:.2f}"])
table(rows, ["version", "executor", "per-trial p", "pass@3 if independent", "p^3 if independent",
             "pass^3 observed"])
above = sum(E[name, ex][0]["pass_hat_3"] >= round(pass_hat_k(E[name, ex][0]["pass_rate"], 3), 2)
            for name in versions for ex in ("sim-cloud", "sim-local"))
note(f"observed pass^3 is not p^3 (at or above it on {above} of {2 * len(versions)} rows): trials of one task are not "
     "independent — some tasks pass every time, some never. Measure pass^3 per task; do not derive "
     "it from the pass rate")
ok("a customer-facing family is promoted on pass^3 (all three mornings right), never on pass@3 (one of three right)")
result("transfer is a measurement per direction, and reliability is the share of tasks that pass every trial")
