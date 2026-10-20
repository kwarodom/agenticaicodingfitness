#!/usr/bin/env python3
"""Lab 06-5 · Harness-level benchmark: identical tasks through the claw's API, p50/p95, and the four-layer report.

Research tutorial Part 5, Lab 5.5. OpenClaw and Hermes claws have no NAT profiler, so you time the API each
harness exposes. On the Spark that is Hermes' OpenAI-compatible API on :8642 (read-only loop; DRY → EXAMPLE).
On THIS laptop the same loop runs for real against a `nat serve` of Alto Ops Claw (/v1/chat/completions):
5 identical tasks, one at a time (≈ 10 LLM calls). p50/p95 are computed two ways: interpolated, and with the
research tutorial's `sort | awk` nearest-rank line, which is coarse at small n.
Then the four layers side by side, from the summaries labs 06-1 to 06-4 left in .runs/ (LAPTOP STAND-IN).

Run: .venv/bin/python week26/06_benchmarking/labs/lab06_5_harness_bench.py [--tasks 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import (LAPTOP_OLLAMA, NAT, PORTS, ROOT, background, banner, free_port, note, ok, result, sh, step,  # noqa: E402
                     table, up, warn)
from benchkit import RUNS, awk_percentile, load_summary, nat_serve_env, percentile, save_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", type=int, default=5, help="identical tasks to send (≤ 5 on a shared laptop)")
args = ap.parse_args()
N = max(2, min(args.tasks, 5))

banner("Lab 06-5 · harness benchmark", "identical tasks through the claw's API · p50/p95 · the four-layer report")
TASK = "Summarise the chiller plant's efficiency over the last 6 hours in 3 bullets."

# ── 1 · the Spark: Hermes :8642 ─────────────────────────────────────────────────
step(1, f"the Spark — 20 identical tasks to Hermes' OpenAI-compatible API on :{PORTS['hermes']} (read-only)")
HERMES_LOOP = r"""for i in $(seq 1 20); do
  /usr/bin/time -f "%e" curl -s -X POST http://localhost:8642/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"hermes","messages":[{"role":"user","content":"Summarise /sandbox/data/chiller_plant.csv in 3 bullets"}]}' >/dev/null
done 2>&1 | sort -n | awk '{a[NR]=$1} END {print "p50",a[int(NR*0.5)],"p95",a[int(NR*0.95)]}'"""
sh(HERMES_LOOP, example="p50 … p95 …", timeout=3600)
sh("openshell logs my-hermes --source sandbox | grep -c inspect_for_inference",
   example="…   (one inspect_for_inference per model call: divide by 20 for LLM calls per task)", timeout=60)
note("Pair the loop with the inspect_for_inference count (LLM calls per task) and Phoenix/Langfuse spans "
     "(tokens per task, Module 05). Watch it live with `openshell term`.")

# ── 2 · THIS laptop: the same loop against a nat serve ───────────────────────────
step(2, f"LAPTOP STAND-IN — {N} identical tasks to Alto Ops Claw's /v1/chat/completions (nat serve on this Mac)")
if not up(LAPTOP_OLLAMA):
    warn("laptop Ollama is not answering on :11434 — start it, then re-run")
    sys.exit(0)
port = free_port(PORTS["nat"])
print(f"◆ nat serve port: {port}" + ("" if port == PORTS["nat"] else f" (:{PORTS['nat']} is busy — another lab's server)"))
cfg = "week26/common/alto_ops/src/alto_ops/configs/workflow.laptop.yml"
times, answers = [], []
with background([NAT, "serve", "--config_file", cfg, "--host", "127.0.0.1", "--port", str(port)],
                ready_url=f"http://127.0.0.1:{port}/docs", log=RUNS / "nat_serve.log", env=nat_serve_env(), cwd=ROOT,
                show=f"nat serve --config_file {cfg} --host 127.0.0.1 --port {port}"):
    body = json.dumps({"model": "alto-ops", "messages": [{"role": "user", "content": TASK}], "stream": False}).encode()
    print(f"→ POST http://127.0.0.1:{port}/v1/chat/completions × {N} · one at a time · the same task every time")
    for i in range(N):
        t0 = time.perf_counter()
        with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, method="POST",  # noqa: S310
                             headers={"Content-Type": "application/json"}), timeout=300) as r:
            d = json.loads(r.read().decode("utf-8", errors="replace"))
        dt = time.perf_counter() - t0
        times.append(dt)
        txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        answers.append(txt)
        print(f"◆ task {i + 1}: {dt:.2f} s · {txt.strip().splitlines()[0][:80] if txt.strip() else '(empty)'!r}")

step(3, "p50 / p95 — interpolated vs the tutorial's nearest-rank awk line")
table([["interpolated (benchkit.percentile)", f"{percentile(times, 50):.2f}", f"{percentile(times, 95):.2f}"],
       ["sort | awk a[int(NR*p)]", f"{awk_percentile(times, 50):.2f}", f"{awk_percentile(times, 95):.2f}"]],
      ["method", "p50 s", "p95 s"])
note(f"With n = {N}, awk's p95 is just the {max(1, int(N * 0.95))}th-fastest run: one slow outlier moves it a lot. "
     "The tutorial sends 20 tasks for a reason; on the Spark, send 20 or more.")
same = len({a.strip() for a in answers})
note(f"{same} distinct answer text(s) for {N} identical tasks at temperature 0. Check correctness as well as speed.")
save_summary("harness", {"tasks": N, "p50_s": percentile(times, 50), "p95_s": percentile(times, 95), "times": times})

# ── 4 · the report ───────────────────────────────────────────────────────────────
step(4, "the four layers side by side (all LAPTOP STAND-IN, from .runs/summary_*.json)")
eng, wf, q, sb, hz = (load_summary(k) for k in ("engine", "workflow", "quality", "sandbox", "harness"))
rows = []
if eng:
    s1 = eng["sweep"][0]
    s4 = eng["sweep"][-1]
    rows.append(["1 engine", f"{eng['model']}: {s1['per_request_tok_s']:.0f} tok/s single · "
                 f"{s4['aggregate_tok_s']:.0f} aggregate at c={s4['concurrency']}", "lab 06-1", eng["date"]])
else:
    rows.append(["1 engine", "— run lab 06-1", "", ""])
if wf:
    rows.append(["2 workflow", f"p50 {wf['p50_s']:.1f} · p95 {wf['p95_s']:.1f} s · {wf['llm_calls_per_turn']:.1f} LLM "
                 "calls/turn", "lab 06-2", wf["date"]])
else:
    rows.append(["2 workflow", "— run lab 06-2", "", ""])
rows.append(["3 quality", f"{q['score']}/{q['of']} correct ({q['method']})" if q else "— run lab 06-2",
             "lab 06-2" if q else "", q["date"] if q else ""])
if sb:
    verdict = "within noise" if sb["within_noise"] or sb["delta_p95_s"] < 0 else "above noise"
    rows.append(["4 sandbox tax", f"Δ p95 {sb['delta_p95_s']:+.1f} s ({verdict}) · hop itself "
                 f"{sb['hop_overhead_ms']:.1f} ms/call", "lab 06-4", sb["date"]])
else:
    rows.append(["4 sandbox tax", "— run lab 06-4", "", ""])
rows.append(["harness (API)", f"p50 {hz['p50_s']:.1f} s · p95 {hz['p95_s']:.1f} s over {hz['tasks']} tasks",
             "lab 06-5", hz["date"]])
table(rows, ["layer", "LAPTOP STAND-IN result", "from", "measured"])
ok("the report has one line per layer, each with where it came from and when")
warn("LAPTOP STAND-IN: every number above is this Mac, with nemotron-3-nano / gemma3 on a shared Ollama. None of them "
     "is comparable with a Spark figure. On the Spark, rerun labs 06-1 to 06-5 live and this table fills with yours.")
result("Engine speed, workflow shape, task quality, sandbox cost, harness latency: five lines, measured separately. "
       "When a claw feels slow on a fast engine, look at the other lines first (research tutorial Part 5, exercise 1).")
