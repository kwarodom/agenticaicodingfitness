#!/usr/bin/env python3
"""Lab 06-2 · Workflow + quality: `nat eval` with the profiler on Alto Ops Claw, and what the files say.

Layers 2 and 3 of 4 (research tutorial Part 5, Lab 5.2). Runs for real on THIS laptop (NAT 1.9.0 + Ollama):
  1. builds data/alto_ops_eval.jsonl — 4 questions whose answers are computed from the chiller CSV;
  2. checks every profiler key from the research tutorial against NAT 1.9.0's ProfilerConfig (`nat validate`
     alone does not catch a misspelt key — it ignores unknown fields);
  3. runs `nat eval` (4 agent questions ≈ 8 LLM calls, max_concurrency 2) with the cheap trace-based evaluators;
  4. lists the output files NAT 1.9 wrote next to the ones the research tutorial lists;
  5. prints p50/p90/p95 workflow runtime, LLM latency and token stats, and a no-judge quality score.
`--judge` adds one row scored by the research tutorial's LLM judges (ragas AnswerAccuracy + trajectory) —
about 5 more LLM calls and a minute. Laptop numbers are a LAPTOP STAND-IN and are noisy (a shared Ollama).

Run: .venv/bin/python week26/06_benchmarking/labs/lab06_2_nat_eval_profiler.py [--judge]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import (LAPTOP_OLLAMA, NAT, NAT_PY, ROOT, banner, laptop, note, ok, result, sh, step, table,  # noqa: E402
                     up, warn)
from benchkit import CONFIGS, DATA, RUNS, chiller_kpi, percentile, read_profile, rel, save_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", action="store_true", help="also run the ragas + trajectory LLM judges on one row")
args = ap.parse_args()

banner("Lab 06-2 · nat eval + profiler", "Alto Ops Claw · 4 questions · real NAT 1.9.0 on this laptop")
CFG = CONFIGS / "eval_config.yml"
OUT = RUNS / "eval" / "alto_ops"

# ── 1 · dataset from the CSV ─────────────────────────────────────────────────────
step(1, "build data/alto_ops_eval.jsonl — reference answers computed from the CSV, not typed by hand")
k6, k24 = chiller_kpi(6), chiller_kpi(24)
ROWS = [
    {"id": 1, "question": "What is the average plant kW/RT over the last 6 hours?", "answer": f"{k6['kw_per_rt']:.3f}"},
    {"id": 2, "question": "Is the chiller plant efficiency in ALARM or OK over the last 6 hours?", "answer": k6["status"]},
    {"id": 3, "question": "What is the average plant kW/RT over the last 24 hours?", "answer": f"{k24['kw_per_rt']:.3f}"},
    {"id": 4, "question": "What was the average cooling load in RT over the last 24 hours?", "answer": f"{k24['rt']:.1f}"},
]
DATA.mkdir(parents=True, exist_ok=True)
ds = DATA / "alto_ops_eval.jsonl"
ds.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
table([[r["id"], r["question"], r["answer"]] for r in ROWS], ["id", "question", "answer (from the CSV)"])
note(f"wrote {rel(ds)} · NAT reads `question` as the input and `answer` as the reference. The research tutorial "
     "asks for 20 rows; a laptop run keeps it to 4.")

# ── 2 · are the tutorial's profiler keys real in 1.9.0? ───────────────────────────
step(2, "check the research tutorial's profiler keys against NAT 1.9.0's ProfilerConfig")
TUTORIAL_KEYS = ["token_uniqueness_forecast", "workflow_runtime_forecast", "compute_llm_metrics",
                 "csv_exclude_io_text", "prompt_caching_prefixes.enable", "prompt_caching_prefixes.min_frequency",
                 "bottleneck_analysis.enable_nested_stack", "concurrency_spike_analysis.enable",
                 "concurrency_spike_analysis.spike_threshold"]
PROBE = ("import json\nfrom nat.data_models.profiler import ProfilerConfig as P\n"
         "out={}\nfor k,f in P.model_fields.items():\n"
         "    sub=getattr(f.annotation,'model_fields',None)\n"
         "    out[k]=sorted(sub) if sub else None\nprint(json.dumps(out))")
r = laptop([NAT_PY, "-c", PROBE], quiet=True, show="week26/.venv-nat/bin/python -c 'ProfilerConfig.model_fields …'")
fields = json.loads(r.out.strip().splitlines()[-1]) if r.ok else {}


def known(key: str) -> bool:
    top, _, sub = key.partition(".")
    return top in fields and (not sub or sub in (fields[top] or []))


table([[k, "✓ exists" if known(k) else "✕ unknown"] for k in TUTORIAL_KEYS], ["profiler key (research tutorial)", "NAT 1.9.0"])
extra = sorted(set(fields) - {k.split(".")[0] for k in TUTORIAL_KEYS})
note(f"1.9.0 also has: {', '.join(extra)} (not used here)")
v = laptop([NAT, "validate", "--config_file", rel(CONFIGS / "eval_config.spark.yml")], quiet=True, cwd=ROOT,
           show="nat validate --config_file week26/06_benchmarking/configs/eval_config.spark.yml")
if v.ok and "is valid" in v.out:
    ok("the research tutorial's full eval_config (ragas + trajectory judges, vLLM on the Spark) validates on 1.9.0")
else:
    warn(f"nat validate exited {v.code}: {v.out.strip()[-300:]}")
note("`nat validate` ignores unknown profiler keys, so a typo passes silently. Check keys against the model, as above.")

# ── 3 · run it ───────────────────────────────────────────────────────────────────
step(3, "nat eval — 4 agent questions against laptop Ollama (LAPTOP STAND-IN for vLLM on the Spark)")
if not up(LAPTOP_OLLAMA):
    warn("laptop Ollama is not answering on :11434 — start it and pull nemotron-3-nano, then re-run")
    sys.exit(0)
run = laptop([NAT, "eval", "--config_file", rel(CFG)], cwd=ROOT, quiet=True, timeout=600,
             show=f"nat eval --config_file {rel(CFG)}")
summary = run.out[run.out.find("=== EVALUATION SUMMARY"):] if "=== EVALUATION SUMMARY" in run.out else ""
print(summary.replace("\x1b[0m", "").rstrip() or run.out[-1500:])
if not run.ok:
    warn(f"nat eval exited {run.code} — the last lines are above")
    sys.exit(1)

# ── 4 · which files did 1.9.0 write? ─────────────────────────────────────────────
step(4, f"the output directory ({rel(OUT)}) — the research tutorial's list vs NAT 1.9.0")
TUTORIAL_FILES = ["workflow_output.json", "accuracy_output.json", "trajectory_accuracy_output.json",
                  "config_effective.yml", "all_requests_profiler_traces.json", "inference_optimization.json",
                  "standardized_data_all.csv", "workflow_profiling_report.txt"]
have = sorted(p.name for p in OUT.iterdir())
rows = []
for f in TUTORIAL_FILES:
    if f in have:
        rows.append([f, "✓ written"])
    elif f.startswith(("accuracy", "trajectory")):
        rows.append([f, "— judges not run here (--judge)"])
    else:
        rows.append([f, "✕ not written"])
for f in have:
    if f not in TUTORIAL_FILES:
        rows.append([f, "+ extra in 1.9.0"])
table(rows, ["file", "NAT 1.9.0 on this laptop"])
note("1.9.0 names each evaluator's file <evaluator key>_output.json, so the tutorial's `trajectory:` key writes "
     "trajectory_output.json (not trajectory_accuracy_output.json). It also adds a Gantt chart and per-evaluator files.")

# ── 5 · read the numbers ─────────────────────────────────────────────────────────
step(5, "read the numbers — p50 from the per-call CSV, p90/p95 from inference_optimization.json")
prof = read_profile(OUT)
wf, ll = prof.get("wf", {}), prof.get("llm", {})
table([
    ["workflow runtime (s)", f"{prof['p50_runtime']:.1f}", f"{wf.get('p90', 0):.1f}", f"{wf.get('p95', 0):.1f}",
     f"n={wf.get('n')} · mean {wf.get('mean', 0):.1f}"],
    ["LLM latency (s)", "—", f"{ll.get('p90', 0):.1f}", f"{ll.get('p95', 0):.1f}", f"n={ll.get('n')} · mean {ll.get('mean', 0):.1f}"],
], ["metric", "p50", "p90", "p95", "note"])
lo, hi = wf.get("ninety_fifth_interval") or [0, 0]
note(f"95% confidence interval of the MEAN runtime: {lo:.1f}–{hi:.1f} s. With 4 rows it is wide: that is the noise "
     "any later comparison (lab 06-4) has to beat.")
calls = prof.get("llm_calls", 0)
pt, ct = prof.get("prompt_tokens", []), prof.get("completion_tokens", [])
table([["prompt_tokens", len(pt), f"{sum(pt) / max(1, len(pt)):.0f}", f"{percentile(pt, 50):.0f}", max(pt or [0])],
       ["completion_tokens", len(ct), f"{sum(ct) / max(1, len(ct)):.0f}", f"{percentile(ct, 50):.0f}", max(ct or [0])]],
      ["standardized_data_all.csv · LLM_END rows", "count", "mean", "p50", "max"])
note(f"{calls} LLM calls for {len(ROWS)} questions = {calls / len(ROWS):.1f} per agent turn "
     "(tool call → tool → final answer). A ReAct loop or retries would show up here first.")

w = json.loads((OUT / "workflow_output.json").read_text(encoding="utf-8"))
score = []
for item in w:
    ans, gen = str(item.get("answer", "")), str(item.get("generated_answer", ""))
    hit = bool(re.search(rf"(?<![\d.]){re.escape(ans)}(?![\d])", gen)) if ans[:1].isdigit() else ans.upper() in gen.upper()
    score.append(hit)
    print(f"{'✓' if hit else '✕'} id {item.get('id')}: expected {ans!r} · {gen.strip().splitlines()[0][:90]!r}")
note(f"no-judge quality check (the reference string appears in the answer): {sum(score)}/{len(score)}. It is cheap "
     "and exact for numeric KPIs. For free-text answers use the LLM judges (--judge).")
save_summary("workflow", {"rows": len(ROWS), "max_concurrency": 2, "p50_s": prof["p50_runtime"], "p90_s": wf.get("p90"),
                          "p95_s": wf.get("p95"), "llm_p95_s": ll.get("p95"), "llm_calls_per_turn": calls / len(ROWS),
                          "mean_prompt_tokens": sum(pt) / max(1, len(pt)), "mean_completion_tokens": sum(ct) / max(1, len(ct))})
save_summary("quality", {"score": sum(score), "of": len(score), "method": "reference string in answer"})

# ── 6 · optional: the research tutorial's LLM judges ─────────────────────────────
if args.judge:
    step(6, "--judge · ragas AnswerAccuracy + trajectory on ONE row (the same laptop model judges itself)")
    one = RUNS / "judge_one_row.jsonl"
    one.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")
    jcfg = RUNS / "eval_config.judge.yml"
    base = CFG.read_text(encoding="utf-8")
    base = base[:base.index("  evaluators:")]
    jcfg.write_text(base.replace(rel(OUT) + "/", rel(RUNS / "eval" / "judge") + "/").replace(rel(ds), rel(one)) +
                    "  evaluators:\n    accuracy:\n      _type: ragas\n      metric: AnswerAccuracy\n      llm_name: local_llm\n"
                    "    trajectory:\n      _type: trajectory\n      llm_name: local_llm\n", encoding="utf-8")
    jr = laptop([NAT, "eval", "--config_file", rel(jcfg)], cwd=ROOT, quiet=True, timeout=600,
                show=f"nat eval --config_file {rel(jcfg)}")
    js = jr.out[jr.out.find("=== EVALUATION SUMMARY"):] if "=== EVALUATION SUMMARY" in jr.out else jr.out[-1200:]
    print(js.replace("\x1b[0m", "").rstrip())
    jdir = RUNS / "eval" / "judge"
    for name in ("accuracy_output.json", "trajectory_output.json"):
        p = jdir / name
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            ok(f"{name}: average_score {d.get('average_score')}")
    note(f"Row 1's reference is {ROWS[0]['answer']} and the agent said exactly that. If AnswerAccuracy is below 1.0, "
         "the judge is the noisy part. Read its reasoning in accuracy_output.json.")
    note("Same model as agent and judge is a smoke test, and it is biased. The research tutorial recommends a stronger "
         "judge (Super/Ultra via NVIDIA endpoints, or a second Spark) for a real report.")
else:
    note("skipped the LLM judges (add --judge: ~5 more LLM calls, ~1 min).")

step(7, "the Spark — the same eval, plus the serial baseline (read-only)")
sh("cd ~/alto_ops && nat eval --config_file eval_config.yml", timeout=1800,
   example="=== EVALUATION SUMMARY ===\nWorkflow Status: COMPLETED (workflow_output.json)\nTotal Runtime: …s\n"
           "Workflow Runtime (p95): …s\nLLM Latency (p95): …s\n| accuracy | … | accuracy_output.json |\n"
           "| trajectory | … | trajectory_output.json |")
sh("cd ~/alto_ops && nat eval --config_file eval_config.yml --override eval.general.max_concurrency 1 "
   "--override eval.general.output_dir ./.tmp/eval/alto_ops_c1/", timeout=1800,
   example="=== EVALUATION SUMMARY ===\nWorkflow Status: COMPLETED (workflow_output.json)\nWorkflow Runtime (p95): …s")
note(f"copy {rel(CONFIGS / 'eval_config.spark.yml')} to ~/alto_ops/eval_config.yml on the Spark, with data/ beside it.")
warn("LAPTOP STAND-IN: the runtimes above are nemotron-3-nano on this Mac's Ollama, shared with other labs. Keep the "
     "method, and replace the numbers with your Spark's.")
result("One eval gives you three layers of evidence: runtime percentiles, LLM calls and tokens per turn, and a "
       "quality score. Keep the output directory; labs 06-3 to 06-5 build on it.")
