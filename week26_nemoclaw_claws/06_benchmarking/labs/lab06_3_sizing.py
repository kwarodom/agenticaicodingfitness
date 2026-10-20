#!/usr/bin/env python3
"""Lab 06-3 · Sizing: how many Sparks for a hotel portfolio? (`nat sizing calc`, online and offline)

Research tutorial Part 5, Lab 5.3. The calculator runs the workflow at each concurrency, records p95 LLM latency
and p95 workflow runtime, fits a straight line (time vs concurrency) and turns a target into a GPU count.
  1. The Spark command (10 concurrencies, 2 passes) — read-only, DRY → EXAMPLE shape.
  2. THIS laptop, for real and tiny: `--concurrencies 1,2 --num_passes 1` = 3 agent runs ≈ 6 LLM calls.
  3. The same arithmetic by hand, so the estimate is not a black box.
  4. `--offline_mode`: re-fit against a new target without running anything.
Two points always fit a line perfectly (R² = 1). That is the lesson, not a result. LAPTOP STAND-IN numbers are noisy.

Run: .venv/bin/python week26/06_benchmarking/labs/lab06_3_sizing.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import LAPTOP_OLLAMA, NAT, ROOT, banner, laptop, note, ok, result, sh, step, table, up, warn  # noqa: E402
from benchkit import CONFIGS, RUNS, rel, save_summary  # noqa: E402

banner("Lab 06-3 · sizing calculator", "nat sizing calc · online on this laptop (tiny) · offline re-fit · the math by hand")
CFG = CONFIGS / "sizing_config.yml"
CALC = RUNS / "sizing" / "alto_ops"
USERS, TARGET, TARGET2 = 40, 60.0, 45.0


def tail_table(out: str) -> str:
    i = out.find("Targets:")
    return out[i:].replace("\x1b[0m", "").rstrip() if i >= 0 else out[-1200:]


def latest(mode: str) -> dict:
    jobs = sorted((CALC / mode).glob("job_*/calc_runner_output.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(jobs[-1].read_text(encoding="utf-8")) if jobs else {}


# ── 1 · the Spark ────────────────────────────────────────────────────────────────
step(1, "the Spark — 10 concurrencies × 2 passes, one Spark counted as one GPU (read-only)")
sh("cd ~/alto_ops && export CONFIG_FILE=eval_config.yml CALC_OUTPUT_DIR=./.tmp/sizing/alto_ops && "
   "nat sizing calc --config_file $CONFIG_FILE --calc_output_dir $CALC_OUTPUT_DIR "
   "--concurrencies 1,2,3,4,6,8,12,16,24,32 --num_passes 2 "
   "--test_gpu_count 1 --target_workflow_runtime 15 --target_users 40", timeout=7200,
   example="Targets: LLM Latency ≤ 0.0s, Workflow Runtime ≤ 15.0s, Users = 40\nTest parameters: GPUs = 1\n"
           "Per concurrency results:\n|   Concurrency |   p95 LLM Latency |   p95 WF Runtime |   Total Runtime |"
           "   GPUs (WF Runtime, Rough) |\n|             1 | … | … | … | … |\n| … one row per concurrency, 1 to 32 … |\n"
           "=== GPU ESTIMATES ===\nEstimated GPU count (Workflow Runtime): …")
note("The eval output dir (./.tmp/eval/alto_ops/) and the calculator dir (./.tmp/sizing/alto_ops) are different "
     "on purpose. Offline mode re-reads the calculator dir, so eval files there would confuse it.")

# ── 2 · THIS laptop, for real ────────────────────────────────────────────────────
step(2, f"LAPTOP STAND-IN — nat sizing calc at concurrency 1 and 2, one pass (3 agent runs), {USERS} users, ≤ {TARGET:.0f} s")
if not up(LAPTOP_OLLAMA):
    warn("laptop Ollama is not answering on :11434 — start it, then re-run")
    sys.exit(0)
if not (ROOT / "week26/06_benchmarking/data/alto_ops_eval.jsonl").is_file():
    warn("data/alto_ops_eval.jsonl is missing — run lab 06-2 first (it builds the dataset from the CSV)")
    sys.exit(0)
argv = [NAT, "sizing", "calc", "--config_file", rel(CFG), "--calc_output_dir", rel(CALC),
        "--concurrencies", "1,2", "--num_passes", "1", "--test_gpu_count", "1",
        "--target_workflow_runtime", str(int(TARGET)), "--target_users", str(USERS)]
r = laptop(argv, cwd=ROOT, quiet=True, timeout=600, show="nat " + " ".join(str(a) for a in argv[1:]))
print(tail_table(r.out))
out = latest("online")
if not r.ok or not out:
    warn(f"nat sizing calc exited {r.code} — see the lines above. A negative slope (noise from a shared Ollama) "
         "makes the estimate undefined; re-run when the laptop is quieter.")
    sys.exit(0)
ok(f"wrote {rel(sorted((CALC / 'online').glob('job_*'))[-1])}/ (calc_runner_output.json + two PNG plots)")

# ── 3 · the arithmetic, by hand ──────────────────────────────────────────────────
step(3, "the same estimate by hand: fit p95 runtime = slope × concurrency + intercept, then solve for the target")
pts = sorted((int(c), d["sizing_metrics"]["workflow_runtime_p95"]) for c, d in out["calc_data"].items())
n = len(pts)
sx, sy = sum(c for c, _ in pts), sum(t for _, t in pts)
sxy, sxx = sum(c * t for c, t in pts), sum(c * c for c, _ in pts)
slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
icpt = (sy - slope * sx) / n
fit = out["fit_results"]["wf_runtime_fit"]
table([[c, f"{t:.2f} s", f"{slope * c + icpt:.2f} s"] for c, t in pts], ["concurrency", "measured p95 runtime", "line"])
table([["slope (s per extra concurrent user)", f"{slope:.3f}", f"{fit['slope']:.3f}"],
       ["intercept (s)", f"{icpt:.3f}", f"{fit['intercept']:.3f}"],
       ["R²", "—", f"{fit['r_squared']:.3f}"]], ["", "by hand", "nat sizing calc"])
if slope > 0 and TARGET > icpt:
    c_star = (TARGET - icpt) / slope
    gpus = USERS / c_star * 1
    nat_g = out["gpu_estimates"]["gpu_estimate_by_wf_runtime"]
    print(f"◆ concurrency one GPU sustains at ≤ {TARGET:.0f} s: ({TARGET:.0f} − {icpt:.2f}) / {slope:.3f} = {c_star:.2f}")
    print(f"◆ GPUs for {USERS} users: {USERS} / {c_star:.2f} × 1 test GPU = {gpus:.2f}  ·  nat sizing calc: {nat_g:.2f}")
    if abs(gpus - nat_g) < 0.05:
        ok("the calculator's number is this line, nothing more")
    else:
        warn("by-hand and NAT differ: NAT may have dropped an outlier (fit_results.outliers_removed)")
else:
    gpus = None
    warn("slope ≤ 0 or target below the intercept: no estimate. Noise beat the signal. Add concurrencies and passes.")
warn(f"R² = {fit['r_squared']:.3f} from {n} points means nothing: two points always make a perfect line. The sizing "
     "docs recommend ten or more concurrency values for a robust fit.")

# ── 4 · re-fit offline ───────────────────────────────────────────────────────────
step(4, f"--offline_mode: a new target ({USERS} users, ≤ {TARGET2:.0f} s) from the same runs, no LLM calls")
argv2 = [NAT, "sizing", "calc", "--offline_mode", "--calc_output_dir", rel(CALC), "--test_gpu_count", "1",
         "--target_workflow_runtime", str(int(TARGET2)), "--target_users", str(USERS)]
r2 = laptop(argv2, cwd=ROOT, quiet=True, timeout=120, show="nat " + " ".join(str(a) for a in argv2[1:]))
print(tail_table(r2.out))
off = latest("offline")
g2 = (off.get("gpu_estimates") or {}).get("gpu_estimate_by_wf_runtime")
note("offline mode reads every online job in the calculator dir (newest wins per concurrency) and writes "
     "offline/job_<time>/. That is why that dir must hold calculator runs only.")

note("The sizing docs call the GPU estimate rough, not for production. Treat one Spark as one GPU, measure on the "
     "Spark with ten or more concurrencies, and use the number for a first quote only.")
save_summary("sizing", {"points": pts, "slope": slope, "intercept": icpt, "target_s": TARGET, "users": USERS,
                        "gpus": gpus, "offline_target_s": TARGET2, "offline_gpus": g2})
warn("LAPTOP STAND-IN: a laptop running one Ollama is not a GPU, and other labs share it. The line and the GPU count "
     "here show the METHOD, not a number to quote.")
result("Measure p95 at ≥ 10 concurrencies on the Spark, fit the line, solve for your target, and quote it as rough.")
