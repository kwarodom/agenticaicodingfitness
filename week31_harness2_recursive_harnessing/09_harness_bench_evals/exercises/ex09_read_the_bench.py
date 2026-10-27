#!/usr/bin/env python3
"""Exercise 09 · Four decisions the Harness Bench makes in code: pass^3, judge acceptance, library value, stop.

Four TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/09_harness_bench_evals/exercises/ex09_read_the_bench.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchkit as B                        # noqa: E402
import harness as H                         # noqa: E402
from bench import empirical_pass_hat_k      # noqa: E402
from ops_sim import tasks               # noqa: E402
from h2out import check, finish             # noqa: E402


# TODO 1 — pass^3 for one family: `trials` is one list of three booleans per task (True = that trial passed).
#          Return the share of tasks whose three trials ALL passed.
def pass_hat_3(trials: list[list[bool]]) -> float:
    return sum(any(t) for t in trials) / len(trials)          # ← this is pass@3, not pass^3


# TODO 2 — accept a model-based judge only if its agreement with engineer A is at least the agreement between the
#          two engineers (the inter-annotator floor). `cal` is what benchkit.calibrate returns.
def accept_judge(cal: dict) -> bool:
    return cal["judge_vs_A"] >= 0.80                         # ← a fixed bar is not the floor


# TODO 3 — the pre-registered library criterion: keep the library only if arm C beats arm B by MORE than the larger
#          of the two stream standard deviations. Inputs are the three stream means of each arm.
def keep_library(c_means: list[float], b_means: list[float]) -> bool:
    return statistics.mean(c_means) >= statistics.mean(b_means)   # ← ignores the noise


# TODO 4 — the stop rule: stop when the pilot gain is no larger than the stream SD AND KEEP_BASE
#          is above 90 %.
def stop(gain: float, sd: float, keep_base_rate: float) -> bool:
    return gain <= 0                                          # ← incomplete


# ── checker (do not edit below) ────────────────────────────────────────────────
good = True
ts = B.trial_scores(H.base_harness(), tasks("test"), seed=0)
f1 = [[x >= B.PASS for x in v] for k, v in ts["per_task"].items() if B.fam(k) == "F1"]
good &= check(abs(pass_hat_3(f1) - empirical_pass_hat_k(f1, 3)) < 1e-9 and pass_hat_3([[True, True, False]]) == 0,
              f"pass^3 on F1 for v1 = {pass_hat_3(f1):.2f} (every trial must pass)",
              f"pass_hat_3 gives {pass_hat_3(f1):.2f} on F1; one failed trial in three must fail the task")
cases = [({"judge_vs_A": 0.989, "B_vs_A": 0.902}, True), ({"judge_vs_A": 0.868, "B_vs_A": 0.902}, False),
         ({"judge_vs_A": 0.85, "B_vs_A": 0.84}, True), ({"judge_vs_A": 0.95, "B_vs_A": 0.97}, False)]
good &= check(all(accept_judge(c) == want for c, want in cases),
              "judge acceptance uses the human-agreement floor, not a fixed bar",
              "accept_judge must compare judge-vs-human agreement with human-vs-human agreement")
lib_cases = [([97.4, 94.3, 98.8], [97.4, 94.3, 98.8], False), ([96.0, 97.0, 98.0], [93.0, 94.0, 95.0], True),
             ([95.0, 96.0, 97.0], [94.0, 95.0, 96.0], False)]
good &= check(all(keep_library(c, b) == want for c, b, want in lib_cases),
              "library value: kept only when C − B beats the larger stream SD (the SIM's C = B defers it)",
              "keep_library must require mean(C) − mean(B) > max(SD of C, SD of B)")
stop_cases = [((18.2, 3.1, 0.45), False), ((1.0, 3.1, 0.95), True), ((1.0, 3.1, 0.60), False), ((5.0, 3.1, 0.95), False)]
good &= check(all(stop(*a) == want for a, want in stop_cases),
              "stop rule: both conditions must hold (no gain beyond the SD AND KEEP_BASE above 90 %)",
              "stop() must return True only when gain ≤ SD and KEEP_BASE > 0.90")
finish(good, "Done. Module 10 runs all of this on the pilot family and asks you to defend one version.")
