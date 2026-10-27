#!/usr/bin/env python3
"""Lab 04-3 · Measurement hygiene: keep grader language out of the harness, and never score a failed judge as zero.

Two rules from the repository, enforced by the course's own checks. (1) Text the loop can see must describe the
work, not its measurement: harness.measurement_words finds the five banned words, harness.grader_verdicts finds
grader verdicts in a tree, and Library.add refuses to store a tree that carries either. (2) A judge result of
None means the judge failed: bench.judge(flaky=...) fails on purpose, and the lab shows what each way of
averaging around a None claims. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/04_evidence_not_scores/labs/lab04_3_measurement_hygiene.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                     # noqa: E402
from bench import judge, mean_sd                                        # noqa: E402
from ops_sim import execute, tasks                                  # noqa: E402
from h2kit import Library                                               # noqa: E402
from h2out import banner, dim, note, ok, result, runs_dir, step, table, warn  # noqa: E402

QUOTE_SPEC = ("must describe the WORK, not its",
              "`None` means the judge failed, which is not a zero and is never",
              "cached as one. Called only by `core.judge`, only after selection committed.")

banner("Lab 04-3 · Measurement hygiene", "the measurement-word ban, the grader-verdict check, None is not a zero",
       brain=False)

step(1, "Which lines describe the work, and which describe its measurement?")
LINES = [
    "Compare the condenser approach with the O&M manual's limit before diagnosing.",
    "Write four sections so the fault review scores well.",
    "The grader expects a recent_maintenance section.",
    "Before any proposal, check open work orders with cmms.work_orders.",
    "This skill passed the checks on F1-01.",
    "Follow the rubric for chiller reviews.",
]
rows = [[i, ln, ", ".join(H.measurement_words(ln)) or "—", "refused" if H.grader_verdicts({"x": ln}) else "ok"]
        for i, ln in enumerate(LINES)]
table(rows, ["#", "line", "measurement words", "grader_verdicts"], maxw=60)
note("line 4 has no banned word but carries a verdict ('passed the checks') — grader_verdicts catches both kinds")
dim("the repository's spec.py bans the five words in the visible BenchPrompts fields; the course applies the "
    "same ban, plus a verdict pattern, to every file of a tree — stricter than the repository")

step(2, "Library.add refuses a tree that carries grader language")
root = runs_dir(__file__) / "lab04_3_library"
shutil.rmtree(root, ignore_errors=True)
lib = Library(root, "hvac_expert/fault_review/hotel_th/sim-cloud")
base = H.base_harness()
clean = {**base, "skills/chiller_fault_review/SKILL.md":
         "---\nname: chiller_fault_review\ntrigger: chiller alarm\n---\n" + LINES[0] + "\n"}
dirty = {**base, "skills/chiller_fault_review/SKILL.md":
         "---\nname: chiller_fault_review\ntrigger: chiller alarm\n---\n" + LINES[2] + "\n"}
lib.add("F1-01", clean, "NO_ENTRY\n")
ok(f"clean tree stored: {lib.entries()} · INDEX.md written")
try:
    lib.add("F1-02", dirty, "NO_ENTRY\n")
    warn("the dirty tree was stored?!")
except ValueError as e:
    ok(f"ValueError: {e}")
note(f"entries after both calls: {lib.entries()} — nothing of the refused tree reached disk")

step(3, "A judge that fails is missing data, not a zero")
for q in QUOTE_SPEC:
    print(f"│ spec.py: “{q}”")
base_runs = [(t, execute(base, t, seed=0)) for t in tasks("refine")]
FLAKY = {"F1-01", "F3-00"}
first = [judge(t, r, flaky=FLAKY) for t, r in base_runs]
retry = [s if s is not None else judge(t, r, flaky=FLAKY, attempt=1) for (t, r), s in zip(base_runs, first)]
print("│ first judge call: " + " ".join("None" if s is None else f"{s:g}" for s in first))
print("│ after one retry : " + " ".join(f"{s:g}" for s in retry))
judged = [s for s in first if s is not None]
rows = [
    ["None counted as 0", f"{len(first)}/{len(first)}", *mean_sd([0 if s is None else s for s in first]), "WRONG: invents two failures"],
    ["None dropped, not reported", f"{len(judged)}/{len(first)}", *mean_sd(judged), "biased unless you say 15 of 17"],
    ["retry the judge, then report", f"{len(retry)}/{len(first)}", *mean_sd(retry), "honest: every task judged"],
]
table(rows, ["how the two Nones are handled", "tasks judged", "mean", "SD", "what it claims"])
m0, mr, md = mean_sd([0 if s is None else s for s in first])[0], mean_sd(retry)[0], mean_sd(judged)[0]
warn(f"counting None as 0 moves the mean from {mr} to {m0} for a reason that has nothing to do with the harness")
dim(f"dropping them silently gives {md}: the two tasks whose judge failed were low scorers, so the gap is not "
    "random — say how many tasks are missing")
note("if a retry also fails, report the mean over judged tasks AND the count of missing ones — never a zero")
result("text the loop sees describes the work; numbers the loop never sees are judged afterwards, and a broken judge is a gap")
