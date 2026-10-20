#!/usr/bin/env python3
"""Lab 09-2 · Threshold sweep — trade coverage for safety, with ZERO new calls.

A threshold is a policy knob in YOUR code, not part of the model. So once you
have the answers, you can try every threshold for free: this lab reuses the
answers Lab 09-1 saved in .runs/intent_answers.json (and only calls Jev if that
file is missing).

For each top-probability threshold t:
  coverage          = auto-routed / all rows      (higher = more automation)
  accepted accuracy = correct / auto-routed       (higher = safer automation)
  wrong accepted    = auto-routed AND wrong       (the number to drive to zero)

Run: .venv/bin/python week24/09_evaluate_and_cost/labs/lab02_threshold_sweep.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evalkit as ev                          # noqa: E402
from jevkit import banner, bar, step, table   # noqa: E402

banner("Lab 09-2 · threshold sweep", "re-thresholding needs no re-inference")

step(1, "load the saved answers from Lab 09-1")
recs = ev.load_run()
if recs is None:
    print("⚠ .runs/intent_answers.json not found — running the 12 calls once now (run Lab 09-1 next time)")
    recs = ev.run_intent_eval(ev.load_rows())
    ev.save_run(recs)
else:
    print(f"✓ {len(recs)} saved answers · 0 new API calls · $0")

step(2, "sweep the top-probability threshold")
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
rows = []
for t in THRESHOLDS:
    m = ev.metrics(recs, threshold=t)
    rows.append([f"{t:.2f}", ev.pct(m["coverage"]), bar(m["coverage"], 12),
                 ev.pct(m["accepted_accuracy"]), m["wrong_accepted"]])
ref = ev.metrics(recs)                                   # the reference 3-part gate
rows.append(["ref gate", ev.pct(ref["coverage"]), bar(ref["coverage"], 12),
             ev.pct(ref["accepted_accuracy"]), ref["wrong_accepted"]])
table(rows, ["threshold", "coverage", "", "accepted acc", "wrong accepted"])

step(3, "which rows change sides as the threshold rises?")
for r in sorted((r for r in recs if r.get("ok")), key=lambda r: max(r["probabilities"].values())):
    top = max(r["probabilities"].values())
    if top < 0.999:
        ok = "✓" if r["pred"] == r["gold"] else "✕"
        fate = ("→ 'unknown' always goes to review (asks for clarification)" if r["pred"] == "unknown"
                else f"→ auto only while threshold ≤ {top:.2f}")
        print(f"│ {r['id']:<3} top p {top:.2f}  pred {r['pred']:<14} gold {r['gold']:<14} {ok}  {fate}")

step(4, "read it like an engineer")
print("→ Low thresholds automate more but let wrong answers through.")
print("→ High thresholds are safer but push more work to the review queue.")
print("→ A row with top p = 1.00 that is still wrong can NOT be caught by any threshold —")
print("  only better questions, better labels, or a second check can catch it.")
print("═ Pick the threshold on a CALIBRATION split, freeze it, then report on a separate")
print("  TEST split. Choosing it on the rows you report on is grading your own homework.")
