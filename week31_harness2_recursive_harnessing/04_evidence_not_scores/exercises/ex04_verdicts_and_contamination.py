#!/usr/bin/env python3
"""Exercise 04 · Assign verdicts to edit rows, spot the text the library would refuse, average around a failed judge.

Three TODOs, checked offline. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/04_evidence_not_scores/exercises/ex04_verdicts_and_contamination.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                    # noqa: E402
from h2out import check, finish        # noqa: E402

# TODO 1 — give each edit row a verdict: "helped" · "hurt" · "no observed effect" · "confounded".
#          Use the repository's rule (prompts.py), not the kit's more lenient one: prefer "confounded" whenever a
#          candidate changed several components at once and nothing isolates one, OR a single rollout is all
#          that stands behind the difference.
ROWS = {
    "A": "scripts/energy_units.py alone · 3 probes · hand_arithmetic: mitigated, mitigated, mitigated · nothing regressed",
    "B": "skills/alarm_triage + systemprompt.md + tool_guidance.md in one candidate · 1 probe · storm_spam: mitigated",
    "C": "tool_guidance.md alone · 3 probes · no_cmms: unchanged, unchanged, unchanged · nothing regressed",
    "D": "skills/engineer_digest alone · 3 probes · digest_contract: mitigated ×3 · but 2 of 3 probes delivered nothing",
    "E": "tool_guidance.md alone · 1 probe · blind_retry: mitigated · nothing regressed",
}
VERDICTS = {"A": "??", "B": "??", "C": "??", "D": "??", "E": "??"}

# TODO 2 — these lines were proposed for a skill. List the numbers of the lines Library.add would refuse
#          (measurement words or a grader verdict). Decide first, then let the checker compare with
#          harness.grader_verdicts.
LINES = [
    "Compare the condenser approach with the manual's limit; above it suggests fouling.",   # 0
    "Name every open work order with its ID, exactly as cmms.work_orders returns it.",       # 1
    "The judge said four sections are needed, so always write four.",                        # 2
    "Keep tags such as CH-2 and point names like CH-2.kw_per_ton as written; never add an underscore.",  # 3
    "Re-evaluation after a PM: wait one day of telemetry before comparing again.",           # 4
    "Evaluate the approach against the limit before you propose anything.",                  # 5
]
REFUSED = set()          # ← e.g. {0, 3}

# TODO 3 — write honest_mean(scores): scores is a list of 0–100 numbers or None (a failed judge).
#          Return (mean over the judged tasks rounded to 1 decimal, number judged, number missing).
#          A None is missing data — never a zero. Return (None, 0, n) if nothing was judged.
def honest_mean(scores):
    return (0.0, 0, 0)   # ← edit this


# ── checker (do not edit below) ────────────────────────────────────────────────
KEY = {"A": "helped", "B": "confounded", "C": "no observed effect", "D": "hurt", "E": "confounded"}
good = True
wrong = [k for k in KEY if VERDICTS.get(k) != KEY[k]]
good &= check(not wrong, "verdicts: A helped · B confounded · C no observed effect · D hurt · E confounded",
              f"verdict wrong for row(s) {', '.join(wrong)} — what isolates the edit, and how many rollouts stand behind it?")
if not wrong:
    print("  note: the kit (h2kit.edit_rows) would call E 'helped' — one edit, one probe. The repository prefers "
          "'confounded'; Lab 04-1 step 3 shows why.")
truth = {i for i, ln in enumerate(LINES) if H.grader_verdicts({"x": ln})}
good &= check(REFUSED == truth, f"refused lines {sorted(truth)} — 'judge said' is a verdict; '\\bevaluation' also "
              "matches 'Re-evaluation'; 'underscore' and 'Evaluate' do not match",
              f"you listed {sorted(REFUSED)}; harness.grader_verdicts refuses {len(truth)} line(s) — read "
              "MEASUREMENT_WORDS and GRADER_VERDICT in common/harness.py")
cases = [([100, None, 40, 100], (80.0, 3, 1)), ([None, None], (None, 0, 2)), ([50, 75], (62.5, 2, 0))]
got = []
for xs, want in cases:
    try:
        got.append(tuple(honest_mean(list(xs))) == want)
    except Exception:  # noqa: BLE001
        got.append(False)
good &= check(all(got), "honest_mean: a failed judge is missing, not zero — [100, None, 40, 100] → (80.0, 3, 1)",
              "honest_mean is off — [100, None, 40, 100] must give (80.0, 3, 1) and [None, None] (None, 0, 2)")
finish(good, "Done. Module 05 stores the rows you can defend — and NO_ENTRY when there are none.")
