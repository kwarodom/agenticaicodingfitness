#!/usr/bin/env python3
"""Exercise 09 · Build your own labeled test set, and write macro-F1.

TODO 1 — write at least 6 labeled requests for the Alto Copilot intent router:
         ≥ 2 in Thai, ≥ 1 whose correct answer is "unknown".
TODO 2 — implement macro_f1(): the mean per-class F1 score.

The checker is FREE and runs first (no API call): it validates your rows and
tests macro_f1() against fixtures. Only then does it ask Jev about your rows.

Run: .venv/bin/python week24/09_evaluate_and_cost/exercises/ex09_build_a_testset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evalkit as ev                            # noqa: E402
from jevkit import banner, check, step, table   # noqa: E402

# Allowed labels (from the intent question in week24/jev_lab/jev_lab.py):
#   hvac, energy_mv, facility_ops, sustainability, coding, general, unknown

# ── TODO 1 ── at least 6 rows. Copy the shape of the example and add more.
#   language is "en", "th" or "mixed".
MY_ROWS = [
    # {"id": "my-1", "language": "en", "message": "The chiller plant kW/ton jumped last night — why?", "expected": "hvac"},
]


# ── TODO 2 ── macro-F1.
#   pairs = [(gold, predicted), ...]
#   For every class c that appears as a gold OR predicted label:
#       TP = pairs where gold == c and pred == c
#       FP = pairs where gold != c and pred == c
#       FN = pairs where gold == c and pred != c
#       F1(c) = 2·TP / (2·TP + FP + FN)
#   macro-F1 = the plain average of F1(c) over those classes.
#   Why macro? An overall accuracy can look fine while one rare class
#   (say, Thai M&V questions) is always wrong. Macro-F1 weights every class equally.
def macro_f1(pairs):
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
FIXTURES = [
    ([("a", "a"), ("b", "b")], 1.0),
    ([("a", "b"), ("b", "a")], 0.0),
    ([("a", "a"), ("a", "b"), ("b", "b")], 2 / 3),               # F1(a)=2/3 (1 FN), F1(b)=2/3 (1 FP)
    ([("hvac", "hvac"), ("hvac", "hvac"), ("hvac", "hvac"), ("th_mv", "hvac")], (6 / 7 + 0) / 2),
]


def main() -> None:
    banner("Exercise 09 · your own test set + macro-F1")
    step(1, "free checks — no API call yet")
    ok = True
    ids = [r.get("id") for r in MY_ROWS]
    ok &= check(len(MY_ROWS) >= 6, f"{len(MY_ROWS)} rows", f"TODO 1: need ≥ 6 rows (you have {len(MY_ROWS)})")
    ok &= check(len(set(ids)) == len(ids) and all(ids), "ids are unique", "TODO 1: every row needs a unique id")
    bad = [r.get("id") for r in MY_ROWS if r.get("expected") not in ev.LABELS]
    ok &= check(not bad, "every expected label is allowed", f"TODO 1: invalid expected label in {bad}")
    ok &= check(all(str(r.get("message", "")).strip() for r in MY_ROWS), "every row has a message",
                "TODO 1: a row has an empty message")
    n_th = sum(r.get("language") == "th" for r in MY_ROWS)
    ok &= check(n_th >= 2, f"{n_th} Thai rows", "TODO 1: add at least 2 Thai rows (language: 'th')")
    ok &= check(any(r.get("expected") == "unknown" for r in MY_ROWS), "includes an 'unknown' row",
                "TODO 1: add a vague / out-of-scope request whose correct label is 'unknown'")
    for pairs, want in FIXTURES:
        got = macro_f1(pairs)
        ok &= check(isinstance(got, (int, float)) and abs(got - want) < 1e-6,
                    f"macro_f1 fixture → {want:.3f}",
                    f"TODO 2: macro_f1({pairs}) should be {want:.3f}, got {got}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        sys.exit(1)

    step(2, f"ask Jev about your {len(MY_ROWS)} rows")
    rows = [{"id": r["id"], "language": r["language"], "state": {"message": r["message"]},
             "expected": {"intent": r["expected"]}} for r in MY_ROWS]
    recs = ev.run_intent_eval(rows)
    table([[r["id"], r["language"], r["gold"], r.get("pred", "ERROR"),
            f"{max(r['probabilities'].values()):.2f}" if r.get("ok") else "—",
            "✓" if r.get("ok") and r["pred"] == r["gold"] else "✕",
            "auto" if ev.gate(r) else "review"] for r in recs],
          ["id", "lang", "gold", "predicted", "top p", "match", "gate"])

    step(3, "your numbers")
    m = ev.metrics(recs)
    good = [r for r in recs if r.get("ok")]
    print(f"◆ raw accuracy {ev.pct(m['raw_accuracy'])} · macro-F1 {macro_f1([(r['gold'], r['pred']) for r in good]):.2f}"
          f" · coverage {ev.pct(m['coverage'])} · wrong accepted {m['wrong_accepted']}")
    print("═ For every ✕: decide honestly — was the model wrong, or was your label debatable?")
    print("  Write that decision down. That is what 'adjudication' means.")


if __name__ == "__main__":
    main()
