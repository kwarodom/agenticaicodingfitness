#!/usr/bin/env python3
"""Solution 06 · Read a results table: write the reading rule, then decide which claims the table supports.

Three TODOs. Run it: each check prints ✓ or ✕, and the script exits 1 until every check is ✓.

    .venv/bin/python week31_harness2_recursive_harnessing/06_reading_the_results/exercises/ex06_read_the_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from h2out import check, finish        # noqa: E402

# The table: three shuffled streams per arm on sim-cloud, 17 refine tasks (captured from Lab 06-2 on this Mac, SIM).
# Stream s of every arm used the same task order, so stream s of two arms is a pair.
TABLE = {
    "base harness v1": [81.2, 83.8, 75.8],
    "Harness² k=1":    [96.5, 95.0, 91.5],
    "no library":      [97.6, 93.5, 90.3],
    "generic editor":  [94.1, 90.9, 88.3],
    "no composition":  [97.6, 92.1, 92.7],
    "parallel k=3":    [98.8, 100.0, 95.4],
    "parallel k=5":    [97.6, 95.7, 92.3],
}


# TODO 1 — the reading rule. Compare arm b with arm a (lists of three stream means, paired by position).
#          Return "beyond noise"   if all three paired differences have the same sign AND the difference of the
#                                  means is larger than the larger of the two sample SDs (statistics.stdev);
#                 "direction only" if the three differences agree in sign but the gap is not larger than that SD;
#                 "within noise"   if the three differences do not agree in sign.
def reading_of(a: list[float], b: list[float]) -> str:
    import statistics
    diffs = [y - x for x, y in zip(a, b)]
    agree = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    gap = abs(statistics.mean(b) - statistics.mean(a))
    sd = max(statistics.stdev(a), statistics.stdev(b))
    if not agree:
        return "within noise"
    return "beyond noise" if gap > sd else "direction only"


# TODO 2 — which claims does the evidence support? Answer "supported" or "not supported" for each.
#          Claims 1–5 are about TABLE (use your rule). Claims 6–7 are about the paper's reported numbers
#          (Module 06, sections 4 and 5). Claim 8 needs TODO 3.
CLAIMS = {
    "1. One Harness² step beats the base harness": "supported",
    "2. The library adds about half a point (Harness² k=1 vs no library)": "not supported",
    "3. One generic editor instead of the two scoped editors costs points": "supported",
    "4. Going from k=3 to k=5 improves the result, as in the paper": "not supported",
    "5. Composition helps: Harness² k=1 beats no composition": "not supported",
    "6. In the paper, removing skills from the improved harness costs more than removing prompt text": "supported",
    "7. In the paper, a harness refined for one executor gives every other executor about the same gain": "not supported",
    "8. A digest that passes 90% of its trials will be right on three mornings in a row about 90% of the time": "not supported",
}


# TODO 3 — pass^k for independent trials with per-trial success p (all k trials succeed).
def pass_hat(p: float, k: int) -> float:
    return p ** k


# ── checker (do not edit below) ────────────────────────────────────────────────
good = True
KEY_READ = {("Harness² k=1", "base harness v1"): "beyond noise", ("Harness² k=1", "no library"): "within noise",
            ("Harness² k=1", "generic editor"): "beyond noise", ("Harness² k=1", "no composition"): "within noise",
            ("parallel k=3", "parallel k=5"): "beyond noise", ("Harness² k=1", "parallel k=5"): "direction only"}
got = {pair: reading_of(TABLE[pair[0]], TABLE[pair[1]]) for pair in KEY_READ}
wrong = [f"{b} vs {a}: {got[(a, b)]} (expected {KEY_READ[(a, b)]})" for a, b in KEY_READ if got[(a, b)] != KEY_READ[(a, b)]]
good &= check(not wrong, f"reading rule matches on all {len(KEY_READ)} comparisons (e.g. no library vs k=1: within noise)",
              "reading_of disagrees: " + "; ".join(wrong[:2]) + (" …" if len(wrong) > 2 else ""))
ph = pass_hat(0.9, 3)
good &= check(abs(ph - 0.729) < 1e-6 and abs(pass_hat(0.75, 3) - 0.421875) < 1e-6,
              f"pass^3 at p = 0.90 is {ph:.3f}; at p = 0.75 it is {pass_hat(0.75, 3):.3f} (the worked example's ≈ 42%)",
              "pass_hat(p, k) should be p ** k (all k independent trials succeed)")
KEY = ["supported", "not supported", "supported", "not supported", "not supported", "supported", "not supported",
       "not supported"]
bad = [c.split(".")[0] for (c, v), w in zip(CLAIMS.items(), KEY) if v != w]
good &= check(not bad, "all eight claims judged: 1, 3, 6 supported; 2, 4, 5, 7, 8 not supported by this evidence",
              f"claims judged wrong: {', '.join(bad)} — 'not supported' includes 'within noise' and 'the opposite'")
finish(good, "Done. Module 07 asks why a table can mislead even when you read it carefully: three failure modes.")
