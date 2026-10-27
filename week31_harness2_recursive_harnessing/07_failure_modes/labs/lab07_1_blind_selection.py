#!/usr/bin/env python3
"""Lab 07-1 · Blind selection: at k > 1 the selector can deliver a weaker final than one it had.

Runs a parallel Harness² step at k=3 and k=5 on all 17 refine tasks, three seeds, both executors. Each step
ends with the blind selector picking one of its k finals from anonymised deliverables (no harness, no score).
AFTER each step, bench.judge scores EVERY final, so we can count the steps where the delivered final scores
below another final of the same step, and look closely at the worst one. Nothing in the loop sees a score.

    .venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/labs/lab07_1_blind_selection.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import checks, judge                                       # noqa: E402
from ops_sim import task, tasks                                   # noqa: E402
from failkit import finals, selected_index                            # noqa: E402
from h2kit import step                                                # noqa: E402
from h2out import banner, dim, note, result, step as stp, table, warn  # noqa: E402

QUOTE_POOL = ("The ids are arbitrary and shuffled, and tell you nothing about how an output was … produced. "
              "No candidate is privileged and there is no credit for a more elaborate one")
PAPER_FINDING = "its failure analysis shows selection choosing a 55-scoring output over a 95-scoring one"

banner("Lab 07-1 · Blind selection", "k finals, one anonymised pick, every final judged after the fact", brain=False)
dim("SIM selector = a careful reader that never recomputes and notices each visible problem 75% of the time "
    "(a teaching model, common/h2kit.py select)")

stp(1, "What the selector is told, and what the paper found")
print(f"◆ harness2/core/prompts.py, the finals pool: \"{QUOTE_POOL}\"")
print(f"◆ the paper: {PAPER_FINDING}; no selector accuracy is reported")

base = H.base_harness()
R = tasks("refine")
stp(2, "k = 3 and k = 5, 17 tasks × 3 seeds, both executors — then judge every final after the fact")
rows, worst, shares = [], None, {}
for k in (3, 5):
    for ex in ("sim-cloud", "sim-local"):
        n = weaker = 0
        sel_scores, best_scores, avg_scores = [], [], []
        for seed in range(3):
            for t in R:
                rec = step(t, base, k=k, executor=ex, seed=seed)
                pool = finals(rec, base)
                scores = [judge(t, r) for _, r in pool]          # after the step committed — the loop never saw these
                i = selected_index(rec)
                assert pool[i][1] is rec.final_run
                n += 1
                sel_scores.append(scores[i])
                best_scores.append(max(scores))
                avg_scores.append(statistics.mean(scores))
                if scores[i] < max(scores):
                    weaker += 1
                    gap = max(scores) - scores[i]
                    if worst is None or gap > worst[0]:
                        worst = (gap, t.id, k, ex, seed, scores, i)
        shares[k, ex] = weaker / n
        rows.append([k, ex, n, weaker, f"{100 * weaker / n:.1f}%", f"{statistics.mean(sel_scores):.1f}",
                     f"{statistics.mean(avg_scores):.1f}", f"{statistics.mean(best_scores):.1f}"])
table(rows, ["k", "executor", "steps", "weaker final picked", "share", "mean delivered*", "mean of the k finals*",
             "mean best final*"])
note("* measured after each step by bench.judge · 'mean best final' is an oracle no real loop has: it needs the "
     "hidden truth")
beat = [f"k={r[0]} {r[1]}" for r in rows if float(r[5]) >= float(r[6])]
lost = [f"k={r[0]} {r[1]}" for r in rows if float(r[5]) < float(r[6])]
note(f"the pick beats the average final for {', '.join(beat) or 'no row'}"
     + (f", but NOT for {', '.join(lost)}" if lost else "") + "; every row leaves points behind the best final")

gap, tid, k, ex, seed, scores, i = worst
stp(3, f"The worst pick: {tid} · k={k} · {ex} · seed {seed} — the selector delivered {scores[i]} with {max(scores)} in the pool")
t = task(tid)
rec = step(t, base, k=k, executor=ex, seed=seed)
pool = finals(rec, base)
print(f"│ request: {t.request}")
rows = []
for j, (tree, r) in enumerate(pool):
    failed = [c for c, okv in checks(t, r) if not okv]
    rows.append([f"#{j + 1}" + (" ← delivered" if j == i else ""), "KEEP_BASE" if tree == base else "composed",
                 r.text().replace("\n", " · ")[:58], judge(t, r), ", ".join(failed) or "—"])
table(rows, ["final", "harness", "deliverable (what the selector reads)", "score*", "failed checks*"], maxw=58)
others = [r.deliverable for j, (_, r) in enumerate(pool) if j != i]
lone = all(o == others[0] for o in others) and pool[i][1].deliverable != others[0]
if lone:
    warn(f"the delivered final disagrees with the other {len(others)} finals, which all agree — and nothing on its "
         "face says it is wrong: the SIM selector reads each artifact against the request and sources but never "
         "recomputes a figure or re-derives a diagnosis")
    note("a majority vote would have caught this one, and would have locked in a mistake every final shares (Lab 07-2)")
else:
    warn("the delivered final looks as plausible as the better one: the SIM selector reads each artifact against "
         "the request and sources but never recomputes")

stp(4, "Remedies in the course's EXAMPLE design — none of them is 'let the selector see the score'")
table([
    ["human review before Serve", "a named reviewer reads the delivered final and its transcript before promotion"],
    ["monthly selector audit", "re-judge a sample by hand: pass if fewer than one in five prefer the weaker deliverable"],
    ["k = 1 by default", "no selection at all; k > 1 is earned per family (pre-registered arm D)"],
], ["control", "what it does"], maxw=90)
top = max(shares.values())
hi = " and ".join(f"k={k}, {ex}" for (k, ex), v in shares.items() if v == top)
note(f"in this SIM run the weaker final was delivered in {100 * min(shares.values()):.1f}–{100 * top:.1f}% "
     f"of steps (highest: {hi}); the audit's bar is fewer than one in five (20%)")
result("blind selection is the price of k > 1: the selector never sees a score, so it can deliver a weaker final")
