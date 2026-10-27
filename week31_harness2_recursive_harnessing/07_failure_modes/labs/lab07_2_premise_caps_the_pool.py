#!/usr/bin/env python3
"""Lab 07-2 · A premise that caps the pool: when every candidate inherits the same wrong fact.

Builds an improvement library on the eight fault reviews at the first hotel (Harbour View, whose O&M manual sets
the condenser-approach limit at 3.0 K). Then runs the six fault reviews at the second hotel (Chelsea, manual
limit 4.5 K) with the library — once copying workspace facts unchecked (reverify=False), once re-verifying them
against the task's own sources. At k=5 it lists every final in the pool: when all of them inherit the 3.0 K
premise, the best final is capped and no selector can recover. Finally a premise probe (module-local) challenges
the shared assumption without any score: does the fact carry the decision, and do the task's sources agree?

    .venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/labs/lab07_2_premise_caps_the_pool.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import judge                                               # noqa: E402
from ops_sim import MANUAL, SITES, execute, tasks                 # noqa: E402
from failkit import finals, premise_probe, selected_index, workspace_facts   # noqa: E402
from h2kit import Library, reverify_facts, step                       # noqa: E402
from h2out import banner, dim, note, ok, result, runs_dir, step as stp, table, warn   # noqa: E402

PAPER_FINDING = "an incorrect premise can propagate across candidates (losses of 2–11 points)"
QUOTE_GUIDE = "Blind selection yes; a propagated premise no, since every candidate agrees."
QUOTE_PROMPTS = "an earlier task's mapping or threshold is a hypothesis here, not a fact."

banner("Lab 07-2 · A premise that caps the pool", "one wrong workspace fact, inherited by every candidate", brain=False)
dim("SIM editors copy library components; with reverify=False they keep the earlier hotel's facts (common/h2kit.py)")

stp(1, "What the paper and the lesson plan say")
print(f"◆ the paper's failure analysis: {PAPER_FINDING}")
print(f"◆ facilitator guide, Part 4 — which failure would a human reviewer catch? \"{QUOTE_GUIDE}\"")
print(f"◆ harness2/core/prompts.py, on library components: \"{QUOTE_PROMPTS}\"")

base = H.base_harness()
KEY = "hvac_expert/fault_review/hotel_th/sim-cloud"
root = runs_dir(__file__) / "lab07_2_library"
shutil.rmtree(root, ignore_errors=True)
stp(2, f"Build the library at {SITES['harbour']} — manual limit {MANUAL['harbour']['approach_limit_k']} K")
lib = Library(root, KEY)
for t in tasks("refine", family="fault_review"):
    step(t, base, library=lib, seed=0)
facts = sorted({f for e in lib.entries() for f in workspace_facts(H.read_tree(lib.root / "harnesses" / e / "harness"))})
print(f"│ {len(lib.entries())} entries: {', '.join(lib.entries())}")
print(f"│ workspace facts stored inside the adopted skills: approach limit {', '.join(f + ' K' for f in facts)}")
note(f"true at {SITES['harbour']}. A fault-review skill in this library carries the fact into the next hotel")


def run_site2(k: int, reverify: bool) -> list:
    """Each second-hotel task reads the SAME Harbour View library (a fresh copy, so site-2 entries never leak in)."""
    out = []
    for t in tasks("site2"):
        d = Path(tempfile.mkdtemp(prefix="h2lib-site2-"))
        shutil.copytree(root, d, dirs_exist_ok=True)
        rec = step(t, base, k=k, library=Library(d, KEY), seed=2, reverify=reverify)
        out.append((t, rec))
        shutil.rmtree(d, ignore_errors=True)
    return out


stp(3, f"Six fault reviews at {SITES['chelsea']} — manual limit {MANUAL['chelsea']['approach_limit_k']} K · k=1 · seed 2")
rows = []
for rv in (False, True):
    res = run_site2(1, rv)
    scores = [judge(t, rec.final_run) for t, rec in res]
    rows.append(["re-verified" if rv else "copied unchecked", " / ".join(f"{s:.0f}" for s in scores),
                 f"{sum(scores) / len(scores):.1f}", ", ".join(sorted({f for _, rec in res for f in workspace_facts(rec.final_tree)})) + " K"])
table(rows, ["library facts", "score per task*", "mean*", "limit in the adopted harnesses"])
note("* measured after each step by bench.judge — the loop never saw a score")

stp(4, "k=5 with the facts copied unchecked: list EVERY final in each pool")
res5 = run_site2(5, False)
rows, capped = [], []
for t, rec in res5:
    pool = finals(rec, base)
    sc = [judge(t, r) for _, r in pool]
    fs = sorted({f for tree, _ in pool for f in workspace_facts(tree)})
    every = all(workspace_facts(tree) == ["3.0"] for tree, _ in pool)
    if every and max(sc) < 100:
        capped.append((t, rec, pool))
    rows.append([t.id, t.fixtures["condenser_approach_k"], ", ".join(fs) or "—", "yes" if every else "no",
                 " / ".join(f"{s:.0f}" for s in sc), f"{max(sc):.0f}", f"{sc[selected_index(rec)]:.0f}"])
table(rows, ["task", "approach K", "limits in the pool", "all 5 carry 3.0", "score per final*", "pool best*",
             "delivered*"])
for t, rec, pool in capped:
    warn(f"{t.id}: all five finals carry 3.0 K and the best of them scores {max(judge(t, r) for _, r in pool):.0f}. "
         "The selector cannot deliver a final the pool does not contain")
note("contrast this with Lab 07-1: there the better final was IN the pool. Here no candidate disagrees, so a "
     "reviewer comparing candidates sees agreement, not a problem")

stp(5, "The remedy: a probe that challenges the SHARED assumption (no score, no hidden truth)")
t, rec, pool = capped[0]
print(f"│ {t.id} · request: {t.request}")
print(f"│ the task's own source: \"{t.visible()['sources']['manual']}\"")
rows = []
for j, (tree, _) in enumerate(pool):
    p = premise_probe(tree, t, seed=2 * 100 + j)
    rows.append([f"#{j + 1}", ", ".join(p["facts"]) + " K", f"{p['source']} K", p["with"], p["without"],
                 "yes" if p["load_bearing"] else "no", "FLAG" if p["contradicts"] else "ok"])
table(rows, ["final", "fact asserted", "source says", "diagnosis with fact", "without fact", "carries the decision",
             "vs source"])
note("the fact contradicts the task's own manual in every candidate — a flag that no comparison BETWEEN candidates "
     "could raise, because they all agree")
repaired = []
for j, (tree, _) in enumerate(pool):
    fixed = {p: reverify_facts(txt, t) for p, txt in tree.items()}
    r = execute(fixed, t, seed=2 * 100 + j + 5000)
    repaired.append((workspace_facts(fixed), judge(t, r)))
print(f"│ re-verified pool: limits {', '.join(sorted({f for fs, _ in repaired for f in fs}))} K · score per final* "
      + " / ".join(f"{s:.0f}" for _, s in repaired))
ok("re-verifying the shared fact against the evidence in front of the editor lifts the cap")

stp(6, "Where the defences live (EXAMPLE design for this course)")
table([
    ["re-verify copied facts", "an earlier hotel's threshold is a hypothesis here; check it against this task's sources"],
    ["premise probes", "run a candidate without its shared assumption; a flipped decision means the fact carries it"],
    ["facts in the knowledge graph", "workspace facts go to the graph per site, procedures to skills; memory never authorises"],
], ["defence", "what it does"], maxw=100)
result("a premise every candidate shares caps the pool: selection can only choose among what was proposed")
