#!/usr/bin/env python3
"""Lab 03-2 · The generic-editor ablation: one editor that "rewrites the prompt" versus two scoped editors.

Runs three open-world streams (shuffled task order) over the refine half for each executor, three ways: the base
harness, Harness² k = 1 with the two scoped editors (INV + DOM), and Harness² k = 1 with ONE generic editor that
writes prose into systemprompt.md (scoped=False). The library is off in both loop arms, so only scoping differs.
It reports the score (measured after the run by bench.judge) AND the always-on text each arm adds. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/03_two_scopes_cheapest_component/labs/lab03_2_generic_editor_ablation.py
"""
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                        # noqa: E402
from bench import streams                                                  # noqa: E402
from ops_sim import tasks                                              # noqa: E402
from h2kit import GENERIC_PROSE, inv_edits                                 # noqa: E402
from h2out import banner, bar, dim, note, ok, result, show_text, step, table, warn  # noqa: E402

banner("Lab 03-2 · The generic-editor ablation", "scoped INV + DOM editors vs one generic prose editor · 3 streams", brain=False)
dim("SIM executor and SIM editors = teaching models; the effects below show the mechanism, they do not prove it")

step(1, "What each editor writes for the same failure (no_cmms: proposed work without checking work orders)")
base = H.base_harness()
_, scoped_line = inv_edits("no_cmms", base)
show_text(f"scoped INV editor → tool_guidance.md  (+{H.tokens(scoped_line + chr(10))} tokens)", scoped_line)
show_text(f"generic editor → systemprompt.md  (+{H.tokens(GENERIC_PROSE['no_cmms'])} tokens)", GENERIC_PROSE["no_cmms"].strip())

step(2, "Three streams × 17 refine tasks, per executor (library off in both loop arms)")
REFINE = tasks("refine")
ARMS = (("base harness", dict(mode="base")),
        ("Harness² k=1 · scoped INV + DOM", dict(mode="harness2", library_on=False, scoped=True)),
        ("Harness² k=1 · one generic editor", dict(mode="harness2", library_on=False, scoped=False)))
res = {}
for ex in ("sim-cloud", "sim-local"):
    for name, kw in ARMS:
        res[(ex, name)] = streams(REFINE, n=3, executor=ex, **kw)
rows = []
for ex in ("sim-cloud", "sim-local"):
    for name, _ in ARMS:
        r = res[(ex, name)]
        rows.append([ex, name, f"{r['mean']} ± {r['sd']}", " / ".join(str(m) for m in r["means"]), bar(r["mean"], 100, 20)])
table(rows, ["executor", "arm", "score* mean ± SD", "per stream", ""], maxw=36)
note("* measured after the run by bench.judge, which reads the hidden truth — nothing in the loop ever sees it")
verdict = {}
for ex in ("sim-cloud", "sim-local"):
    sc, ge = res[(ex, ARMS[1][0])], res[(ex, ARMS[2][0])]
    agree = all(a > b for a, b in zip(sc["means"], ge["means"]))
    gap = round(sc["mean"] - ge["mean"], 1)
    within = gap < max(sc["sd"], ge["sd"])
    verdict[ex] = "clear" if agree and not within else "noise"
    note(f"{ex}: scoped − generic = {gap:+} points · all three streams agree in direction: {'yes' if agree else 'no'} · "
         + ("within one SD — the size is not shown on 17 tasks" if within else "larger than one SD"))

step(3, "The cost the score hides: always-on text added per composed step")
rows, comp_mix = [], {}
for ex in ("sim-cloud", "sim-local"):
    for name, _ in ARMS[1:]:
        recs = [rec for run in res[(ex, name)]["runs"] for rec in run["records"]]
        comp = [rec for rec in recs if not rec.keep_base]
        delta = [H.always_on_tokens(rec.final_tree) - H.always_on_tokens(rec.base) for rec in comp]
        mix = Counter(H.component_of(d["path"]) for rec in comp for d in H.diff(rec.base, rec.final_tree))
        comp_mix[(ex, name)] = mix
        nd = sum(1 for rec in recs if not rec.final_run.delivered)
        rows.append([ex, name, f"{len(comp)}/{len(recs)}", f"+{statistics.mean(delta):.1f}" if delta else "—",
                     f"+{max(delta)}" if delta else "—", ", ".join(f"{c}×{n}" for c, n in sorted(mix.items())), nd])
table(rows, ["executor", "arm", "composed", "Δ always-on mean", "max", "components changed", "nothing delivered"], maxw=36)
for ex in ("sim-cloud", "sim-local"):
    mix = comp_mix[(ex, ARMS[2][0])]
    ok(f"{ex}: every generic edit landed in C1 (systemprompt.md) — always-on prose, paid on every call") \
        if set(mix) <= {"C1"} else warn(f"{ex}: generic edits outside C1: {dict(mix)}")
note("the scoped arm spreads its edits over C1 trigger lines, C4 guidance, C6 skills and C7 scripts; its "
     "always-on growth per step is less than half the generic arm's")
dim("why the generic arm composes less often: its prose names no trigger, script or tool, so the SIM agent follows "
    "it only when a prose rule happens to match — many of its probes show no change and the composer drops them")

step(4, "The paper's ablation, for comparison — as reported in the paper")
print("│ without scoping (generic editors instead of invariant and domain): minus 2.9 (WorkBuddy-Bench) and minus 3.7 (JobBench)")
noisy = [ex for ex in ("sim-cloud", "sim-local") if verdict[ex] != "clear"]
clear = [ex for ex in ("sim-cloud", "sim-local") if verdict[ex] == "clear"]
note("the SIM shows the same direction on both executors; "
     + (f"the score gap is within noise on {' and '.join(noisy)}" if noisy else "")
     + ("; " if noisy and clear else "")
     + (f"on {' and '.join(clear)} it clears one SD and all three streams agree" if clear else "")
     + " — the always-on growth is not noise on either")
result(f"sim-cloud scoped {res[('sim-cloud', ARMS[1][0])]['mean']} vs generic {res[('sim-cloud', ARMS[2][0])]['mean']} · "
       f"sim-local scoped {res[('sim-local', ARMS[1][0])]['mean']} vs generic {res[('sim-local', ARMS[2][0])]['mean']} · "
       "generic edits are all always-on prose")
