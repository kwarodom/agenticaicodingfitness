#!/usr/bin/env python3
"""Lab 09-3 · The smallest defense set that stops every attack — and why smallest is not best.

Offline. The exhaustive search over all 8192 switch combinations takes well under a minute on this Mac
(the lab measures and prints the time). The lab pins the SIM brain.

  1. greedy: add the switch that stops the most remaining attacks, until ASR 0 and utility 100 %
  2. exhaustive: try all 2^13 = 8192 combinations (spotlight on = "nonce"), with early exit
  3. the minimal sets against the three probes of Lab 09-2
  4. smallest is not best: knock one control out of a minimal set and out of the full set, count what reopens

Everything is simulated: a fake hotel, fake guests, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_3_minimal_set.py
"""
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maskit import banner, bar, dim, note, ok, result, step, table, warn  # noqa: E402
from redteam import SCENARIOS, run_scenario, run_suite, score  # noqa: E402
from scorekit import PROBES, SWITCHES, defenses_of, run_probe  # noqa: E402

banner("Lab 09-3 · the minimal defense set", "greedy · exhaustive over 8192 combinations · knock-out test", brain=False)
dim("this lab pins the 🧠 SIM brain: a deterministic teaching model, not a measurement of any real LLM")
ATTACKS = [s for s in SCENARIOS if s.kind == "attack"]
BENIGN = [s for s in SCENARIOS if s.kind == "benign"]


def short(names) -> str:
    return " + ".join(names) or "(none)"


step(1, "greedy: add the switch that stops the most remaining attacks")
chosen = []
while True:
    sc = score(run_suite(defenses_of(chosen)))
    print(f"│ {short(chosen):<84} ASR {sc['succeeded']:>2}/12 · utility {sc['done']}/8")
    if sc["succeeded"] == 0 and sc["done"] == 8:
        break
    trial = {n: score(run_suite(defenses_of(chosen + [n]))) for n in SWITCHES if n not in chosen}
    chosen.append(min(trial, key=lambda n: (trial[n]["succeeded"], -trial[n]["done"], SWITCHES.index(n))))
result(f"greedy stops at {len(chosen)} switches: {short(chosen)}")
dim("greedy is fast but not guaranteed minimal: a locally best pick can force extra picks later")

step(2, "exhaustive: all 2^13 = 8192 combinations, with early exit")
single = {n: {s.id for s in ATTACKS if not run_scenario(s, defenses_of([n])).harms} for n in SWITCHES}
order = sorted(ATTACKS, key=lambda s: sum(s.id in v for v in single.values())) + BENIGN   # hardest attacks first
t0, runs, passing = time.time(), 0, []
for mask in range(1 << len(SWITCHES)):
    names = [SWITCHES[i] for i in range(len(SWITCHES)) if mask >> i & 1]
    d = defenses_of(names)
    for s in order:
        o = run_scenario(s, d)
        runs += 1
        if (s.kind == "attack" and o.harms) or (s.kind == "benign" and not o.done):
            break
    else:
        passing.append(names)
secs = time.time() - t0
full_grid = (1 << len(SWITCHES)) * len(SCENARIOS)
result(f"{1 << len(SWITCHES)} combinations in {secs:.1f} s · {runs:,} scenario runs instead of {full_grid:,} "
       f"(early exit) · {secs / runs * 1000:.2f} ms per run")
k = min(len(p) for p in passing)
minimal = [p for p in passing if len(p) == k]
ok(f"{len(passing)} combinations reach ASR 0 % with utility 100 % · the smallest have {k} switches · "
   f"there are {len(minimal)} of them")
table([[i + 1, short(m)] for i, m in enumerate(minimal)], ["#", f"minimal set ({k} switches)"], maxw=100)
sizes = {}
for p in passing:
    sizes[len(p)] = sizes.get(len(p), 0) + 1
for n in sorted(sizes):
    print(f"│ size {n:>2}: {sizes[n]:>4} passing sets {bar(sizes[n], max(sizes.values()), 30)}")
core = [n for n in SWITCHES if all(n in m for m in minimal)]
note(f"in every minimal set: {', '.join(core)} — the only blockers of A04, A09 and A12 (Lab 09-2, step 2)")

step(3, "the minimal sets against the three probes (ordinary tasks the suite does not contain)")
rows = []
for i, m in enumerate(minimal):
    res = [run_probe(p, defenses_of(m))[0] for p in PROBES]
    rows.append([i + 1, short(m)] + ["✓" if r else "✕" for r in res])
table(rows, ["#", "minimal set", *[p.id for p in PROBES]], maxw=90)
keep = next(m for m in minimal if all(run_probe(p, defenses_of(m))[0] for p in PROBES))
note(f"set #{minimal.index(keep) + 1} passes the suite AND all three probes: {short(keep)}")

step(4, "smallest is not best: knock one control out and count the attacks that come back")


def knock_out(names):
    out = {}
    for n in names:
        rest = [x for x in names if x != n]
        out[n] = [s.id for s in ATTACKS if run_scenario(s, defenses_of(rest)).harms]
    return out


ko_min, ko_all = knock_out(keep), knock_out(SWITCHES)
table([[n, ", ".join(ko_min[n]) or "—"] for n in keep], [f"minimal set #{minimal.index(keep) + 1}: remove…",
                                                          "attacks that succeed again"], maxw=60)
table([[n, ", ".join(ko_all[n]) or "—"] for n in SWITCHES], ["all 13 on: remove…", "attacks that succeed again"],
      maxw=60)
fragile_min = sum(1 for v in ko_min.values() if v)
fragile_all = sum(1 for v in ko_all.values() if v)
result(f"minimal set: {fragile_min}/{len(keep)} single failures reopen an attack "
       f"({sum(len(v) for v in ko_min.values())} attack-reopenings) · all on: {fragile_all}/13 "
       f"({sum(len(v) for v in ko_all.values())})")
warn("in a minimal set every control is the LAST blocker of some attack: one bug, one misconfiguration, one bypass, "
     "and that attack is back")
note("defense in depth = keep overlap on purpose. Even with all 13 on, A04, A09 and A12 still have one blocker each")
pairs = sum(1 for a, b in combinations(keep, 2)
            if any(run_scenario(s, defenses_of([x for x in keep if x not in (a, b)])).harms for s in ATTACKS))
dim(f"two controls failing together: {pairs} of the {len(keep) * (len(keep) - 1) // 2} pairs in the minimal set "
    "reopen an attack")
