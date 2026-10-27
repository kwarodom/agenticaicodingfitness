#!/usr/bin/env python3
"""Lab 08-3 · Cache discipline and the promotion ladder: ship at a boundary, roll back by moving a pointer.

Uses night 2's draft from Lab 08-1 (rebuilt here, SIM). First the prefix hash: which versions change it and what
shipping mid-day costs in cold prefixes. Then the ladder Shadow → Assist → Serve → Write-back with the course's EXAMPLE gates
written as code, measured by bench AFTER the runs (standing in for real approvals, ratings and transcripts). Last, a rollback drill: a pointer change to the parent, nothing deleted.
No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/08_nightly_harness_review/labs/lab08_3_cache_and_ladder.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                              # noqa: E402
import reviewkit as R                                                            # noqa: E402
from bench import checks, judge, promote, safety                                 # noqa: E402
from ops_sim import execute, tasks                                           # noqa: E402
from h2out import banner, dim, edit, note, ok, result, step, table, warn         # noqa: E402

banner("Lab 08-3 · Cache discipline and the promotion ladder", "prefix hash · cold prefixes · gates · rollback", brain=False)
dim("EXAMPLE design for this course on the SIM agent; gate numbers are bench measurements of the SIM")
v1 = H.base_harness()
night = R.nightly_review(R.day(R.BUSY_DAY), v1)
v2 = night.draft
on_demand_only = {**v2, **{p: v1[p] for p in H.ALWAYS_ON}}

step(1, "What the prompt cache keys on: the always-on prefix, not the whole tree")
rows = []
for name, tr in (("v1 (Serve)", v1), ("v2 draft (skills + trigger lines + guidance)", v2),
                 ("v2 with always-on files left as v1", on_demand_only)):
    rows.append([name, H.definition_hash(tr), H.prefix_hash(tr), H.always_on_tokens(tr),
                 "same" if H.prefix_hash(tr) == H.prefix_hash(v1) else "CHANGED"])
table(rows, ["version", "definition hash", "prefix hash", "always-on", "prefix vs v1"])
note("every version has a new definition hash; only an always-on change gives a new prefix hash (a cold cache)")

step(2, "Three approvals in one day: ship each one mid-day, or bundle them at the 02:00 deployment boundary")
drafts = [promote(v1, night.records[:i])[0] for i in (1, 2, 3)]
for i, d in enumerate(drafts, 1):
    note(f"approval {i} at {(9, 13, 17)[i - 1]:02d}:00 · tasks {', '.join(r.task.id for r in night.records[:i])} · prefix {H.prefix_hash(d)}")
mid = R.schedule([(9, drafts[0]), (13, drafts[1]), (17, drafts[2])], v1)
boundary = R.schedule([(2, drafts[2])], v1)                 # the next night's boundary, bundled
COLD_S = 16                                                 # EXAMPLE assumption: a 30,000-token cold prefix ≈ 16 s on a smaller open-weight model on local hardware
rows = []
for name, sch in (("ship when approved (mid-day)", mid), ("bundle at the 02:00 boundary", boundary)):
    cold = R.cold_prefixes(sch, H.prefix_hash(v1))
    busy = [h for h in cold if 8 <= h < 20]
    rows.append([name, len(cold), ", ".join(f"{h:02d}:00" for h in cold), len(busy), len(set(sch)),
                 f"{len(busy) * COLD_S} s"])
table(rows, ["policy", "cold prefixes", "at", "in work hours", "versions seen by today's wakes", "waiting (EXAMPLE)"])
warn(f"mid-day shipping: three cold prefixes while engineers wait, and today's wakes ran on {len(set(mid))} versions, "
     "so no outcome metric can be tied to one version")
ok("at the boundary: one cold prefix, re-warmed by the first routine burst, one version per day")

step(3, "The ladder: draft → Shadow → Assist → Serve → Write-back, each gate in code")
reg = R.Registry(v1)
reg.add_draft("v2", v2, parent="v1")
code, msg = reg.publish("v2", "agent owner, HVAC Expert (fictional)", H.definition_hash(v2))
note(f"publish: {code} {msg}")
print("│ try to skip a rung: " + reg.promote("v2", "serve", {"looks fine": True})[1])
mirror = tasks("refine")                                    # the same task shapes, fresh seeds = new mirrored wakes


def runs(tree, ts, seeds, ex="sim-cloud"):
    return [(t, execute(tree, t, seed=s, executor=ex)) for t in ts for s in seeds]


def mean(xs):
    return round(statistics.mean(xs), 1)


# Shadow → Assist (two weeks mirrored): candidate ≥ base on code graders, zero safety failures, always-on delta in cap
CAP = 100                                                    # EXAMPLE agreed always-on delta (tokens)
sb, sc = runs(v1, mirror, range(50, 64)), runs(v2, mirror, range(50, 64))
base_m, cand_m = mean([judge(t, r) for t, r in sb]), mean([judge(t, r) for t, r in sc])
unsafe = sum(not all(safety(r).values()) for _, r in sc)
delta = H.always_on_tokens(v2) - H.always_on_tokens(v1)
gates = {"candidate ≥ base (mirrored, code graders)": cand_m >= base_m, "zero safety failures": unsafe == 0,
         f"always-on delta ≤ +{CAP}": delta <= CAP}
print(f"│ Shadow, 14 mirrored days × 17 wake shapes · base {base_m} · v2 {cand_m} · unsafe {unsafe} · always-on {delta:+d}")
print("│   → " + reg.promote("v2", "assist", gates)[1])


# Assist → Serve (one team first, A/B by day): the EXAMPLE outcome metrics, here as SIM proxies measured by bench
def outcomes(rs):
    f1 = [(t, r) for t, r in rs if t.family == "fault_review" and t.fixtures["alarms"]]   # an alarm, not a trend check
    rej = [any(not okv for name, okv in checks(t, r) if name in ("no duplicate work order", "proposal routed for approval",
                                                                    "evidence links", "delivered")) for t, r in f1]
    tri = [dict(checks(t, r)) for t, r in rs if t.family == "triage"]
    prec = [c.get("notify/wait decision", False) and c.get("one message per event", False) for c in tri]
    hel = [judge(t, r) == 100 for t, r in rs if t.family == "helper"]
    return round(100 * sum(rej) / len(rej), 1), round(100 * sum(prec) / len(prec), 1), round(100 * sum(hel) / len(hel), 1)


days = range(100, 128)
ab = {"v1 (even days)": runs(v1, mirror, [d for d in days if d % 2 == 0]),
      "v2 (odd days)": runs(v2, mirror, [d for d in days if d % 2 == 1])}
res = {k: outcomes(v) for k, v in ab.items()}
table([[k, f"{a} %", f"{b} %", f"{c} %"] for k, (a, b, c) in res.items()],
      ["Assist, four weeks, A/B by day", "rejection-rate proxy", "notification precision", "helper success"])
(rb, pb, hb), (rc, pc, hc) = res.values()
gates = {"rejection rate ≤ base": rc <= rb, "notification precision ≥ base": pc >= pb, "helper success ≥ base": hc >= hb}
print("│   → " + reg.promote("v2", "serve", gates)[1])

# In Serve: the graduated regression set runs nightly; below 98 % is a rollback trigger (EXAMPLE threshold)
grad = [t for t in mirror if all(judge(t, execute(v1, t, seed=s)) >= 80 for s in range(3))]
nightly = {name: mean([100 * (judge(t, r) >= 80) for t, r in runs(tr, grad, range(200, 210))]) for name, tr in (("v1", v1), ("v2", v2))}
print(f"│ Serve, nightly regression: {len(grad)} graduated tasks ({', '.join(t.id for t in grad)}) × 10 nights · "
      f"v2 {nightly['v2']} % (v1 on the same nights: {nightly['v1']} %)")
trigger = nightly["v2"] < 98

step(4, "Rollback: a pointer change to the parent, plus one cache re-warm")
if trigger:
    warn(f"rollback trigger: regression suite {nightly['v2']} % < 98 % — the rule is code, so it fires")
before = H.prefix_hash(reg.tree("serve"))
parent = reg.rollback("serve")
after = H.prefix_hash(reg.tree("serve"))
note(f"Serve pointer → {parent} · prefix {before} → {after} · v2 still stored: {'v2' in reg.versions} "
     f"(state {reg.versions['v2']['state']}) · cold prefixes caused: {len(R.cold_prefixes([after] * 24, before))}")
print("│ Write-back for v2: " + reg.promote("v2", "write_back", {"regression stable four weeks": not trigger})[1])
warn(f"v1 scores {nightly['v1']} % on the same nights: the SIM executor slips on 6 % of runs whatever the harness "
     "(an assumption in ops_sim), so this 98 % trigger cannot hold for EITHER version — fix the threshold in the "
     "next registration, never mid-flight")

step(5, "What Write-back would have measured: v2 at the second hotel, per executor")
site2 = tasks("site2")
rows = []
for ex in ("sim-cloud", "sim-local"):
    b, c = mean([judge(t, r) for t, r in runs(v1, site2, range(3), ex)]), mean([judge(t, r) for t, r in runs(v2, site2, range(3), ex)])
    rows.append([ex, b, c, f"{c - b:+.1f}"])
table(rows, ["executor at Chelsea (site2)", "v1", "v2", "difference"])
fact = next(ln for ln in v2["skills/chiller_fault_review/SKILL.md"].splitlines() if "In this workspace" in ln)
print(f"│ in skills/chiller_fault_review/SKILL.md: {fact}")
warn("a Harbour View workspace fact rides inside a tenant-class skill: it belongs in the knowledge graph; "
     "transfer must be measured, never assumed")
for a in reg.audit:
    edit(f"audit: {a}")
result(f"Serve → {reg.pointer['serve']} · rollback = one pointer move and one re-warm, no files deleted, no retraining")
