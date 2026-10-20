#!/usr/bin/env python3
"""Lab 09-2 · One defense at a time: what each control stops, and what each control costs.

Offline, about a second. The lab pins the SIM brain.

  1. the single-defense × attack matrix: turn on ONE switch, run the 12 attacks, mark what it blocked
  2. what the matrix teaches: no single control is enough, and some attacks have only one blocker
  3. the cost of each control on the suite's 8 benign tasks: tasks broken, denials, human approvals asked
  4. the suite's blind spot: three ordinary tasks it does not contain (scorekit.PROBES), and what breaks them

Everything is simulated: a fake hotel, fake guests, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_2_defense_matrix_and_cost.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble import Defenses  # noqa: E402
from maskit import banner, dim, fail, note, ok, result, runs_dir, step, table, warn  # noqa: E402
from redteam import SCENARIOS  # noqa: E402
from scorekit import PROBES, SWITCHES, defenses_of, run_detailed, run_probe  # noqa: E402

banner("Lab 09-2 · defense matrix and cost", "one switch at a time · what it stops · what it breaks", brain=False)
dim("this lab pins the 🧠 SIM brain: a deterministic teaching model, not a measurement of any real LLM")
ATTACKS = [s for s in SCENARIOS if s.kind == "attack"]
BENIGN = [s for s in SCENARIOS if s.kind == "benign"]

step(1, "the single-defense × attack matrix (✓ = this switch alone blocked the attack)")
blocked = {}
print("│ " + "defense".ljust(17) + " ".join(s.id for s in ATTACKS) + "  | stops")
for name in SWITCHES:
    d = defenses_of([name])
    blocked[name] = {s.id for s in ATTACKS if not run_detailed(s, d)[0].harms}
    marks = " ".join(" ✓ " if s.id in blocked[name] else " · " for s in ATTACKS)
    print(f"│ {name:<17}{marks}  | {len(blocked[name])}")

step(2, "what the matrix teaches")
by_attack = {s.id: [n for n in SWITCHES if s.id in blocked[n]] for s in ATTACKS}
table([[a, len(ns), ", ".join(ns)] for a, ns in by_attack.items()], ["attack", "blockers", "which switches"], maxw=72)
best = max(SWITCHES, key=lambda n: len(blocked[n]))
result(f"best single control: {best} ({len(blocked[best])}/12) — no switch alone gets to 12")
single = [a for a, ns in by_attack.items() if len(ns) == 1]
warn(f"attacks with exactly ONE blocker even when every switch is on: {', '.join(single)} — single points of failure")
note("tool_allowlist alone blocks nothing: each attack is routed to the agent that does hold the tool (Module 05)")
note("hitl alone blocks A01 but not A11: the same request, approved by a rubber-stamp human")

step(3, "the cost of each control on the suite's 8 benign tasks")
rows, cost = [], {}
for name in ["(none)"] + SWITCHES + ["ALL"]:
    d = Defenses() if name == "(none)" else Defenses.all() if name == "ALL" else defenses_of([name])
    broken = denials = asked = 0
    for s in BENIGN:
        o, e = run_detailed(s, d)
        broken += 0 if o.done else 1
        denials += o.denials
        asked += len(e.human.asked)
    atk_asked = sum(len(run_detailed(s, d)[1].human.asked) for s in ATTACKS)
    cost[name] = {"broken": broken, "denials": denials, "asked": asked}
    rows.append([name, f"{8 - broken}/8", denials, asked, atk_asked])
table(rows, ["defense", "benign done", "benign denials", "approvals asked (benign)", "approvals asked (attacks)"])
note("on the suite's own benign tasks, no control costs a single task: every cost here is a human approval")
note("HITL asks the duty manager for every critical action, even the staff's own (B03 chiller, B04 order): "
     "that is the approval load that turns a careful human into a rubber stamp (A11)")

step(4, "the blind spot: three ordinary tasks the suite does not contain")
for p in PROBES:
    print(f"│ {p.id}  {p.title:<56} {p.steps[0][0]}: {p.steps[0][1]}")
rows, broke = [], {}
for name in ["(none)"] + SWITCHES + ["ALL"]:
    d = Defenses() if name == "(none)" else Defenses.all() if name == "ALL" else defenses_of([name])
    res = [run_probe(p, d)[0] for p in PROBES]
    broke[name] = [p.id for p, r in zip(PROBES, res) if not r]
    if broke[name] or name in ("(none)", "ALL"):
        rows.append([name] + ["✓ done" if r else "✕ broken" for r in res])
table(rows, ["defense (only rows that break something)"] + [p.id for p in PROBES])
for p in PROBES:
    who = [n for n in SWITCHES if p.id in broke[n]]
    why = {"P01": "the outside contractor is not on the egress allow-list: a policy decision, not a bug",
           "P02": "forward_summary forwards ONE task; tool_allowlist stops energy from opening the work order",
           "P03": "the router sends 'check room' to the concierge, which has no get_room_status: a routing bug "
                  "that least agency exposed"}[p.id]
    fail(f"{p.id} broken by {', '.join(who)} — {why}")
result(f"with every control on: the suite says utility 8/8, the probes say {3 - len(broke['ALL'])}/3")
dim("utility is measured on YOUR benign set. Add a probe for every complaint from the front desk.")

out = runs_dir(__file__) / "matrix.json"
out.write_text(json.dumps({"blocked": {k: sorted(v) for k, v in blocked.items()}, "cost": cost,
                           "probes_broken": broke}, indent=2), encoding="utf-8")
ok(f"saved the matrix and the costs to {out.relative_to(out.parents[2])} (Lab 09-3 and the capstone read it)")
