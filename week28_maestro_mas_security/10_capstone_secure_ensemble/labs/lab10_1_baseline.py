#!/usr/bin/env python3
"""Lab 10-1 · Where we start: the naive scorecard, the inherent risk heatmap, and the price of every control.

MAESTRO steps 1–4 on the design-time threat model (data/threat_model.base.yaml): what the system is made of,
how risky each threat is before any control, and what the CI gate says about that. Then the red-team suite on
the naive ensemble and on the previous team's plan, and the planning table you will use in lab 10-2: which single
switch stops which attack on its own, and what each switch costs to run (the course's price list).

Always the SIM brain (it runs 20 scenarios × 15 configurations); every number is from a teaching model on a
simulated hotel. Offline, about one second.

Run: .venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_1_baseline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capkit as ck  # noqa: E402
import maestro  # noqa: E402
from ensemble import Defenses  # noqa: E402
from maskit import attack, banner, dim, note, result, step, table  # noqa: E402
from redteam import scorecard  # noqa: E402

t0 = time.time()
banner("Lab 10-1 · where we start", "the Hotel Ops Ensemble before you harden it · MAESTRO steps 1–4")
dim("this lab always runs the SIM brain: 20 scenarios × 15 configurations is too many model calls for LIVE")

tm, errs = ck.load_tm(ck.BASE_TM)
threats = tm["threats"]

step(1, "System Decomposition — the design-time model, by MAESTRO layer")
rows = []
for lid in maestro.LAYER_ORDER:
    comps = [c["id"] for c in tm["components"] if c["layer"] == lid]
    lead = [t["id"] for t in threats if t["layers"][0] == lid]
    rows.append([lid, maestro.layer(lid)["name"], ", ".join(comps), ", ".join(lead) or "—"])
table(rows, ["layer", "name", "components", "threats led by it"], maxw=56)
note(f"{len(tm['components'])} components · {len(threats)} threats · validate(): {'clean' if not errs else errs}")

step(2, "Risk Assessment — inherent risk, before any control (the course's 1–5 matrix)")
rank = sorted(threats, key=lambda t: -maestro.risk(t["likelihood"], t["impact"])[0])
table([[t["id"], t["title"], "/".join(t["layers"]), f"{t['likelihood']}×{t['impact']}",
        "{} {}".format(*maestro.risk(t["likelihood"], t["impact"])), t.get("scenario", "—")] for t in rank],
      ["id", "threat", "layers", "L×I", "risk", "tested by"], maxw=62)
print(ck.heatmap(threats))
crit = [t["id"] for t in threats if ck.is_critical_impact(t)]
note(f"critical impact (impact 5): {', '.join(crit)} — each will need two independent controls")

step(3, "the CI gate on the design-time model: fail on high")
code, text = ck.gate_text(tm, "high")
print(text)
result(f"gate exit code {code}: every threat is still open, so the model blocks a release")

step(4, "the red-team suite on the naive ensemble (no defenses)")
attack("12 toy attacks on a simulated hotel · 8 ordinary tasks that must keep working")
naive = scorecard(ck.suite(Defenses()), "no defenses · SIM brain")

step(5, "the previous team's plan: least agency + a human approver who approves everything")
prev =Defenses(tool_allowlist=True, hitl=True)
sc = ck.score(ck.suite(prev, "rubber_stamp"))
sc2 = ck.score(ck.suite(prev, "careful"))
table([["tool_allowlist + hitl", "rubber_stamp", f"{sc['succeeded']}/12", f"{sc['done']}/8", ck.cost(prev)[0]],
       ["tool_allowlist + hitl", "careful", f"{sc2['succeeded']}/12", f"{sc2['done']}/8", ck.cost(prev)[0]]],
      ["switches", "approver", "attacks that harm", "benign done", "cost"])
note("least agency stops nothing here: the attacker asks an agent that does hold the tool (Module 05)")
note("HITL is exactly as good as the person behind it (Module 05, A11)")

step(6, "the planning table: which single switch stops each attack on its own (careful approver)")
mapped = {t.get("scenario"): t["id"] for t in threats}
rows = []
for sid in ck.ATTACKS:
    b = ck.blockers(sid, "careful")
    rows.append([sid, ck.BY_ID[sid].title, mapped.get(sid, "— none"),
                 ", ".join(f"{n}({ck.COSTS[n][0]})" for n in b), len(b)])
table(rows, ["id", "attack", "threat", "switches that stop it alone (cost)", "#"], maxw=64)
broken = [n for n in ck.SWITCHES if ck.score(ck.suite(ck.single(n))).get("done") < 8]
note(f"single switches that break a benign task: {', '.join(broken) or 'none'}")

step(7, "the price list (the course's estimate of what each control costs to run)")
ck.show_costs()
result(f"naive ensemble: {naive['succeeded']}/12 attacks harm, {naive['done']}/8 tasks done · "
       f"your job: 0/12 and 8/8 within {ck.BUDGET} points · {time.time() - t0:.1f}s")
