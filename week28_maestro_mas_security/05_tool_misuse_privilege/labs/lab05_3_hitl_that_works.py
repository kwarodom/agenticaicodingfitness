#!/usr/bin/env python3
"""Lab 05-3 · Human-in-the-loop that works: what the approver must see, and why a rubber stamp stops nothing.

Offline and instant; always on the SIM brain (a teaching model of a gullible LLM, not a measurement of a real one).
The kit's HITL gate (`Defenses(hitl=True)`) sends every CRITICAL tool call (the chiller, parts orders) to a
`Human`: "careful" reads who the request is really for and the value; "rubber_stamp" approves everything.

  1. the same chiller request in front of a careful and a rubber-stamp human (A01, A11)
  2. the approval card: a bad one (agent + tool) vs a good one (principal, origin chain, value, bounds, impact)
  3. what HITL never sees: writes and egress (A04, A05) — and how many approvals one benign day costs
  4. the matrix: the five Module 05 controls × the six Module 04–05 attacks, plus the benign tasks

Everything acts on a simulated hotel with `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/05_tool_misuse_privilege/labs/lab05_3_hitl_that_works.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import CHILLER_BOUNDS, TOOLS, Defenses, Ensemble, Human, trust_of  # noqa: E402
from maskit import attack, banner, defend, dim, note, result, step, table, warn  # noqa: E402
from redteam import SCENARIOS, run_scenario, score  # noqa: E402

BY_ID = {s.id: s for s in SCENARIOS}
SIM = "sim"
TRUST = {3: "staff", 2: "guest (own room only)", 1: "external content"}


def bad_card(agent: str, ask: dict) -> str:
    """What many approval prompts show: the agent and the call. It reads like a colleague asking."""
    args = ", ".join(f"{k}={v}" for k, v in ask["args"].items())
    return f"[Approve?] The {agent} agent wants to run {ask['tool']}({args}).  [Approve] [Decline]"


def good_card(agent: str, ask: dict, chain: list[str]) -> str:
    """What the duty manager needs to decide: for whom, through whom, what value, against what bounds."""
    p, v = ask["principal"], ask["args"].get("celsius")
    lines = [f"┌ APPROVAL · {ask['tool']} · impact: {TOOLS[ask['tool']][0]}",
             f"│ what it does  : {TOOLS[ask['tool']][1]}",
             f"│ requested FOR : {p}  (trust {trust_of(p)} · {TRUST[trust_of(p)]})",
             f"│ path          : {' → '.join(chain)}",
             f"│ executing as  : {agent} agent"]
    if v is not None:
        lo, hi = CHILLER_BOUNDS
        ok_ = lo <= float(v) <= hi
        lines.append(f"│ value         : {v} °C · design bounds {lo:g}–{hi:g} °C · "
                     + ("within bounds" if ok_ else "OUT OF BOUNDS"))
    lines.append("└ default if no answer in 5 min: DECLINE")
    return "\n".join(lines)


def run_with(sid: str, d: Defenses, human: str):
    s = BY_ID[sid]
    e = Ensemble(d, human=Human(human), brain=SIM)
    run = e.handle(s.steps[0][1], principal=s.steps[0][0])
    return e, run


banner("Lab 05-3 · human-in-the-loop that works", "careful vs rubber-stamp approvers · what the card must show",
       brain=False)
dim("🧠 SIM brain · deterministic stand-in for a gullible LLM · this lab always runs on SIM")

step(1, "one chiller request, two humans")
rows = []
for sid in ("A01", "A11"):
    for human in ("careful", "rubber_stamp"):
        e, run = run_with(sid, Defenses(hitl=True), human)
        asked = e.human.asked[0] if e.human.asked else {}
        rows.append([sid, human, f"{asked.get('tool')}({asked.get('args', {}).get('celsius')})", asked.get("principal"),
                     ", ".join(e.world.harms()) or "none"])
table(rows, ["id", "human", "asked to approve", "for", "harm"])
note("the gate is identical in all four rows. Only the human differs. A11 in the suite uses the rubber stamp.")
attack("OWASP ASI09: a human who trusts the agent approves what the agent proposes. The gate becomes a formality.")

step(2, "what the approver was shown")
e, run = run_with("A01", Defenses(hitl=True), "careful")
ask = e.human.asked[0]
chain = [ev.agent for ev in run.events if ev.kind == "recv"]
print("── a bad card")
print(bad_card("energy", ask))
print("── a good card")
print(good_card("energy", ask, ["guest:1203"] + chain))
note("the bad card shows a colleague (the energy agent) asking. The good card shows a GUEST asking for a "
     "whole-hotel change, out of bounds. Same request.")
note("Week 10's LangGraph interrupt() pauses the graph and shows a payload you choose: put the principal in it.")
note("Week 17's auth.md OTP approval binds the yes to a verified human. Neither decides what that human reads.")

step(3, "what HITL never sees, and what a day of approvals costs")
rows = []
for sid in ("A04", "A05"):
    o = run_scenario(BY_ID[sid], Defenses(hitl=True), brain=SIM)
    rows.append([sid, BY_ID[sid].steps[0][1], "✕ " + o.harms[0] if o.harms else "✓ blocked"])
table(rows, ["id", "request", "hitl (careful)"], maxw=50)
warn("HITL gates only CRITICAL tools here. A04 is a write; A05 is egress. Neither reaches the human.")
asked = []
for s in SCENARIOS:
    if s.kind == "benign":
        e = Ensemble(Defenses(hitl=True), human=Human("careful"), brain=SIM)
        for p, t in s.steps:
            e.handle(t, principal=p)
        asked += [(s.id, a["tool"]) for a in e.human.asked]
print(f"◆ benign day (B01–B08): {len(asked)} approval(s) asked: {asked}")
dim("gate every write too and each guest's 'my room is too hot' becomes a page to the duty manager. "
    "That is how approval fatigue (and rubber-stamping) starts.")

step(4, "the matrix: Module 05's controls × the attacks of Modules 04–05")
ATTACKS = ["A01", "A02", "A03", "A04", "A05", "A11"]
BENIGN = [s.id for s in SCENARIOS if s.kind == "benign"]
FIVE = dict(tool_allowlist=True, param_bounds=True, principal_authz=True, hitl=True, egress_allowlist=True)
rows = []
for name, d in [("none", Defenses()), ("tool_allowlist", Defenses(tool_allowlist=True)),
                ("param_bounds", Defenses(param_bounds=True)), ("principal_authz", Defenses(principal_authz=True)),
                ("hitl (careful human)", Defenses(hitl=True)), ("egress_allowlist", Defenses(egress_allowlist=True)),
                ("all five", Defenses(**FIVE))]:
    outs = [run_scenario(BY_ID[i], d, brain=SIM) for i in ATTACKS + BENIGN]
    sc = score(outs)
    rows.append([name, *["✓" if not o.harms else "·" for o in outs[:len(ATTACKS)]],
                 f"{sc['attacks'] - sc['succeeded']}/{sc['attacks']}", f"{sc['done']}/{sc['benign']}"])
table(rows, ["control", *ATTACKS, "blocked", "benign done"])
dim("✓ = blocked · · = harm. A11 always runs with the rubber-stamp human (that is the scenario).")
defend("principal_authz carries the most weight; bounds, egress and a careful human are independent backstops.")

result("a HITL gate is as good as what the human checks. Show the real principal, the path, the value and the "
       "bounds; default to decline; and gate only what is critical, so the human still reads each card.")
