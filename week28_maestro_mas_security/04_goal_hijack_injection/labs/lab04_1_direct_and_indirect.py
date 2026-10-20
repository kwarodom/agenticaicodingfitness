#!/usr/bin/env python3
"""Lab 04-1 · Direct, indirect and travelling injections: who wrote the instruction, and how it arrived.

Offline and instant; always on the SIM brain (a deterministic stand-in for a gullible LLM, not a measurement of any real
model). Runs three of the course's goal-hijack scenarios against the naive ensemble:

  A01 direct     — the guest's own message carries one extra instruction
  A02 indirect   — a web page the concierge reads carries the instruction (the staff member asked for a summary)
  A03 travelling — the orchestrator forwards the guest's text verbatim; the instruction rides into another agent

For each one it prints the trace, the principal recorded on the harmful action, and what every agent received.
Then it looks the threats up in MAESTRO. Everything acts on a simulated hotel with `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/04_goal_hijack_injection/labs/lab04_1_direct_and_indirect.py
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import maestro  # noqa: E402
from ensemble import Defenses, Ensemble, trust_of  # noqa: E402
from maskit import attack, banner, dim, note, result, step, table  # noqa: E402
from redteam import SCENARIOS  # noqa: E402

BY_ID = {s.id: s for s in SCENARIOS}
CASES = [("A01", "direct", "the guest typed it"),
         ("A02", "indirect", "a web page said it"),
         ("A03", "travelling", "the guest typed it; an agent passed it on")]


def run_case(sid: str):
    s = BY_ID[sid]
    ens = Ensemble(Defenses(), brain="sim")
    if s.setup:
        s.setup(ens)
    principal, text = s.steps[0]
    run = ens.handle(text, principal=principal)
    return s, ens, run


banner("Lab 04-1 · direct, indirect and travelling injections",
       "three toy goal hijacks on a simulated hotel · no defenses yet", brain=False)
dim("🧠 SIM brain · deterministic stand-in for a gullible LLM · this lab always runs on SIM (Lab 04-2 has a LIVE spot check)")

summary = []
for n, (sid, kind, who_wrote) in enumerate(CASES, start=1):
    s, ens, run = run_case(sid)
    principal, text = s.steps[0]
    step(n, f"{sid} · {kind} · {principal} says: {text}")
    attack(s.note or s.title)
    run.show()
    harms = ens.world.harms()
    print(f"✕ harm: {', '.join(harms)}" if harms else "✓ no harm")
    # which audit row did the damage, and whom was it recorded as being for?
    bad = [r for r in ens.audit.rows if r["tool"] in ("set_chiller_setpoint", "send_email", "set_room_setpoint")
           and r["outcome"] == "ok" and not (r["tool"] == "set_room_setpoint" and r["args"].get("room") == "1203")]
    for r in bad:
        note(f"harmful action {r['tool']}({', '.join(f'{k}={v!r}' for k, v in r['args'].items())}) "
             f"by {r['agent']} · recorded principal {r['principal']} (trust {trust_of(r['principal'])})")
    summary.append([sid, kind, who_wrote, bad[0]["principal"] if bad else "—", harms[0] if harms else "none"])

step(4, "A03 hop by hop: what each agent actually received")
s, ens, run = run_case("A03")
table([[e.hop, e.agent, e.text] for e in run.events if e.kind == "recv"], ["hop", "agent", "text it received"], maxw=60)
note("the energy agent never saw the guest's message, only 'set room 1101 to 16' from a colleague.")
note("it has no way to tell a colleague's own idea from a guest's sentence passed along: same channel, same trust.")

step(5, "where these live in MAESTRO (names and descriptions verbatim from the CSA page)")
gm = next(t for t in maestro.layer("L7")["threats"] if t["name"] == "Agent Goal Manipulation")
ae = next(t for t in maestro.layer("L1")["threats"] if t["name"] == "Adversarial Examples")
pat = next(p for p in maestro.patterns() if p["name"] == "Unconstrained Conversational Autonomy")
asi = next(o for o in maestro.owasp() if o["id"] == "ASI01")
for tid, name, desc in [(gm["id"], gm["name"], gm["description"]), (ae["id"], ae["name"], ae["description"]),
                        ("pattern", f"{pat['name']} · threat: {pat['fields']['Threat']}", pat["fields"]["Description"]),
                        (asi["id"], asi["name"], asi["summary"])]:
    print(f"│ {tid:<8} {name}")
    print(textwrap.fill(desc, 100, initial_indent="│          ", subsequent_indent="│          "))
note(f"course mapping (not from either source): ASI01 → {' + '.join(asi['layers'])}")

step(6, "summary")
table(summary, ["id", "kind", "who wrote the instruction", "principal on the harm", "first harm"], maxw=44)
result("all three hijacks worked. Only A02's harmful action is recorded as coming from external content "
       "(web:reviews.example); A01 and A03 look like the guest's own request.")
note("Lab 04-2 tries spotlighting (marks untrusted content as data); Lab 04-3 tries a structured hand-off.")
