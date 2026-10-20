#!/usr/bin/env python3
"""Lab 05-2 · The confused deputy, and authorising by provenance instead of by the agent's powers.

Offline and instant; always on the SIM brain (a teaching model of a gullible LLM, not a measurement of a real one).

  1. A03 with least agency on: the energy agent is a confused deputy
  2. provenance: every action carries the origins of the text that asked for it; principal_of() keeps the weakest
  3. principal_rule(): one decision table, read row by row
  4. principal_authz on: A04, A02, A03 blocked, and the exact reason each time
  5. egress_allowlist: a second, independent check on the only door out of the hotel (A05, A02)
  6. the limits: a trusted-looking origin (A07's SOP) and a gate that only looks at principals

Everything acts on a simulated hotel with `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/05_tool_misuse_privilege/labs/lab05_2_confused_deputy_and_provenance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import Defenses, Ensemble, principal_of, principal_rule, trust_of  # noqa: E402
from maskit import attack, banner, defend, dim, note, result, step, table, warn  # noqa: E402
from redteam import SCENARIOS, run_scenario  # noqa: E402

BY_ID = {s.id: s for s in SCENARIOS}
SIM = "sim"


def replay(sid: str, d: Defenses):
    s = BY_ID[sid]
    e = Ensemble(d, brain=SIM)
    if s.setup:
        s.setup(e)
    run = e.handle(s.steps[0][1], principal=s.steps[0][0])
    return e, run


banner("Lab 05-2 · the confused deputy and provenance-based authorisation",
       "authorise by WHO the action is ultimately for, not by what the agent can do", brain=False)
dim("🧠 SIM brain · deterministic stand-in for a gullible LLM · this lab always runs on SIM")

step(1, "A03 with least agency on: a confused deputy")
e, run = replay("A03", Defenses(tool_allowlist=True))
run.show()
print(f"✕ harm: {', '.join(e.world.harms())}")
attack("a low-privilege agent (concierge) relayed a valid-looking instruction to a high-privilege one (energy).")
note("energy checked only its OWN powers: 'do I hold set_room_setpoint?' Yes. It never asked 'for whom?'.")

step(2, "provenance: the origins an action carries, and the principal they collapse to")
SETS = [({"guest:1203"}, "A03: the guest's text, relayed by the concierge"),
        ({"staff:duty-manager", "web:reviews.example"}, "A02: staff asked; the page asked for the email"),
        ({"staff:duty-manager", "staff:sop"}, "A07: staff asked; a hotel SOP said the rest"),
        ({"guest:1203", "staff:memory"}, "a guest request + an ungated memory note"),
        ({"staff:duty-manager"}, "B03: the duty manager alone")]
table([[", ".join(sorted(o)), " / ".join(str(trust_of(x)) for x in sorted(o)), principal_of(o), why]
       for o, why in SETS], ["origins", "trust", "principal_of", "where from"], maxw=48)
dim("trust_of: 3 = staff · 2 = a guest (only for their own room) · 1 = external content")
note("an action is only as trusted as the weakest text that caused it. Hops do not launder origins.")

step(3, "principal_rule(): one table, read row by row")
CASES = [("get_room_status", {"room": "1507"}, "guest:1203"),
         ("set_room_setpoint", {"room": "1203", "celsius": 22}, "guest:1203"),
         ("set_room_setpoint", {"room": "1507", "celsius": 26}, "guest:1203"),
         ("set_chiller_setpoint", {"celsius": 7}, "guest:1203"),
         ("send_email", {"to": "ken@mail.example"}, "guest:1203"),
         ("send_email", {"to": "promo@reviews.example"}, "web:reviews.example"),
         ("set_chiller_setpoint", {"celsius": 7.5}, "staff:duty-manager")]
w = Ensemble(brain=SIM).world
table([[t, ", ".join(f"{k}={v}" for k, v in a.items()), p, principal_rule(t, a, p, w) or "✓ allow"]
       for t, a, p in CASES], ["tool", "args", "principal", "decision"], maxw=60)
note("reads are always allowed · staff may do anything their agent can · guests act only on their own room "
     "and address · external content may trigger nothing that writes or leaves the hotel.")

step(4, "principal_authz on: the same attacks")
for sid in ("A04", "A02", "A03"):
    e, run = replay(sid, Defenses(principal_authz=True))
    print(f"── {sid} · {BY_ID[sid].title}")
    run.show()
    print(f"✕ harm: {', '.join(e.world.harms())}" if e.world.harms() else "✓ no harm")
defend("A04: in bounds, right agent, right tool, wrong principal. Only a principal check sees that.")
defend("A02: the staff member asked for a summary; the email was asked for by the page, so it is the page's action.")

step(5, "egress_allowlist: the one door out of the hotel, checked separately")
rows = []
for sid in ("A05", "A02"):
    for name, d in [("principal_authz", Defenses(principal_authz=True)),
                    ("egress_allowlist", Defenses(egress_allowlist=True))]:
        o = run_scenario(BY_ID[sid], d, brain=SIM)
        rows.append([sid, name, "✕ " + o.harms[0] if o.harms else "✓ blocked"])
table(rows, ["id", "control", "outcome"], maxw=50)
note("two independent checks on the same door: if one is misconfigured, the other still holds.")
note("egress only knows WHERE mail goes (hotel domain or a guest's on-file address), not what is in it.")

step(6, "the limits of provenance")
e, run = replay("A07", Defenses(principal_authz=True))
run.show()
print(f"✕ harm: {', '.join(e.world.harms())}" if e.world.harms() else "✓ no harm")
warn("the tampered SOP carries the origin staff:sop, so the chiller change is 'for staff'. Provenance is only as "
     "good as the labels: a poisoned trusted source needs an integrity check (Module 07).")
dim("on a LIVE model every action carries ALL the origins it read (its output cannot be attributed to one input). "
    "A staff request that also read a web page then becomes the page's request: safer, at a utility cost "
    "(the duty manager re-issues the write as a fresh request).")

result("principal_authz asks 'who is this ultimately for?' at the tool, after every hop. That closes the confused "
       "deputy that least agency and bounds leave open.")
