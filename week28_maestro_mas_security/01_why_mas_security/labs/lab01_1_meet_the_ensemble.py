#!/usr/bin/env python3
"""Lab 01-1 · Meet the Hotel Ops Ensemble: who can do what, for whom, and where the trust boundaries are.

Offline and instant on the SIM brain. Prints the cast (agents and their tools), the kinds of principal a request
can come from, then traces two harmless requests end to end and counts every trust boundary they cross:
a guest who is too hot, and a duty manager who orders parts from an outside vendor agent over A2A.

Everything is simulated: a fake hotel, fake guests, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/01_why_mas_security/labs/lab01_1_meet_the_ensemble.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import ROLE_TOOLS, TOOLS, Ensemble, trust_of  # noqa: E402
from maskit import banner, note, ok, result, step, table  # noqa: E402

banner("Lab 01-1 · meet the Hotel Ops Ensemble", "four agents · one external vendor · a simulated hotel")

step(1, "the cast: every agent and the tools it holds")
rows = []
for agent, tools in ROLE_TOOLS.items():
    if not tools:
        rows.append([agent, "(none — it only routes requests)", "—"])
        continue
    worst = max((TOOLS[t][0] for t in tools), key=["read", "personal", "write", "egress", "critical"].index)
    rows.append([agent, ", ".join(sorted(tools)), worst])
rows.append(["vendor (external)", "answers parts orders over A2A", "outside our control"])
table(rows, ["agent", "tools", "riskiest tool impact"], maxw=70)
note("energy holds the two tools that change the building: set_room_setpoint and set_chiller_setpoint (whole hotel).")

step(2, "who a request can come from — and how much we trust them")
table([[o, trust_of(o), w] for o, w in [
    ("staff:duty-manager", "staff — may run the hotel"),
    ("guest:1203", "a guest — a real principal, but only for their own room"),
    ("web:reviews.example", "external content the concierge read"),
    ("vendor:https://a2a.acme-hvac.example", "another organisation's agent"),
]], ["origin", "trust (3 high … 1 low)", "meaning"])

step(3, "trace 1 · a guest says: Room 1203 is too hot.")
ens = Ensemble()
run = ens.handle("Room 1203 is too hot.", principal="guest:1203")
run.show()
crossed = ["guest → orchestrator (the request enters the system)",
           "orchestrator → energy (a hand-off between agents)",
           "energy → set_room_setpoint (an agent touches the building)"]
for c in crossed:
    print(f"│ boundary: {c}")
ok(f"room 1203 setpoint is now {ens.world.rooms['1203']['setpoint']:g} °C · harms: {ens.world.harms() or 'none'}")

step(4, "trace 2 · the duty manager says: Order 2 compressor relays from Acme HVAC Parts.")
ens = Ensemble()
run = ens.handle("Order 2 compressor relays from Acme HVAC Parts.", principal="staff:duty-manager")
run.show()
for c in ["staff → orchestrator", "orchestrator → maintenance",
          "maintenance → agent directory (which card is 'Acme'?)",
          "maintenance → vendor agent over A2A (another organisation)",
          "vendor reply → maintenance (text we did not write, back in an agent's context)"]:
    print(f"│ boundary: {c}")
o = ens.world.orders[-1]
ok(f"order {o['order_id']} sent to {o['endpoint']} · verified vendor: {o['verified_vendor']}")

result("Two ordinary requests crossed 3 and 5 trust boundaries. Each boundary is a place where text written by "
       "someone else becomes an instruction for an agent — that is what this week secures.")
