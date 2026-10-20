#!/usr/bin/env python3
"""CH 6 · A night in the city — the whole twin, end to end  [ADVANCED]

The finale of Capstone II. Three acts of one night in Krung Alto, each crossing
every layer: 23:15 a sideswipe at Asoke (detect → verify → reroute → dispatch →
hospital notified), 01:00 a substation partial trip (district flexibility rides
it through, hospital untouched), 06:00 the morning brief and the flywheel's
consolidation. The hotel twin from Capstone I plays its part as one building in
the district DR fleet.

Run:  python demos/step05_a_night_in_the_city.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from city import agents, grid_mobility as gm, vss  # noqa: E402


def act(title: str) -> None:
    print("┄" * 64)
    print(f"  {title}")
    print("┄" * 64)


def main() -> None:
    view.banner("CH 6", "A night in the city — end to end", "ADVANCED")
    view.mode_line()
    fw = agents.Flywheel()

    act("ACT 1 · 23:15 — sideswipe at Sukhumvit × Asoke (I-11)")
    ev = vss.verify(vss.RAW_EVENTS[1])
    print(f"  23:15:02  RTVI CV ({ev['cam']}) tracks merge+diverge → {ev['type']}")
    print(f"  23:15:11  Alert Verification VLM: {ev['verdict']} — {ev['reason']}")
    print("  23:15:11  corroboration: CAM-14 (anomalous pedestrian movement, same prim)")
    print("  23:15:40  what-if: close S-06, divert S-01/S-02 →")
    for st in gm.reroute_whatif("S-06", ["S-01", "S-02"])[1:]:
        print(f"              {st['name']:<24} v/c {st['vc']}  {st['level']}")
    print("  23:18     ADVISORY: duty engineer approves corridor plan (rung 3)")
    print("  23:19     SUPERVISED: verified alert → ops map · tow + crew dispatched")
    print("  23:20     SUPERVISED: Old Town Hospital notified (bounded template msg)")
    print("  23:52     tracks clear · previous signal plan auto-restored (guardrail)\n")
    fw.record({"kind": "incident", "where": "I-11", "route": "S-01/S-02",
               "reroute_worked": True, "vc_after": 0.89})

    act("ACT 2 · 01:00 — SUB-B partial trip (grid event)")
    trip = gm.substation_trip("SUB-B")
    print(f"  01:00:04  SCADA: {trip['sub']} feeder loss → {trip['lost_mw']} MW must move")
    print("  01:00:04  agent rung check: substation switching = OFFLINE (rung 1) —")
    print("            the agent OBSERVES; grid operators + interlocks act.")
    print(f"  01:00:30  SUPERVISED: DR call to opt-in fleet (hospital hard-excluded)")
    print(f"            covered {trip['covered_mw']} MW of {trip['lost_mw']} MW → "
          f"shortfall {trip['shortfall_mw']} MW")
    print("            • AltoTech Grand Bangkok (App 12 twin): shed 0.5 MW + storage")
    print("            • Sukhumvit Mall 1.1 MW · Tech Park data hall 1.4 MW + storage")
    print("  01:47     SUB-B restored · DR auto-released (≤2 h bound honored)\n")
    fw.record({"kind": "grid", "sub": "SUB-B", "shortfall_mw": trip["shortfall_mw"]})

    act("ACT 3 · 06:00 — morning brief + the flywheel turns")
    new = fw.consolidate()
    print("  consolidation distils tonight into durable memory:")
    for item in new:
        print(f"    + {item}")
    print()
    view.generate(
        "Write the 06:00 duty-engineer morning brief for a smart city: overnight — "
        "verified sideswipe at I-11 23:15 (cleared 23:52, reroute held v/c 0.89), "
        "verified stop-anomaly Old Town 00:00 (towed), SUB-B partial trip 01:00 ridden "
        "through with district flexibility (hospital untouched), 2 false alerts "
        "VLM-filtered before any human saw them. 5 sentences, end with what the "
        "flywheel stored.", max_tokens=260, title="city operator · 06:00 morning brief")
    print()

    print("━" * 64)
    print("  THE WEEK 21 CLOSING MESSAGE, CITY-SIZED")
    print("━" * 64)
    print("""
  Digital Twin = SCENE + STATE + SIMULATION + AGENTS — at every scale:
    a room, a hotel (Capstone I), a district, a city (Capstone II).
  A city twin is a TWIN OF TWINS: buildings reference in with their own
  layers and validators; the city adds eyes (VSS), a street-and-grid
  simulation, and a fleet whose autonomy is earned per action.
  Sovereign the whole way down: video, telemetry and models on GPUs the
  city owns — $0 per token, no data leaves town.
""")


if __name__ == "__main__":
    main()
