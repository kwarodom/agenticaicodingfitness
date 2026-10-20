#!/usr/bin/env python3
"""CH 3 · Go live — bind telemetry, join three stores, light up the viewport  [INTERMEDIATE]

The STATE layer (App 5): a mini bus samples every BACnet-ish point, the binding
service writes each value as a SESSION-layer opinion on the prim with the same
GlobalId, and the canonical spatial-temporal query joins GRAPH + TSDB + SCENE.
Then the payoff a screenshot can't fake: a floor-5 heat-map from live prim
attrs, and one anomaly alert prim — room 1203, still misbehaving since Week 23.

Run:  python demos/step02_go_live.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from twin import runtime, telemetry, world  # noqa: E402

DIAGRAM = """\
   BACnet/Modbus ─► gateway ─► MQTT ─► BINDER ──(GlobalId)──► SCENE session layer
                                  │                                (live:* attrs)
                                  ├──────────► TSDB   (24 h history per point)
                                  └──────────► GRAPH  (Brick-ish: feeds/hasPoint/onFloor)
                        one join key, three stores — ask questions none can answer alone
"""


def main() -> None:
    view.banner("CH 3", "Go live — telemetry bound to the scene", "INTERMEDIATE")
    print("▣ MODE: SIM — deterministic mini bus (stdlib, $0). For real: BACnet→MQTT→Kafka,")
    print("  USD session layer / Fabric for high-frequency values (App 5 shows the stack).\n")
    print(DIAGRAM)

    rt = runtime.build(backfill_hours=24)
    bound, dropped = rt.binder.tick(rt.tsdb, 24)   # one more live cycle, on top of backfill
    print(f"▣ BUS CYCLE @ 15:00 — {bound} points bound to session-layer attrs, "
          f"{dropped} dropped")
    print("  dropped = TS-legacy-0902b: no GlobalId, no join key — Ch 2's WARN, now a live hole.")
    p = rt.stage.find_by_gid(rt.inv['TS-1203'].gid)
    print(f"  e.g. {p.path}  live:temp_c = {p.live['live:temp_c']} °C  (a session-layer")
    print("  opinion — the BIM-derived layers underneath were never touched)\n")

    print("▣ THE CANONICAL QUERY — “mean airflow of every VAV fed by AHU-3, floor 5, last 24 h”")
    print("  GRAPH walks AHU-3 ─feeds→ VAV, TSDB aggregates, SCENE returns the prim to highlight:")
    print(f"  {'vav':<11}{'room':<7}{'min':>6}{'mean':>7}{'max':>6}  prim path")
    print("  " + "─" * 70)
    rows = telemetry.spatial_temporal_query(rt.graph, rt.tsdb, rt.stage, rt.inv,
                                            ahu="AHU-3", floor=5, hours=24)
    for r in rows:
        print(f"  {r['vav']:<11}{r['room']:<7}{r['min']:>6}{r['mean']:>7}{r['max']:>6}  {r['path']}")
    print("  (m³/s — no single store could answer this; the GlobalId join is the twin)\n")

    print("▣ VIEWPORT — floor-5 temps, straight from live session-layer attrs")
    cells = []
    for room in ("0501", "0502", "0503", "0504"):
        t = rt.stage.find_by_gid(rt.inv[f"TS-{room}"].gid).live["live:temp_c"]
        cells.append((room, t))
    print("  " + "┬".join(["─" * 14] * 4).join("┌┐"))
    print("  │" + "│".join(f" {r}: {t:>4}°C " for r, t in cells) + "│")
    print("  │" + "│".join(("  " + ("███ warm" if t > 23.5 else "░░░ ok  ") + "    ") for _, t in cells) + "│")
    print("  " + "┴".join(["─" * 14] * 4).join("└┘"))

    r1203 = rt.stage.find_by_gid(rt.inv["TS-1203"].gid)
    delta = round(r1203.live["live:temp_c"] - world.ROOM_STATE["1203"][1], 1)
    print(f"\n▣ ANOMALY PRIM — /World/Tower/Floor_12/1203")
    print(f"  live:temp_c {r1203.live['live:temp_c']} °C vs setpoint 22.0 °C → Δ +{delta} °C, occupied")
    print("  → alert prim spawned in the session layer; 4 overrides this month (the Week 23")
    print("    incident room). The copilot picks this thread up in Ch 6.\n")

    print("run it for real:")
    print("  $ mosquitto_sub -t 'bms/#'            # watch the real bus")
    print("  # NVIDIA-Omniverse/iot-samples — telemetry → USD live/session layer\n")

    print("Takeaway: STATE = telemetry bound to identity. One GUID-less sensor is now an")
    print("invisible room — identity debt becomes blindness. Next: teach the twin physics.")


if __name__ == "__main__":
    main()
