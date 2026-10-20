#!/usr/bin/env python3
"""CH 6 · A day in the life — the whole twin, end to end  [ADVANCED]

The finale. Three moments of one operating day at AltoTech Grand Bangkok, each
crossing every layer: the 06:00 morning brief (copilot digests overnight twin
state), the 09:30 VIP arrival (pre-condition inside guardrails), and the 14:00
chiller alarm — detect in the twin, diagnose via graph+TSDB, dispatch the
optimized technician, verify recovery, remember the episode. Then the Week 21
closing message.

Run:  python demos/step05_a_day_in_the_life.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402
from twin import copilot, runtime, world  # noqa: E402


def main() -> None:
    view.banner("CH 6", "A day in the life of the living twin", "ADVANCED")
    view.mode_line()
    rt = runtime.build(rung="supervised")

    # ── 06:00 ──────────────────────────────────────────────────────────────────
    print("━━ 06:00 · MORNING BRIEF — the copilot reads the twin, not a dashboard")
    rt.pilot.query_tsdb(rt.inv["CH-1"].point, 8)
    rt.pilot.query_graph("AHU-3")
    rt.pilot.search_docs("thermostat overrides room")
    print("\n".join(rt.pilot.flush_trace()))
    runtime.narrate("Give the 06:00 morning brief for the hotel twin: overnight zone temps, "
                    "chiller rotation, pre-cool result, room-1203 override count, VIP arrival.",
                    title="copilot · morning brief")
    print()

    # ── 09:30 ──────────────────────────────────────────────────────────────────
    print("━━ 09:30 · VIP ARRIVAL, SUITE 1512 — pre-condition, inside the guardrails")
    rt.pilot.search_docs("precool vip suite")
    print("\n".join(rt.pilot.flush_trace()))
    for step in range(2):
        d = rt.operator.step(hour=9)
        print(f"  step {step + 1}: setpoint → {d.applied} °C  ({d.note})")
    t1512 = rt.stage.find_by_gid(rt.inv["TS-1512"].gid).live["live:temp_c"]
    print(f"  session layer confirms: suite 1512 at {t1512} °C, occupied-VIP flag set —")
    print("  60 min ahead, never below the 21 °C clamp. No Week 23-style policy drama needed:")
    print("  the guardrails make the unsafe action inexpressible.\n")

    # ── 14:00 ──────────────────────────────────────────────────────────────────
    print("━━ 14:00 · CRITICAL — CH-2 low delta-P alarm (every layer, one incident)")
    print("  [DETECT]   alert prim on /World/MEP/Plant/CH_2 — session layer flags low ΔP")
    print("  [DIAGNOSE] the copilot joins the three stores:")
    rt.pilot.query_graph("CH-2")
    rt.pilot.query_tsdb(rt.inv["CH-2"].point, 6)
    rt.pilot.query_tsdb(rt.inv["CHWP-2"].point, 6)
    rt.pilot.search_docs("chiller low delta-p strainer")
    print("\n".join(rt.pilot.flush_trace()))
    runtime.narrate("Diagnose the CH-2 low delta-P chiller alarm from graph, TSDB and O&M "
                    "docs, and recommend the action.", title="copilot · diagnosis")
    wo = rt.pilot.open_work_order("CH-2 strainer", "CRITICAL", "fouled strainer per O&M §4.2")
    print("\n".join(rt.pilot.flush_trace()))

    print("\n  [DISPATCH] greedy + 2-opt (a cuOpt stand-in — same shape, toy scale):")
    routes = copilot.dispatch([("CH-2 strainer", "chiller", 0),
                               ("VAV-05-02 transducer", "hvac", 5),
                               ("TS-1203 calibration", "controls", 12),
                               ("VAV-18-04 damper", "hvac", 18)])
    for tech, r in routes.items():
        stops = " → ".join(f"{a} (fl.{f})" for a, f in r["stops"])
        opt = f"2-opt cut travel {r['floors_before']}→{r['floors_after']} floors" \
            if r["floors_after"] < r["floors_before"] else "already optimal"
        print(f"    {tech:<9} {stops}  · {opt}")

    print("\n  [VERIFY]   30 min later, the twin — not a phone call — confirms recovery:")
    print("    /World/MEP/Plant/CH_2  live:kW back on the daily curve · AHU-3 supply temp stable")
    print("  [REMEMBER] episode → flywheel; next fouled strainer starts from a skill:")
    rt.fly.record("14:00", "incident", "CH-2 low ΔP — fouled strainer",
                  f"{wo['id']} · diagnosed in twin · recovered in 30 min", 0.95)
    rt.fly.log_run("chiller ΔP incident", turns=9, minutes=30.0)
    rt.fly.log_run("chiller ΔP incident", turns=4, minutes=12.0)   # next recurrence, with the skill
    print("\n".join("  " + ln for ln in rt.fly.compound_report()))

    # ── the closing message ────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  DIGITAL TWIN = SCENE + STATE + SIMULATION + AGENTS")
    print("═" * 64)
    print("  SCENE       the USD stage that mirrors the building        (Ch 2 · Apps 2–4)")
    print("  STATE       telemetry bound to identity, three stores      (Ch 3 · App 5)")
    print("  SIMULATION  calibrated physics + guarded surrogates        (Ch 4 · Apps 6–8)")
    print("  AGENTS      guardrailed control + copilot + flywheel       (Ch 5–6 · Apps 9–11)")
    print("\n  Same skeleton, different building:")
    for name, brief in sim.VARIANTS:
        print(f"  • {name:<13} — {brief}")
    print(f"\nTakeaway: the Week 23 fleet had hands; now it has a body. {world.HOTEL} runs")
    print("as one living system — on hardware you own, $0 per token. That's Week 21.")


if __name__ == "__main__":
    main()
