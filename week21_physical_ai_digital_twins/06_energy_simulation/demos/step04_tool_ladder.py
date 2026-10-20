#!/usr/bin/env python3
"""PART 4 · The tool ladder — when each simulator wins  [ADVANCED]

EnergyPlus is one rung, not the whole ladder. Its HVAC is idealized and LOAD-BASED —
perfect for annual energy, but it hides the control dynamics a real building lives in:
hunting, valve authority, staging. This demo climbs the ladder — Spawn, Modelica
Buildings, CDL/Guideline 36, TRNSYS, IES-VE — and ends with a decision table.

Run:  python demos/step04_tool_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

SPAWN = """\
   Spawn of EnergyPlus — the hybrid (LBNL, lbl-srg.github.io/soep/):

     E+ envelope: zones · surfaces · solar  ──compiled behind an FMU──┐
     (the part E+ is genuinely great at)                              ├─► ONE co-simulation
     Modelica: HVAC + controls — real valves, PI loops, staging ──────┘
"""

LADDER = [
    # (tool,                license,      engine/idea,                              wins at)
    ("EnergyPlus",          "free (DOE)", "whole-bldg heat balance, idealized HVAC", "annual energy, design"),
    ("Spawn of EnergyPlus", "free (LBNL)", "E+ envelope FMU + Modelica HVAC/ctrl",   "control dynamics"),
    ("Modelica Buildings",  "free (LBNL)", "component library: Fluid·Zones·OBC·DHC", "HVAC + district detail"),
    ("TRNSYS",              "commercial", "Type 56 multizone, component graph",      "solar thermal, storage"),
    ("IES-VE",              "commercial", "ApacheSim engine + full GUI suite",       "compliance, consultancy"),
]

DECISIONS = [
    ("annual energy · design what-ifs",      "EnergyPlus",             "8760-h heat balance (Ch 2)"),
    ("control dynamics · MPC/RL research",   "Spawn / Modelica",       "real valves, loops, staging"),
    ("real-time twin interactivity",         "5R1C / surrogate",       "milliseconds (Ch 3 · App 8)"),
    ("code compliance · certification",      "IES-VE (ApacheSim)",     "approved workflows + GUI"),
]


def main() -> None:
    view.banner("PART 4", "The tool ladder — when each simulator wins", "ADVANCED")
    view.mode_line()

    print("Why Spawn exists — the gap in EnergyPlus, plainly:")
    print("  E+ HVAC is LOAD-BASED: 'the coil meets the load' — an idealized answer that")
    print("  hides what controllers actually fight: HUNTING (loops oscillating around a")
    print("  setpoint), VALVE AUTHORITY (a badly sized valve that barely modulates), and")
    print("  STAGING (chillers cycling on/off). If your twin's question is about CONTROL,")
    print("  the answer is not in E+.\n")
    print(SPAWN)

    print("The Modelica side, plainly — Buildings Library (LBNL, open source):")
    print("  • Buildings.Fluid        — pipes, pumps, fans, coils, chillers: real flow physics")
    print("  • Buildings.ThermalZones — zone models (incl. the E+ envelope via Spawn)")
    print("  • Buildings.Controls.OBC — control blocks + full ASHRAE Guideline 36 sequences")
    print("  • Buildings.DHC          — district heating & cooling plants and networks\n")

    print("CDL — the control spec that travels:")
    print("  The Control Description Language (standardizing as ASHRAE 231P) writes a")
    print("  control sequence ONCE as composable blocks. Buildings.Controls.OBC ships the")
    print("  full Guideline 36 high-performance sequences in CDL — the SAME spec you can")
    print("  verify in simulation and translate toward real BAS code. No more 'the intent")
    print("  was in a PDF and the programmer improvised'.\n")

    print("The ladder at a glance:\n")
    print(f"  {'tool':<22}{'license':<14}{'engine / idea':<42}wins at")
    print("  " + "─" * 100)
    for tool, lic, engine, wins in LADDER:
        print(f"  {tool:<22}{lic:<14}{engine:<42}{wins}")
    print()

    print("The decision table — one question, one tool:\n")
    print(f"  {'your twin’s question':<38}{'reach for':<24}why")
    print("  " + "─" * 92)
    for q, tool, why in DECISIONS:
        print(f"  {q:<38}{tool:<24}{why}")
    print()

    print("run it for real:")
    print("  $ pip install buildingspy                  # LBNL's Python driver for Modelica")
    print("  $ git clone https://github.com/lbl-srg/modelica-buildings")
    print("  # then simulate with OpenModelica (free) or Dymola (commercial)\n")

    print("Honest note (required reading): Spawn adoption today is research labs and")
    print("advanced consultancies, not mainstream practice. Most firms run plain E+ or")
    print("IES-VE; the Spawn/Modelica/CDL stack is where controls-realistic twins and")
    print("MPC/RL research (App 9) are heading, not where the median project is.\n")

    view.generate(
        "A building digital twin needs both annual energy answers and millisecond "
        "what-if answers for its controllers. In two sentences, why should it keep a "
        "calibrated EnergyPlus model AND a fast reduced-order/surrogate model, rather "
        "than one model for everything?",
        max_tokens=300, title="Two models, one twin — the tool-ladder question")

    print("\nTakeaway: match the tool to the QUESTION — annual energy: E+; control")
    print("dynamics: Spawn/Modelica; real-time twin: 5R1C/surrogate; compliance: IES-VE.")
    print("Next: App 07 — none of their answers are credible until the calibration gate.")


if __name__ == "__main__":
    main()
