#!/usr/bin/env python3
"""Wire the living twin together: SCENE · STATE · SIMULATION · AGENTS.

build() assembles the whole stack from the packages in this folder; narrate()
is the one place agent REASONING happens — it calls view.generate, so with a
🔌 Connection the reasoning moments run on a REAL endpoint (Ollama / NIM / DGX)
and otherwise stream a faithful canned answer from sim.py. Everything else
(scene, binding, physics, guardrails, dispatch) is real code either way.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import controls, copilot, energy, flywheel, scene, telemetry, world


@dataclass
class Runtime:
    inv: dict                       # SCENE — the inventory …
    stage: scene.Stage              # … and the composed USD-flavored stage
    graph: telemetry.Graph          # STATE — semantics
    tsdb: telemetry.TSDB            # STATE — history
    binder: telemetry.Binder        # STATE — points → session layer
    params: energy.ZoneParams       # SIMULATION — zone model params (calibrate to trust)
    surrogate: energy.Surrogate     # SIMULATION — fast what-if (fit before use)
    operator: controls.Operator     # AGENTS — guardrailed control
    fly: flywheel.Flywheel          # AGENTS — memory + compound returns
    cmms: copilot.CMMS
    pilot: copilot.Copilot          # AGENTS — the staff copilot


def build(backfill_hours: int = 24, rung: str = "shadow") -> Runtime:
    inv = world.build_inventory()
    stage = scene.assemble(inv)
    graph = telemetry.build_graph(inv)
    tsdb = telemetry.TSDB()
    binder = telemetry.Binder(inv, stage)
    if backfill_hours:
        telemetry.backfill(binder, tsdb, backfill_hours)
    cmms = copilot.CMMS()
    return Runtime(inv=inv, stage=stage, graph=graph, tsdb=tsdb, binder=binder,
                   params=energy.ZoneParams(), surrogate=energy.Surrogate(),
                   operator=controls.Operator(controls.Policy(), rung=rung),
                   fly=flywheel.Flywheel(), cmms=cmms,
                   pilot=copilot.Copilot(graph, tsdb, cmms))


def narrate(prompt: str, title: str) -> dict:
    """One agent-reasoning moment. REAL endpoint if connected, else SIM — always $0."""
    import view  # package root is on sys.path (the demos put it there)
    return view.generate(prompt, title=title, max_tokens=220)


if __name__ == "__main__":
    rt = build()
    print(f"{world.HOTEL}: {len(rt.inv)} entities · {len(rt.stage.prims)} prims · "
          f"{len(rt.graph.triples)} triples · {len(rt.tsdb.series)} series · "
          f"operator rung: {rt.operator.rung}")
