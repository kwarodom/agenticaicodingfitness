#!/usr/bin/env python3
"""PART 2 · Live layers & Fabric — where the writes land  [INTERMEDIATE]

NVIDIA's reference pattern (github.com/NVIDIA-Omniverse/iot-samples): an IoT
connector service subscribes to MQTT and writes sensor values as USD attributes
(`iot:temperature`) on a SESSION/LIVE layer — the strongest layer in the stack —
so live data NEVER dirties the authored asset files (App 2's opinion-strength
lesson). And for high-frequency telemetry you don't author USD at all: Fabric
(USDRT), the runtime scene-graph mirror, absorbs the burst and flushes at
checkpoint cadence. This demo streams 20 ticks and shows all three behaviors.

Run:  python demos/step02_live_layers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

try:
    from pxr import Sdf, Usd  # noqa: E402  — real OpenUSD (pip install usd-core)
    REAL = True
except ImportError:
    REAL = False

PRIM = "/Building/Floor5/HVAC/VAV_5_01"
ATTR = "iot:temperature"

LAYERS = """\
   MQTT tick ──► IoT connector / binding service ──► which layer gets the write?

     SESSION / LIVE layer   RAM only, STRONGEST opinion   ← iot:* attrs land HERE
     ops.usda               operations' overrides
     architecture.usda      authored asset layers         ← NEVER touched by telemetry
     mep.usda               (your BIM investment, App 3)

     Fabric (USDRT)         runtime mirror of the stage   ← 10 Hz bursts stop HERE,
                            (no layer, no file, no undo)     flushed at checkpoints
"""


def main() -> None:
    view.banner("PART 2", "Live layers & Fabric — where the writes land", "INTERMEDIATE")
    if REAL:
        print("▣ MODE: REAL — usd-core is installed; the session layer below is a real")
        print("  Usd.Stage session layer, and the authored-layer check is a real diff.\n")
    else:
        print("▣ MODE: SIM — usd-core not installed; layers are modeled faithfully in")
        print("  dicts. `pip install usd-core` flips this demo to REAL (no GPU needed).\n")

    print("The layer stack every live twin uses (strength grows upward):\n")
    print(LAYERS)

    # ── set up the stage: one authored prim, session layer as edit target ──────
    if REAL:
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(PRIM, "Xform")
        authored_before = stage.GetRootLayer().ExportToString()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        prim = stage.GetPrimAtPath(PRIM)

        def write_session(val: float) -> None:
            prim.CreateAttribute(ATTR, Sdf.ValueTypeNames.Double).Set(val)

        def authored_now() -> str:
            return stage.GetRootLayer().ExportToString()
    else:
        authored = {PRIM: {"type": "Xform"}}        # the asset layer, as authored
        session: dict[str, float] = {}              # the session layer (RAM only)
        authored_before = repr(authored)

        def write_session(val: float) -> None:
            session[f"{PRIM}.{ATTR}"] = val

        def authored_now() -> str:
            return repr(authored)

    # ── stream 20 ticks: 1 Hz steady → session layer; 10 Hz burst → Fabric ─────
    print("Streaming 20 ticks (1–10 steady at 1 Hz, 11–20 a 10 Hz burst):\n")
    usd_writes, fabric_absorbed, fabric_flushes = 0, 0, 0
    fabric_latest, next_checkpoint = None, 10.0     # flush cadence: every 1.0 s
    for i, t, point, val in sim.tick_stream(20):
        if i <= 10:                                  # steady: author USD per tick
            write_session(val)
            usd_writes += 1
            print(f"  t={t:>5.1f}s  tick {i:>2}  {point} = {val:<6} → session layer "
                  f"(USD write #{usd_writes})")
        else:                                        # burst: Fabric absorbs it
            fabric_absorbed += 1
            fabric_latest = val
            note = ""
            if t >= next_checkpoint:                 # checkpoint → one USD flush
                write_session(fabric_latest)
                usd_writes += 1
                fabric_flushes += 1
                next_checkpoint += 1.0
                note = f" → CHECKPOINT: flush latest to session layer (USD write #{usd_writes})"
            print(f"  t={t:>5.1f}s  tick {i:>2}  {point} = {val:<6} → Fabric{note}")

    # ── the receipts ────────────────────────────────────────────────────────────
    print("\nThe receipts:")
    print(f"  • authored asset layer untouched: {authored_now() == authored_before}"
          "   ← zero iot:* opinions in your BIM files")
    print(f"  • session layer carries the live value: {PRIM}.{ATTR} is set (RAM only,")
    print("    strongest opinion — delete the session layer and the building is pristine)")
    print(f"  • Fabric absorbed the burst: {fabric_absorbed} ticks → {fabric_flushes} USD flush"
          f" ({fabric_absorbed - fabric_flushes} writes avoided, "
          f"{round((1 - fabric_flushes / fabric_absorbed) * 100)}% less layer churn)\n")

    print("Why it's built this way:")
    print("  • App 2's lesson: layer order = opinion strength. Live values on the session")
    print("    layer OVERRIDE without EDITING — remove them and the authored scene remains.")
    print("  • Authoring USD has composition cost; at 10 Hz × thousands of points it would")
    print("    melt. Fabric mirrors the stage in runtime memory and takes the bursts.\n")

    print("run it for real (NVIDIA's reference IoT→USD connector):")
    print("  $ git clone https://github.com/NVIDIA-Omniverse/iot-samples")
    print("  $ pip install usd-core        # flips THIS demo to REAL, no GPU needed\n")

    print("Takeaway: telemetry goes to the session layer (or Fabric when it's fast),")
    print("never to the authored files. Next: geometry is only ONE of three stores —")
    print("where meaning and history live, and the GlobalId that joins all three.")


if __name__ == "__main__":
    main()
