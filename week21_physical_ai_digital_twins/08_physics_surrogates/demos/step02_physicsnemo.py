#!/usr/bin/env python3
"""PART 2 · PhysicsNeMo — the surrogate factory  [INTERMEDIATE]

NVIDIA PhysicsNeMo (renamed from Modulus at GTC, March 2025; Apache-2.0) is the
open-source framework for training physics-ML surrogates: Fourier neural operators,
mesh graph nets, physics-informed losses. It does NOT simulate your building — it
turns the solver runs YOU generate into a millisecond model YOU own.

Run:  python demos/step02_physicsnemo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

try:  # optional REAL upgrade — the demo teaches the same workflow either way
    import physicsnemo  # type: ignore  # noqa: F401
    REAL = True
except Exception:
    REAL = False

TOOLBOX = [
    ("FNO / AFNO",       "Fourier neural operators — learn field→field maps on grids "
                         "(temperature/airflow maps); the workhorse for CFD surrogates"),
    ("MeshGraphNet",     "graph nets on unstructured meshes — irregular geometry like "
                         "a ballroom with columns or a chiller-plant room"),
    ("DoMINO",           "geometry-aware model for large-scale external/internal flows"),
    ("diffusion models", "generative — sample plausible high-resolution fields"),
    ("physicsnemo-sym",  "physics-INFORMED training: add PDE-residual losses so the "
                         "network is penalized for violating the physics"),
]

PIPELINE = [
    ("① generate", "run parametric SOLVER cases offline (CFD / EnergyPlus sweeps)",
     "hours–days, batch"),
    ("② train",    "fit an FNO / MeshGraphNet on those cases (this is PhysicsNeMo)",
     "hours on a GPU"),
    ("③ validate", "check against held-out solver runs; record the input envelope",
     "minutes"),
    ("④ serve",    "real-time inference inside the twin viewport",
     "milliseconds/query"),
]


def main() -> None:
    view.banner("PART 2", "PhysicsNeMo — the surrogate factory", "INTERMEDIATE")
    if REAL:
        print("▣ MODE: REAL — the physicsnemo package is importable in this venv.")
    else:
        print("▣ MODE: SIM — physicsnemo not installed; the workflow below is what it runs.")
    print("  github.com/NVIDIA/physicsnemo · Apache-2.0 · renamed from Modulus (GTC 2025)\n")

    print("What's in the box — data-driven architectures plus physics-informed losses:\n")
    for name, what in TOOLBOX:
        print(f"  • {name}")
        print(f"      {what}")
    print()

    print("The workflow — solver offline, surrogate online:\n")
    for stage, what, cost in PIPELINE:
        print(f"  {stage:<11} {what}")
        print(f"  {'':<11} └─ {cost}")
    print()

    print("Honest framing (read this twice):")
    print("  • PhysicsNeMo is NOT a building-energy simulator. It has no zones, no weather")
    print("    files, no HVAC. YOU generate the training data with a solver you trust,")
    print("    and YOU own the ML lifecycle — retraining, validation, drift.")
    print("  • It wins for FIELD surrogates — temperature/airflow maps over a space —")
    print("    not for schedules or annual kWh (App 6's reduced-order models do those).\n")

    print("What a hotel would train — three field surrogates worth their GPUs:")
    print("  • ballroom airflow — occupancy × AHU config → comfort map, before each event")
    print("  • kitchen exhaust/makeup air — does the smoke plume clear at service peak?")
    print("  • chiller-plant room — thermal map vs pump/fan staging, hot-spot alarms\n")

    print("  🖥️ run it for real (on the DGX / any CUDA box):")
    print("     pip install nvidia-physicsnemo")
    print("     git clone https://github.com/NVIDIA/physicsnemo   # examples/ has FNO recipes\n")

    print("Takeaway: PhysicsNeMo is the factory, not the product — solver runs in, a")
    print("millisecond field model out, and the ML lifecycle is yours. Next: train a")
    print("(tiny) surrogate live, time it, and then break it on purpose.")


if __name__ == "__main__":
    main()
