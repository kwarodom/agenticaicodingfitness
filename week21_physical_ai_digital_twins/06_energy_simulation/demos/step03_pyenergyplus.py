#!/usr/bin/env python3
"""PART 3 · pyenergyplus — Python inside the engine  [ADVANCED]

EnergyPlus ships an official Python API in its install directory: pyenergyplus. You
register CALLBACKS at ~15 runtime hook points and E+ calls YOUR code every timestep —
read variables/meters, write actuators, close the loop. This demo shows the real code,
then walks a simulated callback trace including the two gotchas everyone hits.

Run:  python demos/step03_pyenergyplus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

try:
    from pyenergyplus.api import EnergyPlusAPI  # noqa: F401  (ships with EnergyPlus)
    REAL = True
except ImportError:
    REAL = False

CODE = '''\
from pyenergyplus.api import EnergyPlusAPI          # lives in the E+ install dir
api = EnergyPlusAPI()
state = api.state_manager.new_state()               # all calls carry this state handle
h_t = h_act = None

def on_timestep(s):
    global h_t, h_act
    if api.exchange.warmup_flag(s):                 # GOTCHA 1: skip warmup ticks
        return
    if not api.exchange.api_data_fully_ready(s):    # GOTCHA 2: handles are invalid
        return                                      #           before this is True
    if h_t is None:                                 # resolve handles ONCE, then reuse
        h_t = api.exchange.get_variable_handle(
            s, "Zone Mean Air Temperature", "GUESTROOM_501")
        h_act = api.exchange.get_actuator_handle(   # (component_type, control_type, key)
            s, "Zone Temperature Control", "Cooling Setpoint", "GUESTROOM_501")
    t = api.exchange.get_variable_value(s, h_t)
    api.exchange.set_actuator_value(s, h_act, 25.0 if t < 23.5 else 24.0)

api.runtime.callback_begin_system_timestep_before_predictor(state, on_timestep)
api.runtime.run_energyplus(state, ["-w", "BKK.epw", "-d", "out", "model.idf"])
'''

TRACE = [
    ("warmup day 1 · ts 1", "warmup_flag=True", "→ return (ignored)"),
    ("warmup day 1 · ts 2", "warmup_flag=True", "→ return (ignored)"),
    ("warmup day 2 · ts 96", "warmup_flag=True", "→ return (ignored)"),
    ("run day 1 · ts 1", "api_data_fully_ready=True", "resolve handles: h_t=7, h_act=3"),
    ("run day 1 · ts 1", "T_zone = 26.4 °C", "set_actuator(h_act, 24.0)  # cool harder"),
    ("run day 1 · ts 2", "T_zone = 25.1 °C", "set_actuator(h_act, 24.0)"),
    ("run day 1 · ts 3", "T_zone = 23.4 °C", "set_actuator(h_act, 25.0)  # relax"),
    ("… 8760 h later", "run_energyplus returns 0", "your callback ran ~52,000 times"),
]


def main() -> None:
    view.banner("PART 3", "pyenergyplus — Python inside the engine", "ADVANCED")
    if REAL:
        print("▣ MODE: REAL — pyenergyplus importable (EnergyPlus install on sys.path).\n")
    else:
        print("▣ MODE: SIM — pyenergyplus not importable (it ships INSIDE the EnergyPlus")
        print("  install dir, not on PyPI). The code below is exactly what runs there.\n")

    print("The real thing — a supervisory setpoint controller in ~25 lines:\n")
    print(CODE)
    print("The pieces: the STATE MANAGER hands you an opaque state every call; the RUNTIME")
    print("API has ~15 hook points (begin_system_timestep_before_predictor,")
    print("end_zone_timestep_after_zone_reporting, …); the EXCHANGE API reads variables/")
    print("meters (get_variable_handle/value, get_meter_*) and writes actuators.\n")

    print("Simulated callback trace — watch the gotchas play out:\n")
    print(f"  {'engine tick':<22}{'exchange sees':<28}your callback does")
    print("  " + "─" * 76)
    for tick, sees, does in TRACE:
        print(f"  {tick:<22}{sees:<28}{does}")
    print()
    print("The two gotchas, plainly:")
    print("  • Handles are only valid AFTER api_data_fully_ready() — resolve them lazily")
    print("    inside the callback, never at import time.")
    print("  • E+ runs 'warmup' days to settle the thermal mass — skip those ticks or your")
    print("    controller trains on fake weather.\n")

    print("INVERSION OF CONTROL — the big mental shift: you don't step the simulator, the")
    print("simulator calls YOU. A gym-style env.step() therefore needs threads + queues to")
    print("flip the loop inside-out — exactly what Sinergym implements (App 9).\n")

    print("The in-IDF alternative: PythonPlugin:Instance points at a class subclassing")
    print("EnergyPlusPlugin — your Python travels WITH the model file, replacing the")
    print("legacy Erl EMS language.\n")

    print("Takeaway: pyenergyplus turns E+ from a batch tool into a co-simulation partner.")
    print("Next: when E+ itself is the wrong tool — Spawn, Modelica, and the tool ladder.")


if __name__ == "__main__":
    main()
