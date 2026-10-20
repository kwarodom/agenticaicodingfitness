#!/usr/bin/env python3
"""PART 3 · FMI/FMU co-simulation — models from different tools, one loop  [ADVANCED]

FMI (Functional Mock-up Interface, Modelica Association, fmi-standard.org) is the
neutral standard for packaging a simulation model so ANY tool can run it. An FMU
is a zip: modelDescription.xml + binaries. This demo walks the co-simulation wire
protocol — instantiate → setup_experiment → do_step — between a building FMU and
a controller, printed like a wire trace.

Run:  python demos/step03_fmi_cosim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

try:
    import fmpy  # noqa: F401
    REAL = True
except ImportError:
    REAL = False

ANATOMY = """\
   an FMU is a zip:      building.fmu
                          ├── modelDescription.xml   (variables, GUID, capabilities)
                          └── binaries/…             (the compiled model)
   two flavors:
     Model Exchange (ME)  — FMU exports equations; the IMPORTER supplies the solver
     Co-Simulation  (CS)  — FMU embeds its OWN solver; master just calls do_step()
                            ← what building twins want: E+ and Modelica keep their
                              own numerics and only exchange values at each step
   versions: FMI 2.0 is dominant; FMI 3.0 (2022) adds clocks & array variables.
   runners:  FMPy (pip install fmpy) · PyFMI.   producers: EnergyPlusToFMU (LBNL) ·
             Spawn (native) · Dymola / OpenModelica.
"""


class ControllerFMU:
    """A load-shed supervisor as a second CS 'FMU': 24 °C off-peak, 25 °C on-peak."""

    def __init__(self):
        self.t_set = 24.0

    def instantiate(self) -> str:
        return "fmi2Instantiate('controller', CoSimulation, guid={g36-shed-01}) → handle ok"

    def do_step(self, t: float, dt: float) -> str:
        hour = int(t // 3600) % 24
        self.t_set = 25.0 if sim.PEAK_START <= hour < sim.PEAK_END else 24.0
        return "fmi2OK"


def main() -> None:
    view.banner("PART 3", "FMI/FMU co-simulation — the wire protocol", "ADVANCED")
    if REAL:
        print("▣ MODE: REAL — FMPy importable; it runs actual .fmu files the same way.")
        print("  This chapter drives an in-process FMU-shaped RC model to show the verbs.\n")
    else:
        print("▣ MODE: SIM — FMPy not installed (pip install fmpy). The verb sequence below")
        print("  is exactly what FMPy issues to a real building FMU.\n")
    print(ANATOMY)

    bld, ctl = sim.BuildingFMU(day=100, cop=4.5), ControllerFMU()

    def wire(left: str, result: str = "") -> None:
        print(f"  {left:<58}{result}")

    print("The master algorithm, as a wire trace (setpoint → building; temp/power → master):\n")
    wire("wire", "result")
    print("  " + "─" * 68)
    wire(bld.instantiate("building"))
    wire(ctl.instantiate())
    wire(bld.setup_experiment(0, 24 * 3600))
    kwh = 0.0
    for hour in range(24):
        t = hour * 3600.0
        ctl.do_step(t, 3600)
        show = hour in (0, 8, 9, 13, 21, 22, 23)
        rc_set = bld.set_var("T_set", ctl.t_set)
        rc = bld.do_step(t, 3600)
        tz, pe = bld.get_var("T_zone"), bld.get_var("P_el")
        kwh += pe
        if show:
            wire(f"ctrl.do_step(t={t:>6.0f}, dt=3600) → T_set={ctl.t_set:.1f} °C", "fmi2OK")
            wire(f"building.fmi2SetReal(T_set, {ctl.t_set:.1f})", rc_set)
            wire(f"building.do_step(t={t:>6.0f}, dt=3600)", rc)
            wire(f"building.fmi2GetReal → T_zone={tz:5.2f} °C · P_el={pe:6.1f} kW")
        if hour == 0:
            print("  … (hours 01–07 exchange the same four calls) …")
    print("  " + "─" * 68)
    print(f"  master loop done: 24 do_step() pairs · day total ≈ {kwh:,.0f} kWh electric.\n")
    print("Reading the trace: at 09:00 the controller sheds to 25 °C; by 13:00 the zone")
    print("sits at 29+ °C — the 320 kW plant SATURATES at midday and the zone floats.")
    print("The wire shows it honestly: do_step still returns fmi2OK; physics, not the")
    print("protocol, is what failed. (Ch 4 fixes this with pre-cooling.)\n")

    print("Why this matters for the twin: the calibrated E+ model (App 6), a Modelica")
    print("plant model and your Python controller can each stay in their native tool —")
    print("FMI is the contract that lets one master clock them together.\n")

    print("Honest note on Omniverse: there is NO first-party universal FMI runtime in")
    print("Omniverse. The working pattern is an external co-sim process (FMPy/PyFMI)")
    print("streaming state into USD attributes via App 5's live-binding service.\n")

    print("run it for real:")
    print("  $ pip install fmpy")
    print("  $ fmpy simulate building.fmu --show-plot   # any CS FMU, same verbs\n")

    print("Takeaway: FMI turns simulators into LEGO — do_step() is the whole interface.")
    print("Next: Alfalfa wraps a simulated building behind a BMS-shaped REST API, so your")
    print("agents can run what-ifs the way they'd talk to a real BMS.")


if __name__ == "__main__":
    main()
