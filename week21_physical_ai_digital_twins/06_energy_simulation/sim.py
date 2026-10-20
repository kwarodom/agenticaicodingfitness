#!/usr/bin/env python3
"""Energy-simulation simulator — EnergyPlus concepts + a real 5R1C engine, no installs.

What this module provides (pure stdlib):
  • bangkok_day()      — a hot Bangkok design day, hourly outdoor temp + solar + occupancy
  • simulate_5r1c()    — an ISO 13790-style 5R1C reduced-order zone model (the REAL thing:
                         this is genuine physics, not a mock — demos/step02 builds it live)
  • END_USES           — a mock ABUPS-style annual end-use summary for a Bangkok hotel
  • installed_models() / tok_s() / stream_generate() — the standard Week 21/23 sim exports
"""
from __future__ import annotations

import math
import time

# ── Bangkok design-day weather + schedules ────────────────────────────────────
# Outdoor dry-bulb: sine between 27 °C (~3 am) and 36 °C (~3 pm) — a hot-season day.
# Solar gain through the guest room's 4 m² west window (SHGC applied): 0 at night,
# peaking mid-afternoon. Occupancy: a hotel guest room is occupied 18:00–08:00.


def bangkok_day() -> list[dict]:
    """24 hourly records: {hour, t_out, solar_w, occupied, internal_w}."""
    hours = []
    for h in range(24):
        t_out = 31.5 - 4.5 * math.cos(2 * math.pi * (h - 3) / 24)   # 27 @ 3am → 36 @ 3pm
        # daylight 7:00–19:00, west-facing window peaks ~15:00–16:00
        if 7 <= h <= 18:
            solar = 620.0 * math.sin(math.pi * (h - 7) / 11) ** 2   # W into the zone (SHGC·A)
        else:
            solar = 0.0
        occupied = h >= 18 or h < 8
        internal = (120.0 + 200.0) if occupied else 60.0            # people+TV vs standby, W
        hours.append({"hour": h, "t_out": round(t_out, 1), "solar_w": round(solar),
                      "occupied": occupied, "internal_w": internal})
    return hours


# ── the 5R1C reduced-order model (ISO 13790 style) ────────────────────────────
# Five resistances (given here as conductances H = 1/R, in W/K) and ONE capacitance:
#
#   T_out ──H_ve──┐                        H_ve  ventilation / infiltration air path
#   T_out ──H_w───┤ T_air ──H_is── T_s     H_w   windows (glass has ~no mass)
#                 │                 │      H_is  surface ↔ zone-air film
#   T_out ──H_em──┴──── T_mass ──H_ms      H_em  opaque envelope → mass
#                        (C_m)             H_ms  mass ↔ surface
#
# One 30 m² hotel guest room. Values sized from the ISO 13790 defaults
# (A_tot≈4.5·A_f, A_m≈2.5·A_f, medium-weight construction).
R5C1_PARAMS = {
    "H_ve": 40.0,     # W/K  ventilation + infiltration
    "H_w": 11.0,      # W/K  4 m² window at U≈2.8
    "H_em": 60.0,     # W/K  opaque walls/roof into the mass node
    "H_ms": 680.0,    # W/K  mass ↔ surface (9.1 · A_m)
    "H_is": 465.0,    # W/K  surface ↔ air  (3.45 · A_tot)
    "C_m": 5.0e6,     # J/K  thermal mass (medium: ~165 kJ/m²K · 30 m²)
    "setpoint_c": 24.0,
    "ac_cap_w": 3500.0,   # a typical guest-room fan-coil
}


def simulate_5r1c(weather: list[dict], params: dict | None = None,
                  steps_per_hour: int = 6) -> list[dict]:
    """March the 5R1C network through `weather` (any length, wraps hourly records).

    Returns one record per hour: {hour, t_out, t_air, t_mass, cool_w}.
    Same physics as demos/step02 builds live — kept here so other apps can import it.
    """
    p = {**R5C1_PARAMS, **(params or {})}
    H_oa = p["H_ve"] + p["H_w"]                                  # outdoor → air, direct
    H_ma = 1.0 / (1.0 / p["H_ms"] + 1.0 / p["H_is"])             # mass → air via surface
    dt = 3600.0 / steps_per_hour
    t_mass = 27.0
    out: list[dict] = []
    for rec in weather:
        t_out, cool_w = rec["t_out"], 0.0
        gains_air = 0.5 * rec["solar_w"] + rec["internal_w"]     # half of solar hits air…
        gains_mass = 0.5 * rec["solar_w"]                        # …half is absorbed by mass
        t_air = t_mass
        for _ in range(steps_per_hour):
            # zone air is (nearly) massless → solve its balance algebraically
            t_free = (H_oa * t_out + H_ma * t_mass + gains_air) / (H_oa + H_ma)
            if t_free > p["setpoint_c"]:
                need = (H_oa + H_ma) * (t_free - p["setpoint_c"])   # W to hold setpoint
                cool = min(need, p["ac_cap_w"])
                t_air = t_free - cool / (H_oa + H_ma)
            else:
                cool, t_air = 0.0, t_free
            cool_w = max(cool_w, cool)
            # the ONE capacitance: integrate the mass-node temperature
            q_mass = (p["H_em"] * (t_out - t_mass) + H_ma * (t_air - t_mass) + gains_mass)
            t_mass += q_mass * dt / p["C_m"]
        out.append({"hour": rec["hour"], "t_out": t_out, "t_air": round(t_air, 2),
                    "t_mass": round(t_mass, 2), "cool_w": round(cool_w)})
    return out


def time_annual_run(steps_per_hour: int = 6) -> tuple[float, int]:
    """Run a full 8760-hour year through the 5R1C engine and time it.

    Returns (elapsed_seconds, hours_simulated). This is the punchline of App 6:
    milliseconds for a year — the speed MPC and interactive twins actually need.
    """
    day = bangkok_day()
    year = [{**day[h % 24], "hour": h} for h in range(8760)]
    t0 = time.perf_counter()
    simulate_5r1c(year, steps_per_hour=steps_per_hour)
    return time.perf_counter() - t0, 8760


# ── mock ABUPS end-use summary — a cooling-dominated Bangkok hotel ─────────────
# Shaped like the EnergyPlus HTML tabular report (ABUPS "End Uses" table) for a
# ~12,000 m² hotel. MOCK numbers, sized to a plausible ~185 kWh/m²·yr EUI.
END_USES = [
    # (end use,              MWh/yr, kWh/m²·yr)
    ("Cooling",                999.0, 83.3),
    ("Interior Equipment",     333.0, 27.8),
    ("Interior Lighting",      311.0, 25.9),
    ("Fans",                   266.0, 22.2),
    ("Water Systems (DHW)",    222.0, 18.5),
    ("Pumps",                   89.0,  7.4),
]
FLOOR_AREA_M2 = 12_000


# ── standard sim exports (LLM chapters fall back to these when no endpoint) ───
_MODELS = ["nemotron-3-super:120b-a12b", "nemotron-3-nano:30b-a3b"]
_TOK = {"nemotron-3-super:120b-a12b": 20.0, "nemotron-3-nano:30b-a3b": 54.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated twin engineer] Keep two models and let each do its job: a "
           "calibrated EnergyPlus model (NMBE/CV(RMSE) checked against a year of meter "
           "data — App 7) is the slow source of truth for annual energy and design "
           "what-ifs, while a 5R1C or surrogate model fitted to it answers in "
           "milliseconds for MPC and the interactive twin viewport. An uncalibrated "
           "model of either kind is just a guess with more decimal places.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
