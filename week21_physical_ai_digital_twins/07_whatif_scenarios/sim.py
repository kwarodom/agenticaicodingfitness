#!/usr/bin/env python3
"""What-if simulator — a toy RC hotel, a THB tariff, mock meter data, mini-Alfalfa.

Everything the four demos need, pure stdlib and deterministic:

  • a 5R1C-style reduced-order model of a Bangkok hotel's cooled public zone
    (lumped R, lumped C — the same family of model App 6 builds, kept tiny here)
  • Bangkok weather + hotel internal/solar gains, hour by hour
  • a Thai-style TOU tariff (on-peak 09:00–22:00 weekdays)
  • 12 months of mock METERED data (the "real building") for the G14 gate
  • AlfalfaLite — an in-process mimic of Alfalfa's BMS-shaped point API
  • the standard installed_models() / tok_s() / stream_generate() trio for view.py
"""
from __future__ import annotations

import math
import time

# ── the toy building: one lumped cooled zone of a 180-room Bangkok hotel ──────
R_ENV = 0.06        # °C per kW — envelope + ventilation resistance (lumped)
C_MASS = 55.0       # kWh per °C — effective thermal capacitance (slab + walls)
CAP_KW = 320.0      # chiller plant capacity, kW thermal
COMFORT_C = 25.5    # guests complain above this zone temperature
FILM_SOLAR_CUT = 0.35   # window film keeps 35 % of solar gain out

# hotel internal gains by hour of day, kW (people + lights + plug + kitchen)
GAINS_KW = (55, 52, 50, 50, 52, 58, 70, 85, 95, 105, 110, 115,
            118, 118, 115, 110, 105, 110, 120, 125, 115, 95, 75, 62)

# Thai-style TOU tariff, THB per kWh (on-peak 09:00–22:00 Mon–Fri)
TARIFF_ON, TARIFF_OFF = 4.3, 2.6
PEAK_START, PEAK_END = 9, 22


def tariff(day: int, hour: int) -> float:
    weekday = (day % 7) < 5
    return TARIFF_ON if (weekday and PEAK_START <= hour < PEAK_END) else TARIFF_OFF


def weather(day: int, hour: int) -> tuple[float, float]:
    """(outdoor °C, solar gain kW) for a Bangkok day-of-year + hour-of-day."""
    seasonal = 2.2 * math.sin(2 * math.pi * (day - 14) / 365)     # hottest ~mid-April
    t_out = 28.8 + seasonal + 3.6 * math.cos(2 * math.pi * (hour - 15) / 24)
    sun = max(0.0, math.sin(math.pi * (hour - 6) / 12))            # daylight 06–18
    solar = 150.0 * sun * (1 + 0.10 * seasonal / 2.2)
    return t_out, solar


def _step(t_in: float, t_out: float, solar: float, gains: float,
          setpoint: float, cop: float) -> tuple[float, float]:
    """One 1-hour step of the RC zone. Returns (new zone °C, electric kW)."""
    load = (t_out - t_in) / R_ENV + solar + gains                  # kW thermal
    q_need = load + C_MASS * (t_in - setpoint)                     # hold/reach setpoint
    q = min(max(q_need, 0.0), CAP_KW)                              # chiller limits
    t_next = t_in + (load - q) / C_MASS
    return t_next, q / cop


def simulate(cop: float, setpoint: float, film: bool = False, *, days: int = 365,
             gains_scale: float = 1.0, solar_scale: float = 1.0) -> dict:
    """Annual run. Returns kWh, THB cost, comfort-violation hours, peak kW, monthly kWh."""
    t_in, kwh, cost, viol, peak = setpoint, 0.0, 0.0, 0, 0.0
    monthly = [0.0] * 12
    for day in range(days):
        for hour in range(24):
            t_out, solar = weather(day, hour)
            solar *= (1 - FILM_SOLAR_CUT if film else 1.0) * solar_scale
            t_in, elec = _step(t_in, t_out, solar, GAINS_KW[hour] * gains_scale,
                               setpoint, cop)
            kwh += elec
            cost += elec * tariff(day, hour)
            peak = max(peak, elec)
            if t_in > COMFORT_C:
                viol += 1
            monthly[min(11, int(day / 30.44))] += elec
    return {"kwh": kwh, "cost_thb": cost, "comfort_viol_h": viol,
            "peak_kw": peak, "monthly_kwh": monthly}


def simulate_day(day: int, sp_of_hour, cop: float = 4.5, film: bool = False) -> dict:
    """One day with an hourly setpoint schedule sp_of_hour(hour) -> °C.

    Returns day kWh / THB, comfort-violation hours, peak ELECTRIC kW inside the
    on-peak tariff window, and the hourly trace for printing.
    """
    t_in = sp_of_hour(0)
    kwh = cost = peak_window = 0.0
    viol = 0
    trace = []
    for hour in range(24):
        t_out, solar = weather(day, hour)
        solar *= (1 - FILM_SOLAR_CUT) if film else 1.0
        sp = sp_of_hour(hour)
        t_in, elec = _step(t_in, t_out, solar, GAINS_KW[hour], sp, cop)
        kwh += elec
        cost += elec * tariff(day, hour)
        if PEAK_START <= hour < PEAK_END:
            peak_window = max(peak_window, elec)
        if t_in > COMFORT_C:
            viol += 1
        trace.append((hour, sp, round(t_in, 2), round(elec, 1)))
    return {"kwh": kwh, "cost_thb": cost, "comfort_viol_h": viol,
            "peak_window_kw": peak_window, "trace": trace}


# ── mock metered data — the "real building" the G14 gate compares against ─────
# The real hotel is NOT the default model: better chiller, heavier internal
# gains, tinted glazing, a 23.5 °C BMS setpoint — plus meter-level wobble.
TRUE_PARAMS = dict(cop=5.0, setpoint=23.5, gains_scale=1.12, solar_scale=0.90)


def metered_monthly() -> list[float]:
    """12 months of mock metered HVAC kWh (deterministic ±2 % wobble)."""
    truth = simulate(TRUE_PARAMS["cop"], TRUE_PARAMS["setpoint"],
                     gains_scale=TRUE_PARAMS["gains_scale"],
                     solar_scale=TRUE_PARAMS["solar_scale"])["monthly_kwh"]
    return [m * (1 + 0.02 * math.sin(2.7 * i + 0.9)) for i, m in enumerate(truth)]


def nmbe(measured: list[float], simulated: list[float]) -> float:
    """Normalized Mean Bias Error, % (G14 uses n−p degrees of freedom; n here)."""
    n, mean = len(measured), sum(measured) / len(measured)
    return sum(m - s for m, s in zip(measured, simulated)) / (n * mean) * 100


def cvrmse(measured: list[float], simulated: list[float]) -> float:
    """Coefficient of Variation of the RMSE, % (same simplification as nmbe)."""
    n, mean = len(measured), sum(measured) / len(measured)
    rmse = math.sqrt(sum((m - s) ** 2 for m, s in zip(measured, simulated)) / n)
    return rmse / mean * 100


# ── AlfalfaLite — an in-process mimic of Alfalfa's BMS-shaped REST API ─────────
class AlfalfaLite:
    """The real Alfalfa wraps EnergyPlus/Spawn behind REST + Haystack points.

    This mimic wraps the RC model behind the same four verbs the demos need:
    list points → read → write setpoint → advance the sim clock.
    """

    POINTS = {
        "ballroom-zone-temp":  {"tags": "zone air temp sensor point unit:°C", "rw": "r"},
        "ballroom-cooling-sp": {"tags": "zone air temp cooling sp writable point unit:°C", "rw": "rw"},
        "chiller-elec-power":  {"tags": "chiller elec power sensor point unit:kW", "rw": "r"},
        "outdoor-air-temp":    {"tags": "outside air temp sensor point unit:°C", "rw": "r"},
    }

    def __init__(self, day: int = 100, cop: float = 4.5):
        self.day, self.hour, self.cop = day, 0, cop
        self.sp, self.t_in, self.elec_kw = 24.0, 24.0, 0.0

    def list_points(self) -> list[dict]:
        return [{"name": n, **meta} for n, meta in self.POINTS.items()]

    def read(self, name: str) -> float:
        t_out, _ = weather(self.day, self.hour)
        return {"ballroom-zone-temp": round(self.t_in, 2),
                "ballroom-cooling-sp": self.sp,
                "chiller-elec-power": round(self.elec_kw, 1),
                "outdoor-air-temp": round(t_out, 1)}[name]

    def write(self, name: str, value: float) -> bool:
        if self.POINTS.get(name, {}).get("rw") != "rw":
            return False
        self.sp = float(value)
        return True

    def advance(self, hours: int = 1) -> None:
        for _ in range(hours):
            t_out, solar = weather(self.day, self.hour)
            self.t_in, self.elec_kw = _step(self.t_in, t_out, solar,
                                            GAINS_KW[self.hour], self.sp, self.cop)
            self.hour = (self.hour + 1) % 24
            if self.hour == 0:
                self.day += 1

    def clock(self) -> str:
        return f"day {self.day} {self.hour:02d}:00"


# ── BuildingFMU — an FMI 2.0 Co-Simulation shaped wrapper around the RC zone ──
class BuildingFMU:
    """The real thing is a .fmu zip (modelDescription.xml + binaries) that FMPy or
    PyFMI loads. This mimic exposes the same verb sequence — instantiate →
    setup_experiment → set/get → do_step — so Ch 4 can walk the wire protocol
    without a compiled binary. It embeds its own solver: that is what makes it
    CO-SIMULATION rather than Model Exchange.
    """

    GUID = "{bkk-hotel-rc-2026-app07}"
    VARIABLES = {"T_set": "input · °C", "T_zone": "output · °C", "P_el": "output · kW"}

    def __init__(self, day: int = 100, cop: float = 4.5):
        self.day, self.cop = day, cop
        self.t_in, self.sp, self.p_el = 24.0, 24.0, 0.0
        self.t, self.stop = 0.0, 0.0

    def instantiate(self, name: str = "building") -> str:
        return f"fmi2Instantiate('{name}', CoSimulation, guid={self.GUID}) → handle ok"

    def setup_experiment(self, start_s: float, stop_s: float) -> str:
        self.t, self.stop = start_s, stop_s
        return f"fmi2SetupExperiment(start={start_s:.0f}s, stop={stop_s:.0f}s) → fmi2OK"

    def set_var(self, name: str, value: float) -> str:
        if name != "T_set":
            return "fmi2Error (not an input)"
        self.sp = float(value)
        return "fmi2OK"

    def get_var(self, name: str) -> float:
        return {"T_zone": round(self.t_in, 2), "P_el": round(self.p_el, 1)}[name]

    def do_step(self, current_t: float, dt: float) -> str:
        """Advance the FMU's INTERNAL solver by dt seconds (whole hours here)."""
        if abs(current_t - self.t) > 1e-6:
            return "fmi2Error (time out of sync)"
        for _ in range(max(1, round(dt / 3600))):
            hour = int(self.t // 3600) % 24
            t_out, solar = weather(self.day + int(self.t // 86400), hour)
            self.t_in, self.p_el = _step(self.t_in, t_out, solar,
                                         GAINS_KW[hour], self.sp, self.cop)
            self.t += 3600
        return "fmi2OK"


# ── the standard sim trio used by view.py ──────────────────────────────────────
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated planner] Reading the KPI table, three what-ifs worth queuing next: "
           "1) extend the pre-cool to 3 h starting 06:00 — the slab was not saturated, so "
           "more off-peak charge should push more kWh (and baht) out of the window; "
           "2) add the Ch 2 window film — the peak-kW line did not move because the "
           "chiller is pinned at its 320 kW cap at midday, and film is the lever that cuts "
           "the solar spike doing the pinning; 3) re-run the COP sweep at 5.5 to price a "
           "chiller retrofit against this tariff. All three stay inside the 25.5 °C comfort "
           "line. Re-check the G14 gate before trusting any of them on the real building.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
