#!/usr/bin/env python3
"""The SIMULATION — 5R1C zone model · G14 calibration · surrogate  (Apps 6–8 condensed).

A reduced-order RC model (ISO 13790's 5R1C, collapsed to the parts that matter for
a cooling-dominated Bangkok hotel) of a representative guest-room zone, plus:
  • the ASHRAE Guideline 14 calibration gate — hourly NMBE ≤ ±10 %, CV(RMSE) ≤ 30 % —
    because a twin's what-if answers are only credible AFTER calibration;
  • a fitted fast surrogate (closed-form least squares, the PhysicsNeMo pattern at
    toy scale) with a training-envelope guard — fast answers, honestly bounded;
  • a pre-cool decision helper driven by a weather forecast (the Earth-2 pattern).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

# Bangkok hourly outdoor temp (°C), a typical hot-season day — used everywhere.
BKK_WEATHER = [round(27.0 + 4.5 * math.sin((h - 9) / 24 * 2 * math.pi) + (1.5 if 11 <= h <= 16 else 0), 1)
               for h in range(24)]
TARIFF_THB_KWH = {"peak": 5.8, "offpeak": 2.6}      # peak 09:00–22:00 (MEA TOU, rounded)


def tariff(hour: int) -> float:
    return TARIFF_THB_KWH["peak"] if 9 <= hour < 22 else TARIFF_THB_KWH["offpeak"]


@dataclass
class ZoneParams:
    h_tr: float = 95.0        # W/K — envelope transmission (walls+window, 5R collapsed)
    c_m: float = 6.5e6        # J/K — thermal mass (the 1C)
    gains_w: float = 900.0    # W   — internal + solar gains when occupied (daytime)
    cop: float = 3.4          # chilled-water system COP


TRUE_PARAMS = ZoneParams(h_tr=112.0, gains_w=1050.0)   # what the real building actually does


def simulate(p: ZoneParams, setpoints: list[float], weather=None,
             t0: float = 26.0) -> tuple[list[float], list[float]]:
    """One zone, hourly Euler steps → (zone temps °C, cooling electricity kWh)."""
    weather = weather or BKK_WEATHER
    t, temps, kwh = t0, [], []
    for h, (t_out, sp) in enumerate(zip(weather, setpoints)):
        gains = p.gains_w if 7 <= h <= 23 else p.gains_w * 0.3
        drift = (p.h_tr * (t_out - t) + gains) / p.c_m * 3600.0   # °C free-float this hour
        t_free = t + drift
        if t_free > sp:                       # cooling holds the zone at setpoint
            q_w = (t_free - sp) * p.c_m / 3600.0
            kwh.append(round(q_w / 1000.0 / p.cop, 3))
            t = sp
        else:
            kwh.append(0.0)
            t = t_free
        temps.append(round(t, 2))
    return temps, kwh


def mock_meter(setpoints: list[float]) -> list[float]:
    """The 'measured' zone meter: the TRUE building + deterministic sensor noise."""
    _, kwh = simulate(TRUE_PARAMS, setpoints)
    return [round(v * (1 + 0.06 * math.sin(h * 2.1)), 3) for h, v in enumerate(kwh)]


def nmbe(meas: list[float], sim: list[float]) -> float:
    mbar = sum(meas) / len(meas)
    return 100.0 * sum(m - s for m, s in zip(meas, sim)) / ((len(meas) - 1) * mbar)


def cvrmse(meas: list[float], sim: list[float]) -> float:
    mbar = sum(meas) / len(meas)
    rmse = math.sqrt(sum((m - s) ** 2 for m, s in zip(meas, sim)) / (len(meas) - 1))
    return 100.0 * rmse / mbar


def calibrate(setpoints: list[float]) -> tuple[ZoneParams, float, float, bool, list]:
    """Grid-search h_tr × gains against the meter → (params, NMBE, CV(RMSE), gate, trail)."""
    measured, trail = mock_meter(setpoints), []
    best, best_err = None, 1e9
    for h_tr in (85, 95, 105, 112, 120):
        for gains in (800, 900, 1000, 1050, 1150):
            p = ZoneParams(h_tr=h_tr, gains_w=gains)
            _, kwh = simulate(p, setpoints)
            err = cvrmse(measured, kwh)
            trail.append((h_tr, gains, round(err, 1)))
            if err < best_err:
                best, best_err = p, err
    _, kwh = simulate(best, setpoints)
    n, cv = nmbe(measured, kwh), cvrmse(measured, kwh)
    passed = abs(n) <= 10.0 and cv <= 30.0        # ASHRAE Guideline 14, hourly criteria
    return best, round(n, 1), round(cv, 1), passed, trail


def sweep(p: ZoneParams, setpoint_list: list[float]) -> list[tuple[float, float]]:
    """Parametric what-if: constant setpoint → daily kWh."""
    return [(sp, round(sum(simulate(p, [sp] * 24)[1]), 2)) for sp in setpoint_list]


class Surrogate:
    """kWh/day ≈ a + b·setpoint + c·T̄out — closed-form least squares on sim samples.
    Millisecond-class answers, but only INSIDE the training envelope (the honest part)."""

    def __init__(self):
        self.coef, self.env = None, {}

    def fit(self, samples: list[tuple[float, float, float]]) -> None:
        """samples: (setpoint, mean outdoor °C, kWh/day) from the calibrated model."""
        n = len(samples)
        sx = [[1.0, s, t] for s, t, _ in samples]
        y = [k for *_, k in samples]
        # normal equations for the 3-coef fit (stdlib, no numpy)
        ata = [[sum(a[i] * a[j] for a in sx) for j in range(3)] for i in range(3)]
        aty = [sum(a[i] * yi for a, yi in zip(sx, y)) for i in range(3)]
        self.coef = _solve3(ata, aty)
        self.env = {"setpoint": (min(s for s, *_ in samples), max(s for s, *_ in samples)),
                    "t_out": (min(t for _, t, _ in samples), max(t for _, t, _ in samples))}

    def predict(self, setpoint: float, t_out: float) -> tuple[float, bool]:
        """→ (kWh/day, inside_training_envelope). Outside the envelope, don't trust it."""
        lo, hi = self.env["setpoint"]
        lo2, hi2 = self.env["t_out"]
        inside = lo <= setpoint <= hi and lo2 <= t_out <= hi2
        a, b, c = self.coef
        return round(a + b * setpoint + c * t_out, 2), inside


def _solve3(a, b):
    """3×3 Gaussian elimination."""
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        for r in range(3):
            if r != i and m[i][i]:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][3] / m[i][i] for i in range(3)]


def precool_decision(p: ZoneParams, forecast: list[float]) -> dict:
    """Forecast says tomorrow runs hot → compare hold-23 °C vs pre-cool 05:00–09:00."""
    base_sp = [23.0] * 24
    pre_sp = [21.5 if 5 <= h < 9 else (23.0 if h < 14 else 23.5) for h in range(24)]
    _, base_kwh = simulate(p, base_sp, forecast)
    tp, pre_kwh = simulate(p, pre_sp, forecast)
    cost = lambda kwh: sum(k * tariff(h) for h, k in enumerate(kwh))  # noqa: E731
    worst = max(t - sp for t, sp in zip(tp, [23.5] * 24))
    return {"base_kwh": round(sum(base_kwh), 1), "pre_kwh": round(sum(pre_kwh), 1),
            "base_thb": round(cost(base_kwh)), "pre_thb": round(cost(pre_kwh)),
            "comfort_excursion_c": round(max(0.0, worst), 2)}


def time_ms(fn, *args, n: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n * 1000.0


if __name__ == "__main__":
    sp = [23.0] * 24
    p, n, cv, ok, _ = calibrate(sp)
    print(f"calibrated h_tr={p.h_tr} gains={p.gains_w} → NMBE {n}% CV(RMSE) {cv}% gate={'PASS' if ok else 'FAIL'}")
    print("sweep:", sweep(p, [22.0, 23.0, 24.0, 25.0]))
