#!/usr/bin/env python3
"""Physics-surrogate simulator — a toy RC hotel wing, surrogate helpers, a forecast.

Everything the four demos need, pure stdlib and deterministic:

  • a lumped RC model of a Bangkok hotel's cooled public wing (ballroom + lobby):
    inputs (outdoor °C, occupancy, setpoint °C, solar kW) → output cooling ELECTRIC kW.
    It marches a thermal-mass node to quasi-steady state — deliberately the "slow
    truth" (~1 ms/query) the surrogate will be trained to replace (~µs/query).
  • generate_samples() — parametric runs of that model = the surrogate's training set
  • features() / predict() / fit_surrogate() — a polynomial-feature linear model
    fitted by plain gradient descent (no numpy; a surrogate small enough to read)
  • ENVELOPE + in_envelope() + predict_guarded() — the fallback-to-physics guard
  • forecast_24h() — tomorrow's weather with a heat spike, for the Earth-2 chapter
  • run_schedule() — hourly pre-cool what-if over the forecast, with a coolth ledger
  • the standard installed_models() / tok_s() / stream_generate() trio for view.py
"""
from __future__ import annotations

import random
import time

# ── the toy building: one cooled public wing of a 180-room Bangkok hotel ──────
UA = 9.0            # kW/K — envelope + ventilation conductance (lumped)
OCC_GAIN = 80.0     # kW internal gains at 100 % occupancy (people·lights·kitchen)
SHGC = 0.55         # fraction of incident solar that becomes zone load
CAP_KW = 420.0      # chiller plant thermal capacity, kW — real physics SATURATES
C_MASS = 35.0       # kWh/K — effective thermal mass (slab, walls, furniture)
H_MA = 6.0          # kW/K — mass ↔ zone-air coupling
UA_M = 2.0          # kW/K — outdoor → mass path (structure heated by the sun/air)

# the surrogate's TRAINING ENVELOPE — it is only trustworthy inside this box
ENVELOPE = {"t_out": (26.0, 36.0), "occ": (0.10, 1.00),
            "setpoint": (22.0, 26.0), "solar": (0.0, 300.0)}


def _cop(t_out: float) -> float:
    """Chiller COP degrades on hot days — a mild nonlinearity the surrogate must learn."""
    return max(2.5, 6.4 - 0.09 * (t_out - 26.0))


def full_detail(t_out: float, occ: float, setpoint: float, solar: float,
                *, steps: int = 1800) -> tuple[float, float, bool]:
    """The FULL physics: march the RC mass node (1-min steps, 30 h) to steady state.

    Returns (electric kW, zone °C, clamped?). Clamped = the chiller hit CAP_KW and
    the zone floated above setpoint — the saturation a polynomial can't extrapolate.
    """
    gains = occ * OCC_GAIN * (1.0 + 0.18 * occ) + SHGC * solar   # latent ↑ superlinearly
    t_mass, dt = t_out - 1.0, 1.0 / 60.0
    q_th, t_air = 0.0, setpoint
    for _ in range(steps):
        load = UA * (t_out - setpoint) + gains + H_MA * (t_mass - setpoint)
        q_th = min(max(load, 0.0), CAP_KW)                       # the chiller clamp
        t_air = setpoint + max(0.0, load - q_th) / (UA + H_MA)   # floats up if clamped
        t_mass += (UA_M * (t_out - t_mass) + H_MA * (t_air - t_mass)) * dt / C_MASS
    return q_th / _cop(t_out), t_air, q_th >= CAP_KW - 1e-9


def full_model(t_out: float, occ: float, setpoint: float, solar: float) -> float:
    """Electric kW from the full RC march — the slow source of truth (~1 ms)."""
    return full_detail(t_out, occ, setpoint, solar)[0]


def generate_samples(n: int = 200, seed: int = 21) -> list[tuple[tuple[float, ...], float]]:
    """n parametric runs of the full model, sampled INSIDE the training envelope."""
    rng = random.Random(seed)
    lo_hi = [ENVELOPE[k] for k in ("t_out", "occ", "setpoint", "solar")]
    samples = []
    for _ in range(n):
        x = tuple(round(rng.uniform(lo, hi), 3) for lo, hi in lo_hi)
        samples.append((x, full_model(*x)))
    return samples


# ── the surrogate: 15 polynomial features on normalized inputs ────────────────
def features(t_out: float, occ: float, setpoint: float, solar: float) -> list[float]:
    """[1, u1..u4, u1²..u4², all pairwise ui·uj] with ui normalized to [0,1]."""
    lo_hi = [ENVELOPE[k] for k in ("t_out", "occ", "setpoint", "solar")]
    u = [(v - lo) / (hi - lo) for v, (lo, hi) in zip((t_out, occ, setpoint, solar), lo_hi)]
    f = [1.0] + u + [ui * ui for ui in u]
    f += [u[i] * u[j] for i in range(4) for j in range(i + 1, 4)]
    return f                                                     # 1 + 4 + 4 + 6 = 15


Y_SCALE = 100.0     # train on kW/100 so gradient descent is well-conditioned


def predict(w: list[float], t_out: float, occ: float, setpoint: float, solar: float) -> float:
    """The whole surrogate at inference time: one 15-term dot product (~µs)."""
    return sum(wi * fi for wi, fi in zip(w, features(t_out, occ, setpoint, solar))) * Y_SCALE


def fit_surrogate(samples, *, epochs: int = 1200, lr: float = 0.55,
                  report=None, report_every: int = 200) -> list[float]:
    """Full-batch gradient descent on MSE — pure Python, deterministic.

    `report(epoch, mse_kw2)` is called every `report_every` epochs if given.
    """
    feats = [features(*x) for x, _ in samples]
    ys = [y / Y_SCALE for _, y in samples]
    w, n = [0.0] * 15, len(samples)
    for epoch in range(1, epochs + 1):
        grad = [0.0] * 15
        loss = 0.0
        for f, y in zip(feats, ys):
            err = sum(wi * fi for wi, fi in zip(w, f)) - y
            loss += err * err
            for k in range(15):
                grad[k] += err * f[k]
        w = [wi - lr * g / n for wi, g in zip(w, grad)]
        if report and (epoch == 1 or epoch % report_every == 0):
            report(epoch, (loss / n) * Y_SCALE * Y_SCALE)        # back to kW² units
    return w


def in_envelope(t_out: float, occ: float, setpoint: float, solar: float) -> bool:
    vals = {"t_out": t_out, "occ": occ, "setpoint": setpoint, "solar": solar}
    return all(lo <= vals[k] <= hi for k, (lo, hi) in ENVELOPE.items())


def predict_guarded(w, t_out, occ, setpoint, solar) -> tuple[float, bool]:
    """The guard: surrogate inside the envelope, FULL PHYSICS outside. → (kW, fell_back)."""
    if in_envelope(t_out, occ, setpoint, solar):
        return predict(w, t_out, occ, setpoint, solar), False
    return full_model(t_out, occ, setpoint, solar), True


# ── tomorrow's forecast: a Bangkok day with a heat spike (Earth-2 chapter) ────
# (hour, outdoor °C, solar kW, occupancy) — peak 37.8 °C is ABOVE the envelope.
FORECAST = [
    (0, 27.6, 0, 0.25), (1, 27.2, 0, 0.20), (2, 26.9, 0, 0.15), (3, 26.7, 0, 0.15),
    (4, 26.5, 0, 0.15), (5, 26.6, 0, 0.20), (6, 27.4, 20, 0.30), (7, 28.8, 60, 0.45),
    (8, 30.4, 110, 0.55), (9, 32.0, 170, 0.60), (10, 33.6, 230, 0.60), (11, 35.0, 280, 0.65),
    (12, 36.2, 305, 0.70), (13, 37.1, 310, 0.65), (14, 37.7, 295, 0.60), (15, 37.8, 260, 0.60),
    (16, 37.2, 205, 0.65), (17, 36.1, 130, 0.75), (18, 34.6, 55, 0.85), (19, 33.2, 10, 0.90),
    (20, 32.0, 0, 0.90), (21, 31.0, 0, 0.80), (22, 30.1, 0, 0.60), (23, 29.0, 0, 0.40),
]

TARIFF_ON, TARIFF_OFF = 4.3, 2.6      # THB/kWh · Thai-style TOU, on-peak 09:00–22:00
PEAK_START, PEAK_END = 9, 22


def tariff(hour: int) -> float:
    return TARIFF_ON if PEAK_START <= hour < PEAK_END else TARIFF_OFF


def forecast_24h() -> list[dict]:
    return [{"hour": h, "t_out": t, "solar": s, "occ": o} for h, t, s, o in FORECAST]


def run_schedule(w, setpoints: dict[int, float], *, precool_hours=(), relief_hours=(),
                 depth_k: float = 3.0) -> dict:
    """Hourly what-if over the forecast with the GUARDED surrogate + a coolth ledger.

    Pre-cool hours over-cool the thermal mass by `depth_k` kelvin (extra kW, cheap);
    relief hours draw the stored coolth back down (less kW, expensive).
    Returns rows + peak/kWh/THB totals and how often the guard fell back to physics.
    """
    stored = 0.0                                       # kWh of coolth banked in the mass
    stored_target = C_MASS * depth_k if precool_hours else 0.0
    charge = stored_target / len(precool_hours) if precool_hours else 0.0
    discharge = stored_target / len(relief_hours) if relief_hours else 0.0
    rows, peak, kwh, thb, fallbacks = [], 0.0, 0.0, 0.0, 0
    for rec in forecast_24h():
        h, sp = rec["hour"], setpoints.get(rec["hour"], 24.0)
        kw, fell = predict_guarded(w, rec["t_out"], rec["occ"], sp, rec["solar"])
        note = ""
        if h in precool_hours:
            kw += charge / _cop(rec["t_out"]); stored += charge; note = "charge mass"
        elif h in relief_hours and stored > 0:
            take = min(discharge, stored)
            kw = max(0.0, kw - take / _cop(rec["t_out"])); stored -= take; note = "discharge"
        fallbacks += fell
        peak = max(peak, kw); kwh += kw; thb += kw * tariff(h)
        rows.append({"hour": h, "sp": sp, "kw": kw, "fell_back": fell, "note": note})
    return {"rows": rows, "peak_kw": peak, "kwh": kwh, "thb": thb, "fallbacks": fallbacks}


# ── standard LLM-sim exports (used by view.py when no endpoint is reachable) ──
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma-4:27b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma-4:27b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated] Recommendation: pre-cool. Tomorrow's forecast peaks at 37.8 °C "
           "mid-afternoon, above our normal band. Over-cool the ballroom wing to 22 °C "
           "from 05:00–08:00 while power is off-peak, then let it drift to 25.5 °C "
           "through the 13:00–19:00 spike. The thermal mass carries the stored coolth "
           "through the peak: lower peak demand and a lower daily bill for a small "
           "off-peak kWh premium, with no comfort complaints expected. The twin's "
           "surrogate evaluated both schedules in milliseconds; the six hottest hours "
           "were outside its training envelope, so full physics checked those.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
