#!/usr/bin/env python3
"""RC building-zone RL simulator — gyms over the building sim, pure stdlib.

Everything the demos need to learn RL-for-buildings with no GPU, no gym, no numpy:

  • RCZone           — one zone of the Week 21 hotel twin as a 1R1C thermal model with
                       the EXACT MDP shape Sinergym gives you: obs ≈17 floats, action
                       Box(2,) = [heating setpoint, cooling setpoint], LinearReward.
  • schedule_policy / BASELINE — the rule-based fixed-setback schedule every paper
                       compares against.
  • run_episode / hill_climb — an honest, readable policy-search learning loop (Ch 3).
  • district_*       — a 3-building mini-district + battery dispatch (the CityLearn shape).
  • boptest tables   — the standard KPI names and test cases (Ch 5).
  • installed_models / tok_s / stream_generate — the standard sim-mode LLM stubs.
"""
from __future__ import annotations

import math
import random
import time

# ── the Sinergym-shaped MDP on a tiny RC zone ─────────────────────────────────
# Action space (Sinergym 5Zone continuous): Box(2,) = [heating sp, cooling sp] in °C.
HEAT_SP_BOUNDS = (12.0, 23.25)
COOL_SP_BOUNDS = (21.75, 30.0)
# Sinergym default comfort ranges for LinearReward:
COMFORT_WINTER = (20.0, 23.5)
COMFORT_SUMMER = (23.0, 26.0)

# The 5Zone default observation — ≈17 floats (time features + configured E+ variables).
OBS_NAMES = [
    "month", "day_of_month", "hour",
    "outdoor_temperature", "outdoor_humidity", "wind_speed", "wind_direction",
    "diffuse_solar_radiation", "direct_solar_radiation",
    "htg_setpoint", "clg_setpoint",
    "air_temperature", "air_humidity", "people_occupant",
    "co2_emission", "HVAC_electricity_demand_rate", "total_electricity_HVAC",
]


class RCZone:
    """One zone of the Week 21 hotel twin (Bangkok — hot climate, cooling-dominated).

    Physics: 1R1C —  C·dT/dt = (Tout − T)/R + q_gains + q_hvac,  stepped at Δt = 900 s
    (15 min), the same timestep Sinergym uses. HVAC is EnergyPlus-style ideal loads:
    deliver exactly what holds the setpoint, capped at Q_MAX. reset()/step() mirror the
    gym API so the demos read like Sinergym code.
    """

    DT = 900.0                 # s per step (15 min) → 672 steps = 1 simulated week
    R = 1.0 / 320.0            # K/W  (UA = 320 W/K envelope)
    C = 3.5e7                  # J/K  (~30 h time constant)
    Q_MAX = 6000.0             # W thermal, heating or cooling
    COP_COOL, COP_HEAT, FAN_W = 3.2, 3.0, 120.0
    AREA_M2 = 96.0             # zone floor area (for BOPTEST-style kWh/m²)
    CO2_KG_PER_KWH = 0.45      # grid emission factor
    # LinearReward:  r = −w·λE·power − (1−w)·λT·Σ|T − comfort bound|   (Sinergym defaults)
    W, LAMBDA_E, LAMBDA_T = 0.5, 1e-4, 1.0
    COMFORT = COMFORT_SUMMER   # Bangkok: the summer band applies year-round

    EPISODE_STEPS = 672        # one simulated week

    def __init__(self, stochastic: bool = True, seed: int = 21):
        self.stochastic, self.seed = stochastic, seed
        self.reset()

    # gym API ------------------------------------------------------------------
    def reset(self) -> list[float]:
        self._rng = random.Random(self.seed)
        self.t = 0                     # step index
        self.T = 26.5                  # zone air temperature, °C
        self.kwh = 0.0                 # cumulative HVAC electricity
        self.cost = 0.0                # cumulative $ at the dynamic tariff
        self.disc_kh = 0.0             # cumulative comfort violation, K·h (occupied)
        self.co2_kg = 0.0
        self._last_p = 0.0
        return self._obs(21.0, 25.0)

    def step(self, action) -> tuple[list[float], float, bool, dict]:
        heat_sp = min(max(float(action[0]), HEAT_SP_BOUNDS[0]), HEAT_SP_BOUNDS[1])
        cool_sp = min(max(float(action[1]), COOL_SP_BOUNDS[0]), COOL_SP_BOUNDS[1])
        hour = self.hour()
        occ = self.occupied(hour)
        tout, solar = self._weather(hour)
        q_gain = 1200.0 * solar + (800.0 if occ else 150.0)      # W: sun + people/plug

        # free-float, then ideal-loads HVAC toward the setpoint band, capped at Q_MAX
        t_free = self.T + self.DT / self.C * ((tout - self.T) / self.R + q_gain)
        p_elec = 0.0
        if t_free > cool_sp:                                     # cooling
            q = min(self.Q_MAX, (t_free - cool_sp) * self.C / self.DT)
            t_free -= q * self.DT / self.C
            p_elec = q / self.COP_COOL + self.FAN_W
        elif t_free < heat_sp:                                   # heating (rare here)
            q = min(self.Q_MAX, (heat_sp - t_free) * self.C / self.DT)
            t_free += q * self.DT / self.C
            p_elec = q / self.COP_HEAT + self.FAN_W
        self.T = t_free

        # LinearReward — energy term always, comfort term when someone is there
        lo, hi = self.COMFORT
        disc = (max(0.0, lo - self.T) + max(0.0, self.T - hi)) if occ else 0.0
        reward = -self.W * self.LAMBDA_E * p_elec - (1 - self.W) * self.LAMBDA_T * disc

        kwh = p_elec * self.DT / 3.6e6
        self.kwh += kwh
        self.cost += kwh * grid_price(hour)
        self.co2_kg += kwh * self.CO2_KG_PER_KWH
        self.disc_kh += disc * self.DT / 3600.0
        self._last_p = p_elec
        self.t += 1
        done = self.t >= self.EPISODE_STEPS
        info = {"power_w": p_elec, "tout": tout, "occupied": occ, "discomfort_k": disc,
                "t_zone": self.T}
        return self._obs(heat_sp, cool_sp), reward, done, info

    # internals ----------------------------------------------------------------
    def hour(self) -> float:
        return (self.t * self.DT / 3600.0) % 24.0

    @staticmethod
    def occupied(hour: float) -> bool:
        return 7 <= hour < 23

    def _weather(self, hour: float) -> tuple[float, float]:
        tout = 30.0 + 4.5 * math.sin(2 * math.pi * (hour - 9.0) / 24.0)   # peak ~15:00
        if self.stochastic:
            tout += self._rng.gauss(0.0, 0.6)                             # weather noise
        solar = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))         # 0..1, 06–18 h
        return tout, solar

    def _obs(self, heat_sp: float, cool_sp: float) -> list[float]:
        hour = self.hour()
        day = 1 + int(self.t * self.DT // 86400)
        tout, solar = self._weather(hour)
        occ = 4.0 if self.occupied(hour) else 0.0
        return [7.0, float(day), hour,
                round(tout, 2), 68.0 + 6.0 * solar, 2.6, 190.0,
                round(120.0 * solar, 1), round(620.0 * solar, 1),
                heat_sp, cool_sp,
                round(self.T, 2), 55.0, occ,
                round(self.co2_kg, 3), round(self._last_p, 1), round(self.kwh, 3)]


def grid_price(hour: float) -> float:
    """Dynamic electricity tariff, $/kWh (BOPTEST 'dynamic' scenario shape)."""
    return 0.20 if 9 <= hour < 22 else 0.10


# ── controllers ───────────────────────────────────────────────────────────────
# The rule-based baseline: a fixed setback schedule (what most real BAS run).
BASELINE = {"heat_occ": 20.0, "cool_occ": 24.5, "heat_nt": 15.0, "cool_nt": 27.0}


def schedule_policy(params: dict, hour: float) -> tuple[float, float]:
    """Setpoint schedule: one (heat, cool) pair when occupied, one at night."""
    if RCZone.occupied(hour):
        return params["heat_occ"], params["cool_occ"]
    return params["heat_nt"], params["cool_nt"]


def run_episode(env: RCZone, params: dict) -> tuple[float, float, float]:
    """Run one simulated week with a schedule; return (reward, kWh, discomfort K·h)."""
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        action = schedule_policy(params, obs[2])          # obs[2] = hour
        obs, r, done, _ = env.step(action)
        total += r
    return total, env.kwh, env.disc_kh


def hill_climb(env: RCZone, episodes: int = 24, seed: int = 7):
    """The learning loop of Ch 3 — honest label: random-mutation hill-climbing over the
    4 schedule setpoints, NOT PPO. But the loop shape (act → observe reward → keep what
    improved) is exactly what SB3's PPO does at scale on Sinergym. Same seeded weather
    for every candidate (common random numbers) so comparisons are fair."""
    rng = random.Random(seed)
    best = dict(BASELINE)
    best_r, _, _ = run_episode(env, best)
    history = [best_r]
    for _ in range(episodes - 1):
        cand = {k: v + rng.gauss(0.0, 0.7) for k, v in best.items()}
        for k in ("heat_occ", "heat_nt"):
            cand[k] = min(max(cand[k], HEAT_SP_BOUNDS[0]), HEAT_SP_BOUNDS[1])
        for k in ("cool_occ", "cool_nt"):
            cand[k] = min(max(cand[k], COOL_SP_BOUNDS[0]), COOL_SP_BOUNDS[1])
        r, _, _ = run_episode(env, cand)
        if r > best_r:
            best, best_r = cand, r
        history.append(best_r)
    return best, history


# ── Sinergym canned facts (Ch 2) ──────────────────────────────────────────────
# Env ID pattern: Eplus-<building>-<climate>-<actionspace>[-stochastic]-v1
SINERGYM_BUILDINGS = ["5zone (office, VAV)", "datacenter", "office", "warehouse", "shop"]
SINERGYM_CLIMATES = ["hot", "mixed", "cool"]
SINERGYM_ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"

# ── CityLearn mini-district (Ch 4) ────────────────────────────────────────────
# Hourly electrical load profiles (kW) — pre-simulated timeseries, the CityLearn way.
DISTRICT = [
    ("Hotel",  [38, 35, 33, 32, 32, 34, 40, 48, 52, 50, 48, 47,
                46, 45, 44, 46, 52, 60, 68, 72, 70, 62, 52, 44]),
    ("Office", [12, 11, 11, 11, 11, 13, 22, 45, 62, 70, 72, 71,
                68, 70, 69, 66, 58, 44, 30, 22, 18, 16, 14, 13]),
    ("Retail", [8, 8, 8, 8, 8, 9, 14, 22, 34, 44, 50, 54,
                56, 55, 54, 55, 58, 60, 56, 44, 28, 16, 10, 9]),
]
BATTERY = {"capacity_kwh": 120.0, "power_kw": 40.0, "efficiency": 0.92}  # per building


def grid_co2(hour: int) -> float:
    """kg CO₂ per kWh — dirtier peaker plants come online in the evening."""
    return 0.55 if 17 <= hour < 22 else 0.42


def district_baseline() -> list[float]:
    return [sum(b[1][h] for b in DISTRICT) for h in range(24)]


def _simulate_dispatch(level: float, soc: float = 0.0) -> tuple[list[float], list[float], float]:
    """Run the 3 batteries as one pool against a flattening threshold: discharge
    when district load > level, charge (with round-trip losses) when below it.
    Batteries only shift grid energy, they never create it."""
    base = district_baseline()
    cap = 3 * BATTERY["capacity_kwh"]
    pwr = 3 * BATTERY["power_kw"]
    eta = BATTERY["efficiency"]
    grid, flow = [], []
    for h in range(24):
        load = base[h]
        if load > level:                                  # discharge to shave the peak
            d = min(pwr, load - level, soc)
            soc -= d
            grid.append(load - d)
            flow.append(-d)
        else:                                             # charge in the valley
            c = min(pwr, level - load, (cap - soc) / eta)
            soc += c * eta                                # round-trip losses on charge
            grid.append(load + c)
            flow.append(c)
    return grid, flow, soc


def district_dispatch() -> tuple[list[float], list[float]]:
    """Peak-shaving dispatch. A naive 'discharge whenever above the mean' policy
    drains the pool before the 19:00 peak, so we bisect the flattening threshold
    until the batteries can actually hold it, then report a steady-state (cyclic)
    day so the only consumption overhead is honest round-trip loss. Returns
    (grid_load, battery_flow) per hour; flow > 0 = charging, < 0 = discharging."""
    base = district_baseline()
    lo, hi = sum(base) / 24.0, max(base)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        grid, _, _ = _simulate_dispatch(mid)
        if max(grid) > mid + 1e-6:
            lo = mid                                      # ran dry — raise the threshold
        else:
            hi = mid
    soc = 0.0
    for _ in range(3):                                    # settle into the daily cycle
        grid, flow, soc = _simulate_dispatch(hi, soc)
    return grid, flow


def district_kpis(profile: list[float]) -> dict:
    """CityLearn-style district KPIs from an hourly grid-load profile."""
    peak = max(profile)
    consumption = sum(profile)
    ramping = sum(abs(profile[h] - profile[h - 1]) for h in range(1, 24))
    load_factor = (consumption / 24.0) / peak
    cost = sum(profile[h] * grid_price(h) for h in range(24))
    co2 = sum(profile[h] * grid_co2(h) for h in range(24))
    return {"district consumption (kWh)": consumption, "cost ($)": cost,
            "CO₂ (kg)": co2, "ramping (kW)": ramping,
            "1 − load factor": 1.0 - load_factor, "peak demand (kW)": peak}


# ── BOPTEST canned facts (Ch 5) ───────────────────────────────────────────────
BOPTEST_KPIS = [
    ("tdis_tot", "K·h",     "thermal discomfort — deviation from comfort band, occupied"),
    ("idis_tot", "ppm·h",   "IAQ discomfort — CO₂ above the limit, occupied"),
    ("ener_tot", "kWh/m²",  "total HVAC energy use, per floor area"),
    ("cost_tot", "$/m² or €/m²", "energy cost under the scenario's price signal"),
    ("emis_tot", "kgCO₂/m²", "CO₂ emissions, per floor area"),
    ("time_rat", "s/s",     "controller compute time per simulated second"),
]
BOPTEST_CASES = [
    "bestest_air", "bestest_hydronic", "bestest_hydronic_heat_pump",
    "multizone_residential_hydronic", "singlezone_commercial_hydronic",
    "multizone_office_simple_air",
]


def boptest_kpis_for(params: dict) -> dict:
    """Run one simulated week of the RC zone under a schedule and report it in
    BOPTEST's standard KPI names (per-m² where BOPTEST is)."""
    started = time.perf_counter()
    env = RCZone(stochastic=True, seed=21)
    run_episode(env, params)
    elapsed_s = time.perf_counter() - started
    sim_seconds = env.EPISODE_STEPS * env.DT
    return {"tdis_tot": env.disc_kh,
            "idis_tot": 0.0,   # our RC zone has no CO₂ model; real test cases do
            "ener_tot": env.kwh / env.AREA_M2,
            "cost_tot": env.cost / env.AREA_M2,
            "emis_tot": env.co2_kg / env.AREA_M2,
            "time_rat": elapsed_s / sim_seconds}


def lib_mode_line(pkg: str, pip_name: str, note: str) -> None:
    """MODE line for the gym demos: REAL if the optional library imports, else SIM.
    Either way the demo itself runs on the built-in RC zone — SIM teaches the same MDP."""
    try:
        __import__(pkg)
        print(f"▣ MODE: REAL — `{pip_name}` is installed. {note}")
    except ImportError:
        print(f"▣ MODE: SIM — `{pip_name}` not installed (fine, $0). {note}")
        print(f"  the 'run it for real' block below is exactly what you'd run with it.")
    print()


# ── standard sim-mode LLM stubs (used by view.py) ─────────────────────────────
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 41.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated controls-lab] On BOPTEST's fixed scenarios, well-tuned MPC still "
           "generally beats RL in the literature: MPC reads the /forecast endpoint and "
           "exploits a model of the building, while RL must learn the dynamics from "
           "reward alone. RL earns its place when no usable model exists or when the "
           "building drifts away from its model — RL keeps learning while MPC's model "
           "quietly goes stale. Either way, a policy that wins in the gym is not yet "
           "safe on the real building — safe deployment is App 10.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
