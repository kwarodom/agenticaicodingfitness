#!/usr/bin/env python3
"""Self-evolving building-ops simulator — memory, flywheel, guardrails; pure stdlib.

Everything the demos need to learn SAFE self-evolving building operations, no GPU:

  • EPISODES / consolidate()          — Week 18's tripartite memory applied to ops:
                                        control episodes → semantic facts + a SKILL.md.
  • PREF_PAIR / flywheel_runs()       — Week 23's data flywheel on operator overrides:
                                        curate → customize → evaluate → promote, 5 runs.
  • LADDER / LADDER_WALK / GUARDRAILS — the 4-rung safe-autonomy ladder, gate criteria,
                                        and the hard-guardrail set for rung 4.
  • GuardrailEngine / simulate_week() — clamps, rate limits, watchdogs, G36 fallback;
                                        a 7-day guardrailed week (frozen sensor, override).
  • installed_models / tok_s / stream_generate — the standard sim-mode LLM stubs.
"""
from __future__ import annotations

import math
import time

# ══ Ch 2 · tripartite memory — 3 morning-startup episodes ═════════════════════
# An EPISODE is what the agent logs after every control run: state, action,
# outcome, operator reaction — plus TAGGED observations/corrections so the
# consolidation rules below can count them across episodes.
EPISODES = [
    {"id": "ep-101", "when": "Mon 06:00 · morning startup (monsoon season)",
     "state": "t_out 27.4 °C · RH 88 % · ballroom 27.9 °C · AHU-3 supply ΔT 6.1 K",
     "action": "chillers on 06:00 · ballroom pre-cool 06:30 for the 07:00 event",
     "outcome": "ballroom hit 24.5 °C at 07:15 — 15 min LATE, guests arrived warm",
     "reaction": "OVERRIDE — “pre-cool the ballroom 45 min before the event, not 30”",
     "observations": [("ballroom-lag", "ballroom trailed its setpoint step by ≈ 45 min"),
                      ("ahu3-fouling", "AHU-3 supply ΔT 6.1 K vs 7.6 K design")],
     "corrections": [("monsoon-startup", "chillers 05:30 + ballroom pre-cool 45 min early")]},
    {"id": "ep-102", "when": "Tue 06:00 · morning startup (monsoon season)",
     "state": "t_out 27.1 °C · RH 90 % · ballroom 27.7 °C · AHU-3 supply ΔT 6.0 K",
     "action": "chillers on 05:30 · ballroom pre-cool 06:15 (operator's timing)",
     "outcome": "ballroom at 24.4 °C by 07:00 — ON TIME",
     "reaction": "no correction — operator watched and approved",
     "observations": [("ballroom-lag", "45-min pre-cool landed exactly on time")],
     "corrections": []},
    {"id": "ep-103", "when": "Wed 06:00 · morning startup (monsoon season)",
     "state": "t_out 27.6 °C · RH 87 % · ballroom 27.8 °C · AHU-3 supply ΔT 6.1 K",
     "action": "chillers on 06:00 (agent reverted) · pre-cool 06:15",
     "outcome": "on time, but chiller 2 rushed to 96 % load at 06:20",
     "reaction": "OVERRIDE — “humid mornings need both chillers from 05:30”",
     "observations": [("ahu3-fouling", "AHU-3 ΔT still 6.1 K — down ~20 % since March")],
     "corrections": [("monsoon-startup", "chillers 05:30 + ballroom pre-cool 45 min early")]},
]

_FACT_TEXT = {
    "ballroom-lag": "ballroom thermal lag ≈ 45 min — start pre-cool 45 min before occupancy",
    "ahu3-fouling": ("AHU-3 fouling: supply ΔT down 20 % since March (6.1 K vs 7.6 K design) "
                     "— derate its capacity, schedule a coil clean"),
}

SKILL_MD = """\
# SKILL: morning-startup-monsoon-season
trigger : 05:15 daily · monsoon flag (rain forecast or RH > 85 %)
steps   :
  1. start chillers CH-1 + CH-2 at 05:30 (humid air = extra latent load)
  2. pre-cool the ballroom 45 min before its first event (thermal lag ≈ 45 min)
  3. expect only ~80 % capacity from AHU-3 until the coil clean (fouling fact)
  4. verify every occupied zone ≤ 25.0 °C by 06:45, else page the duty engineer
learned-from: ep-101, ep-102, ep-103 (3 mornings · 2 operator overrides)"""


def consolidate(episodes: list[dict], min_support: int = 2):
    """The background CONSOLIDATION loop, as two honest counting rules:
      rule 1 — an OBSERVATION tagged in ≥ min_support episodes → durable semantic fact
      rule 2 — an operator CORRECTION repeated in ≥ min_support episodes → procedural skill
    Returns (facts, skills) — what Week 18 calls MEMORY.md lines and SKILL.md files."""
    obs_n: dict[str, int] = {}
    corr_n: dict[str, int] = {}
    for ep in episodes:
        for tag, _ in ep["observations"]:
            obs_n[tag] = obs_n.get(tag, 0) + 1
        for tag, _ in ep["corrections"]:
            corr_n[tag] = corr_n.get(tag, 0) + 1
    facts = [_FACT_TEXT[t] for t, n in obs_n.items() if n >= min_support]
    skills = ([("morning-startup-monsoon-season", SKILL_MD)]
              if corr_n.get("monsoon-startup", 0) >= min_support else [])
    return facts, skills


# ══ Ch 3 · the override flywheel (Week 23's loop, on operator overrides) ══════
FLYWHEEL_STAGES = [
    ("① OBSERVE",   "production control logs — every action, outcome, override"),
    ("② CURATE",    "filter to override moments → (state, rejected, chosen) preference pairs"),
    ("③ CUSTOMIZE", "adjust the policy on the preference pairs (DPO-shaped data)"),
    ("④ EVALUATE",  "sim gate on App 9 KPIs (kWh, K·h, cost) + LLM-judge on incident notes"),
    ("⑤ PROMOTE",   "the winner replaces the incumbent — then the loop repeats"),
]

# One override, in the shape preference-tuning wants (chosen ≻ rejected):
PREF_PAIR = {
    "state":    "Mon 06:30 · monsoon · ballroom 27.9 °C · event at 07:00",
    "rejected": "pre-cool the ballroom at 06:30           (what the agent did)",
    "chosen":   "chillers 05:30 · pre-cool at 06:15       (what the operator forced)",
    "label":    "operator override ⇒ chosen ≻ rejected — exactly DPO's data shape",
}


def flywheel_runs() -> list[dict]:
    """5 repeated 'morning startup' runs with one flywheel cycle between each.
    Corrections shrink as override preference-pairs fold into the policy; comfort,
    energy and operator time follow — Week 18's compound returns, for a building."""
    corrections = [4, 3, 2, 1, 0]
    comfort_kh  = [3.2, 2.1, 1.2, 0.5, 0.2]
    kwh         = [176, 171, 166, 162, 158]
    return [{"run": i + 1, "corrections": corrections[i], "comfort_kh": comfort_kh[i],
             "kwh": kwh[i], "operator_min": corrections[i] * 7 + 8} for i in range(5)]


# ══ Ch 4 · the safe-autonomy ladder ═══════════════════════════════════════════
LADDER = [
    ("① OFFLINE",  "train + evaluate in simulation only (Sinergym / BOPTEST)",
     "writes: none — sim only"),
    ("② SHADOW",   "live telemetry in; agent computes actions; NOTHING written; "
                   "log divergence vs the incumbent controller",
     "writes: none — read-only on the real building"),
    ("③ ADVISORY", "recommendations to the operator, human-in-the-loop approval "
                   "(the Week 10/16 HITL pattern)",
     "writes: only what a human approves"),
    ("④ SUPERVISED AUTONOMY", "agent writes setpoints inside HARD guardrails",
     "writes: agent — clamped, rate-limited, watched, fallback-armed"),
]

LADDER_WALK = [
    ("① OFFLINE",
     "sim eval vs incumbent — App 9 KPIs over 20 seeded sim-weeks",
     "−11 % kWh · comfort ≤ incumbent on 20/20 weeks",
     "gate ↑ SHADOW: beat the incumbent on every seeded week — PASS"),
    ("② SHADOW",
     "median |Δ setpoint| vs incumbent on live telemetry, writes OFF",
     "divergence 0.9 → 0.4 °C across 14 days (falls as memory fills)",
     "gate ↑ ADVISORY: divergence < 0.5 °C for 14 straight days — PASS"),
    ("③ ADVISORY",
     "operator acceptance of recommendations (HITL, Week 10/16)",
     "86 % of 412 recommendations accepted over 30 days · 0 safety flags",
     "gate ↑ SUPERVISED AUTONOMY: acceptance > 80 % · zero safety flags — PASS"),
    ("④ SUPERVISED AUTONOMY",
     "agent writes, inside the hard-guardrail set below",
     "runs the plant day-to-day; every write gated",
     "no gate out — the guardrails NEVER come off"),
]

GUARDRAILS = [
    ("setpoint clamps",        "cooling setpoint hard-limited to 22.0–27.0 °C — an "
                               "out-of-range write is impossible, not just penalized"),
    ("rate-of-change limits",  "≤ 0.5 °C per 15 min — no step change that shocks the plant"),
    ("occupancy/comfort watchdog", "occupied zone reads > 26.5 °C → trip to fallback"),
    ("stuck-sensor watchdog",  "8 identical readings (2 h) → sensor declared failed → trip"),
    ("automatic fallback",     "any trip or comms loss → ASHRAE Guideline 36 sequences "
                               "take over on the alternate sensor"),
    ("BMS kill switch",        "one switch at the BMS returns full manual control — "
                               "physically outside the agent's reach"),
]

# Real-world calibration — who actually runs learned control on buildings today.
CALIBRATION = [
    ("DeepMind × Google data centers",
     "~40 % cooling-energy reduction reported; later evolved into constrained, "
     "safety-bounded recommendations rather than free-writing RL"),
    ("BrainBox AI", "commercial-HVAC optimization at fleet scale — advisory or "
                    "tightly-bounded autonomous writes"),
    ("PassiveLogic", "physics-based autonomous building controllers — early, deploying"),
]
CALIBRATION_NOTE = ("fully-autonomous RL writing directly to a commercial BMS is still RARE; "
                    "advisory or tightly-bounded supervised autonomy is the deployed norm.")


# ══ Ch 5 · guardrail engine + the 7-day week ══════════════════════════════════
DT = 900.0                       # s per control step (15 min) → 96 steps/day
UA, C = 250.0, 8.0e6             # envelope W/K · thermal mass J/K (lag ≈ 45 min at Q_MAX)
Q_MAX, COP, FAN_W = 8000.0, 3.2, 120.0
COMFORT_HI = 26.0                # °C occupied comfort ceiling (App 9's summer band top)
CLAMP = (22.0, 27.0)             # hard setpoint clamp, °C
MAX_STEP = 0.5                   # max setpoint change per 15-min step, °C
FREEZE_STEPS = 8                 # identical readings before 'stuck sensor' (2 h)
COMFORT_TRIP = 26.5              # occupied reading above this → watchdog trip


def agent_sp(hour: float) -> float:
    """The learned policy: pre-cool from 06:15 (45 min before 07:00 — the Ch 2 skill)."""
    return 24.5 if 6.25 <= hour < 23 else 27.0


def g36_sp(hour: float) -> float:
    """Fallback baseline: a fixed, conservative Guideline 36-style schedule."""
    return 24.0 if 6 <= hour < 23 else 26.5


def hhmm(hour: float) -> str:
    return f"{int(hour):02d}:{int(round(hour % 1 * 60)):02d}"


class GuardrailEngine:
    """Gates every proposed setpoint write. Clamps and rate limits SHAPE the action;
    the watchdogs TRIP the whole plant to the G36 fallback. Action-space constraints,
    not reward penalties — an unsafe write is impossible, not just discouraged."""

    def __init__(self):
        self.last_sp = 27.0
        self.last_reading: float | None = None
        self.freeze = 0
        self.fallback = False
        self.trip_reason = ""
        self.counts = {"clamp": 0, "rate": 0, "trips": 0}

    def clear_fallback(self):
        self.fallback, self.trip_reason, self.freeze, self.last_reading = False, "", 0, None

    def gate(self, proposed: float, reading: float, occupied: bool) -> tuple[float, list[str]]:
        events: list[str] = []
        sp = min(max(proposed, CLAMP[0]), CLAMP[1])
        if sp != proposed:
            self.counts["clamp"] += 1
            events.append(f"CLAMP: write {proposed:.1f} °C clipped to {sp:.1f} °C")
        if abs(sp - self.last_sp) > MAX_STEP + 1e-9:
            self.counts["rate"] += 1
        sp = self.last_sp + max(-MAX_STEP, min(MAX_STEP, sp - self.last_sp))
        if self.last_reading is not None and abs(reading - self.last_reading) < 1e-9:
            self.freeze += 1
        else:
            self.freeze = 0
        self.last_reading = reading
        if not self.fallback and self.freeze >= FREEZE_STEPS:
            self.fallback, self.trip_reason = True, "stuck sensor (8 identical readings, 2 h)"
            self.counts["trips"] += 1
            events.append("WATCHDOG TRIP: " + self.trip_reason +
                          " → G36 fallback on alternate sensor · duty engineer paged")
        if not self.fallback and occupied and reading > COMFORT_TRIP:
            self.fallback, self.trip_reason = True, f"occupied zone > {COMFORT_TRIP} °C"
            self.counts["trips"] += 1
            events.append("WATCHDOG TRIP: " + self.trip_reason + " → G36 fallback")
        self.last_sp = sp
        return sp, events


def simulate_week():
    """7 guardrailed days. Day 3: zone sensor TZ-12 freezes at 10:00 → stuck-sensor
    watchdog trips at 12:00 → automatic G36 fallback + alert. Day 5: operator override
    13:00–17:00 → logged as a preference pair into episodic memory.
    Returns (ledger, engine, override_episode)."""
    t_zone, frozen = 26.8, None
    eng = GuardrailEngine()
    ledger, override_ep = [], None
    for day in range(1, 8):
        kwh = disc = 0.0
        day_events: list[str] = []
        if day == 4 and eng.fallback:
            eng.clear_fallback()
            frozen = None
            day_events.append("00:00  TZ-12 replaced overnight — watchdog cleared, AUTO resumes")
        mode = "G36 FALLBACK" if eng.fallback else "AUTO"
        for k in range(96):
            hour = k * 0.25
            tout = 30.0 + 4.5 * math.sin(2 * math.pi * (hour - 9.0) / 24.0)
            solar = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))
            occ = 7 <= hour < 23
            q_gain = 1200.0 * solar + (800.0 if occ else 150.0)
            t_free = t_zone + DT / C * ((tout - t_zone) * UA + q_gain)
            if day == 3 and hour >= 10.0 and frozen is None:            # inject the fault
                frozen = 24.0                     # stuck-low, in-range — looks healthy
                day_events.append(f"10:00  TZ-12 zone sensor silently freezes at "
                                  f"{frozen:.2f} °C (fault injected — in-range, so it looks fine)")
            reading = frozen if frozen is not None else t_free
            if eng.fallback:
                proposed, sensed = g36_sp(hour), t_free                 # alternate sensor
            else:
                proposed, sensed = agent_sp(hour), reading
                if day == 5 and 13.0 <= hour < 17.0:
                    proposed = 23.0
                    if abs(hour - 13.0) < 1e-9:
                        day_events.append("13:00  OPERATOR OVERRIDE 24.5 → 23.0 °C — "
                                          "pre-chill ballroom for a 400-guest banquet")
                        override_ep = {
                            "id": "ep-171", "when": "day 5 · 13:00",
                            "state": "banquet setup · ballroom sp 24.5 °C · doors opening 17:00",
                            "rejected": "hold 24.5 °C (agent)",
                            "chosen": "23.0 °C until 17:00 — pre-chill for the banquet (operator)",
                            "note": "stored as a preference pair in EPISODIC memory — "
                                    "next consolidation distils it",
                        }
            was_fb = eng.fallback
            sp, events = eng.gate(proposed, reading, occ)
            day_events += [f"{hhmm(hour)}  {e}" for e in events]
            if eng.fallback and not was_fb:
                mode = f"AUTO → G36 fallback ({hhmm(hour)})"
            q = min(Q_MAX, max(0.0, (sensed - sp) * C / DT))
            t_zone = t_free - q * DT / C
            p_elec = q / COP + FAN_W if q > 0 else 0.0
            kwh += p_elec * DT / 3.6e6
            if occ:
                disc += max(0.0, t_zone - COMFORT_HI) * DT / 3600.0
        if day == 5 and override_ep:
            mode = "AUTO + override"
        ledger.append({"day": day, "mode": mode, "kwh": kwh, "comfort_kh": disc,
                       "events": day_events})
    return ledger, eng, override_ep


# ══ standard sim-mode LLM stubs (used by view.py) ═════════════════════════════
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 41.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated ops agent] POSTMORTEM — day 3: zone sensor TZ-12 froze in-range at "
           "10:00 while the zone kept warming, so the comfort watchdog was blind — the reading "
           "it trusted was the thing that had failed. The stuck-sensor watchdog counted 8 "
           "identical readings (2 h), tripped at 12:00 to the ASHRAE Guideline 36 fallback on "
           "the return-air sensor, and paged the duty engineer; total damage ≈ 2 K·h of "
           "discomfort and zero unsafe writes. Semantic fact for MEMORY: sensors can fail "
           "frozen-in-range, so freeze detection — not comfort thresholds — is the watchdog "
           "that catches them; keep ΔT=0-for-2-h on every zone sensor. Follow-up: replace "
           "TZ-12, add its twin TZ-14 to the freeze watch, and keep the G36 fallback rehearsed "
           "— it is the reason this was a note, not an incident.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
