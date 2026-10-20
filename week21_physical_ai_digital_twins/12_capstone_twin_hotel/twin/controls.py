#!/usr/bin/env python3
"""The AGENTS (control half) — guardrailed operator + autonomy ladder  (Apps 9–10).

The learned policy NEVER touches the building directly. Every proposal passes
guardrails (setpoint clamps, rate-of-change limits, a sensor watchdog) and, when
anything smells wrong, the operator falls back to the deterministic ASHRAE
Guideline 36 sequence — the same "hard floor" a real deployment keeps. Autonomy
is earned one rung at a time: offline → shadow → advisory → supervised.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BAND_C = (21.0, 25.0)          # hard setpoint clamp — comfort is non-negotiable
MAX_STEP_C = 0.5               # max change per decision (rate-of-change limit)
WATCHDOG_STALE = 3             # identical sensor polls before the watchdog trips

LADDER = ["offline", "shadow", "advisory", "supervised"]
GATES = {                       # what it takes to EARN the next rung
    "offline":    "policy beats baseline in the calibrated sim (≥5 % kWh, 0 comfort misses)",
    "shadow":     "7 days shadowing live ops: ≥95 % agreement with G36, 0 guardrail hits",
    "advisory":   "operators accept ≥80 % of advice over 7 days; every rejection reviewed",
    "supervised": "acts alone inside guardrails; humans page-able; auto-fallback armed",
}


def g36_setpoint(hour: int) -> float:
    """The Guideline 36 fallback sequence, condensed: occupied 23 °C, setback nights."""
    return 23.0 if 6 <= hour < 23 else 24.5


@dataclass
class Policy:
    """Baseline schedule + learned tweaks (what App 9's RL / App 10's skills produce)."""
    tweaks: dict = field(default_factory=lambda: {
        7: -0.5,    # pre-cool before the morning peak (learned from the surrogate)
        13: +0.5,   # ride thermal mass through the tariff peak
        22: +1.0,   # earlier setback — floor-5 rooms hold temp 40 min after lights-out
    })

    def propose(self, hour: int) -> float:
        return g36_setpoint(hour) + self.tweaks.get(hour, 0.0)


@dataclass
class Decision:
    hour: int
    proposed: float
    applied: float
    rung: str
    note: str = ""


class Operator:
    """The state machine that owns the building's setpoints."""

    def __init__(self, policy: Policy, rung: str = "shadow"):
        self.policy = policy
        self.rung = rung
        self.last_applied = g36_setpoint(0)
        self.fallback = False
        self._sensor_history: list[float] = []
        self.log: list[Decision] = []
        self.violations = 0
        self.agreement = [0, 0]        # [agree, total] while shadowing
        self.accepted = [0, 0]         # [accepted, total] while advisory

    # ── guardrails ────────────────────────────────────────────────────────────
    def _clamp(self, sp: float) -> tuple[float, str]:
        note = ""
        lo, hi = BAND_C
        if not (lo <= sp <= hi):
            self.violations += 1
            sp, note = min(hi, max(lo, sp)), f"clamped to band {lo}–{hi} °C"
        if abs(sp - self.last_applied) > MAX_STEP_C:
            step = MAX_STEP_C if sp > self.last_applied else -MAX_STEP_C
            sp, note = round(self.last_applied + step, 2), (note + " · " if note else "") + \
                f"rate-limited to ±{MAX_STEP_C} °C/step"
        return sp, note

    def watchdog(self, sensor_value: float) -> bool:
        """Feed the zone sensor each tick; trips (→ True) on a frozen reading."""
        self._sensor_history.append(sensor_value)
        recent = self._sensor_history[-WATCHDOG_STALE:]
        if len(recent) == WATCHDOG_STALE and len(set(recent)) == 1:
            self.fallback = True
            return True
        return False

    # ── one decision tick ─────────────────────────────────────────────────────
    def step(self, hour: int, human_accepts: bool = True) -> Decision:
        proposed = self.policy.propose(hour)
        baseline = g36_setpoint(hour)

        if self.fallback:                                  # watchdog tripped → G36 holds
            d = Decision(hour, proposed, baseline, self.rung, "FALLBACK → G36 sequence")
        elif self.rung in ("offline", "shadow"):           # log only, never act
            self.agreement[1] += 1
            if abs(proposed - baseline) <= MAX_STEP_C:
                self.agreement[0] += 1
            d = Decision(hour, proposed, baseline, self.rung, "shadow — logged, not applied")
        elif self.rung == "advisory":                      # a human clicks apply
            self.accepted[1] += 1
            if human_accepts:
                self.accepted[0] += 1
                sp, note = self._clamp(proposed)
                d = Decision(hour, proposed, sp, self.rung, note or "advised → accepted")
            else:
                d = Decision(hour, proposed, baseline, self.rung, "advised → REJECTED (override)")
        else:                                              # supervised: acts inside guardrails
            sp, note = self._clamp(proposed)
            d = Decision(hour, proposed, sp, self.rung, note or "applied autonomously")
        self.last_applied = d.applied
        self.log.append(d)
        return d

    # ── the ladder ────────────────────────────────────────────────────────────
    def gate_report(self) -> tuple[bool, str]:
        if self.rung == "shadow":
            a, t = self.agreement
            ok = t >= 7 and a / t >= 0.95 and self.violations == 0
            return ok, f"agreement {a}/{t} ({100 * a / max(t, 1):.0f}%), {self.violations} guardrail hits"
        if self.rung == "advisory":
            a, t = self.accepted
            ok = t >= 7 and a / t >= 0.80
            return ok, f"accepted {a}/{t} ({100 * a / max(t, 1):.0f}%)"
        return False, "top rung — supervision stays"

    def promote(self) -> str:
        ok, why = self.gate_report()
        if ok and self.rung != LADDER[-1]:
            self.rung = LADDER[LADDER.index(self.rung) + 1]
        return f"{'PROMOTED to ' + self.rung.upper() if ok else 'HELD at ' + self.rung.upper()} — {why}"

    def reset_fallback(self, note: str = "sensor repaired") -> str:
        self.fallback = False
        self._sensor_history.clear()
        return f"fallback cleared ({note}); resuming {self.rung} operation"


if __name__ == "__main__":
    op = Operator(Policy(), rung="supervised")
    for h in (7, 8, 9):
        print(op.step(h))
    op.watchdog(0.31); op.watchdog(0.31); print("trip:", op.watchdog(0.31))
    print(op.step(10))
