#!/usr/bin/env python3
"""Exercise 07 · Compute the flags yourself — the part Jev must never do.

Implement compute_flags(reading) so it returns a dict with EXACTLY these keys:

    stale          age_seconds > 600
    bad_quality    quality_ok is False
    warm_deviation zone_c - setpoint_c  >  2.0
    cold_deviation setpoint_c - zone_c  >  2.0     (new!)
    humidity_high  humidity_pct > 70               (new!)
    deviation_c    zone_c - setpoint_c, rounded to 1 decimal

(Teaching thresholds only — real sites load approved thresholds per equipment.)

Run: .venv/bin/python week24/07_afdd_triage/exercises/ex07_compute_flags.py
The offline asserts run first (free). Then your flags feed a live Jev triage.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1] / "labs"))
from jevkit import ask, banner, check, step, table, validate  # noqa: E402


def compute_flags(reading: dict) -> dict:
    """reading = {"zone_c", "setpoint_c", "humidity_pct", "age_seconds", "quality_ok", ...}"""
    # ── TODO ── return the six flags described in the docstring above.
    return {}


# ─────────────────────────── checker — no need to edit below ────────────────
CASES = [
    ("warm and fresh", dict(zone_c=28.2, setpoint_c=24.0, humidity_pct=60, age_seconds=90, quality_ok=True),
     dict(stale=False, bad_quality=False, warm_deviation=True, cold_deviation=False, humidity_high=False, deviation_c=4.2)),
    ("cold room", dict(zone_c=19.5, setpoint_c=23.0, humidity_pct=45, age_seconds=30, quality_ok=True),
     dict(stale=False, bad_quality=False, warm_deviation=False, cold_deviation=True, humidity_high=False, deviation_c=-3.5)),
    ("exactly 2.0 warm is NOT a deviation", dict(zone_c=26.0, setpoint_c=24.0, humidity_pct=50, age_seconds=10, quality_ok=True),
     dict(stale=False, bad_quality=False, warm_deviation=False, cold_deviation=False, humidity_high=False, deviation_c=2.0)),
    ("stale + humid", dict(zone_c=24.5, setpoint_c=24.0, humidity_pct=78, age_seconds=601, quality_ok=True),
     dict(stale=True, bad_quality=False, warm_deviation=False, cold_deviation=False, humidity_high=True, deviation_c=0.5)),
    ("bad quality", dict(zone_c=35.0, setpoint_c=23.0, humidity_pct=50, age_seconds=90, quality_ok=False),
     dict(stale=False, bad_quality=True, warm_deviation=True, cold_deviation=False, humidity_high=False, deviation_c=12.0)),
]


def main() -> None:
    banner("Exercise 07 · compute the flags in code")
    step(1, "offline asserts — free, exact")
    passed = 0
    for name, reading, want in CASES:
        got = compute_flags(reading)
        passed += check(got == want, f"{name}", f"{name}: expected {want}, got {got}")
    if passed < len(CASES):
        print(f"\n⚠ {passed}/{len(CASES)} pass — fix compute_flags(), save, run again. No API call was made.")
        sys.exit(1)

    step(2, "your flags feed a live Jev triage")
    from lab01_triage_incidents import QUESTIONS, load
    rows = []
    for inc in load():
        raw = inc["state"]
        state = {"operator_note": raw["operator_note"], "computed": compute_flags(raw)}
        a = validate(ask(state, QUESTIONS, quiet=True), QUESTIONS)
        f = state["computed"]
        on = ",".join(k for k in ("stale", "bad_quality", "warm_deviation", "cold_deviation", "humidity_high") if f[k])
        rows.append([inc["id"], on or "-", a["queue"]["choice"], f"{a['safety_concern']['noul']:.2f}"])
    table(rows, ["id", "your flags", "jev queue", "safety"])
    print("◆ fault-4 ('reads 35 C but feels cold'): which queue did Jev pick, given bad_quality?")
    print("═ execute: False · no_bacnet_write: True · no_setpoint_change: True")


if __name__ == "__main__":
    main()
