#!/usr/bin/env python3
"""Solution · Exercise 07 — flags computed in code, never by the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ex07_compute_flags as ex  # noqa: E402

STALE_AFTER_S = 600
DEVIATION_C = 2.0
HUMID_PCT = 70


def compute_flags(reading: dict) -> dict:
    deviation = reading["zone_c"] - reading["setpoint_c"]
    return {
        "stale": reading["age_seconds"] > STALE_AFTER_S,
        "bad_quality": reading["quality_ok"] is False,
        "warm_deviation": deviation > DEVIATION_C,        # strictly greater: 2.0 is not a deviation
        "cold_deviation": -deviation > DEVIATION_C,
        "humidity_high": reading["humidity_pct"] > HUMID_PCT,
        "deviation_c": round(deviation, 1),
    }


ex.compute_flags = compute_flags

if __name__ == "__main__":
    ex.main()
