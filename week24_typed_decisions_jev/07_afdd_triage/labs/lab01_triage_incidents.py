#!/usr/bin/env python3
"""Lab 07-1 · HVAC fault triage — code computes, Jev judges, an engineer decides.

Five synthetic incidents. For each one:
  1. CODE computes the numeric facts: stale data? bad quality? how far off setpoint?
  2. JEV reads the operator note + those computed flags → investigation queue,
     safety concern, severity. (Jev never does the arithmetic.)
  3. CODE applies a deterministic policy: stale or bad data overrides the queue;
     any safety signal → safety review. Everything ends at engineer_review.

No BACnet write, no setpoint change, no work order. A proposal only.

Run: .venv/bin/python week24/07_afdd_triage/labs/lab01_triage_incidents.py
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, guarded, noul, score, show, step, table, validate  # noqa: E402

INCIDENTS = Path(__file__).resolve().parents[1] / "data" / "incidents.jsonl"

# TEACHING thresholds — not validated AltoTech limits, comfort standards or AFDD rules.
STALE_AFTER_S = 600          # telemetry older than 10 minutes is stale
WARM_DEVIATION_C = 2.0       # more than 2 °C above setpoint

# A deterministic safety backstop that runs no matter what the model says.
SAFETY_WORDS = re.compile(r"\b(smoke|fire|burning|sparks?|flood(ing)?|injur(y|ed))\b", re.I)

QUESTIONS = guarded({
    "queue": choice(
        "Using `operator_note` and the `computed` flags, select the next investigation queue. "
        "Do not claim a proven root cause.", {
            "cooling": "Comfort or cooling-performance investigation.",
            "sensor": "Sensor plausibility or calibration investigation.",
            "connectivity": "Offline gateway, missing or stale telemetry investigation.",
            "other": "Insufficient or conflicting evidence; engineer triage.",
        }),
    "safety_concern": noul(
        "Does `operator_note` explicitly mention smoke, electrical burning, fire, injury, flooding or "
        "another immediate safety concern?"),
    "severity": score("How severe is the reported operational impact?", [
        "Informational or no impact stated.",
        "Localized discomfort or limited degradation.",
        "Significant service disruption.",
        "Possible immediate safety incident.",
    ]),
})


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def prepare(raw: dict) -> dict:
    """Validate the reading, compute flags IN CODE, and build the state Jev sees."""
    for k in ("zone_c", "setpoint_c", "age_seconds"):
        if not is_number(raw.get(k)):
            raise ValueError(f"invalid numeric field: {k}")
    if raw["age_seconds"] < 0 or type(raw.get("quality_ok")) is not bool:
        raise ValueError("invalid age or quality flag")
    deviation = round(raw["zone_c"] - raw["setpoint_c"], 2)
    return {
        "operator_note": raw["operator_note"],
        "computed": {                                  # facts, not guesses
            "stale": raw["age_seconds"] > STALE_AFTER_S,
            "bad_quality": not raw["quality_ok"],
            "warm_deviation": deviation > WARM_DEVIATION_C,
            "deviation_c": deviation,
        },
    }


def policy(state: dict, a: dict) -> dict:
    flags = state["computed"]
    data_problem = flags["stale"] or flags["bad_quality"]
    keyword_hit = bool(SAFETY_WORDS.search(state["operator_note"]))
    return {
        "route": "engineer_review",
        # Deterministic override: never trust a comfort diagnosis built on bad data.
        "proposed_queue": "connectivity_or_data_quality" if data_problem else a["queue"]["choice"],
        # OR, never AND: a low model probability can never cancel the keyword backstop.
        "safety_review": a["safety_concern"]["noul"] >= 0.20 or keyword_hit,
        "execute": False, "no_bacnet_write": True, "no_setpoint_change": True,
    }


def load() -> list[dict]:
    return [json.loads(l) for l in INCIDENTS.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    banner("Lab 07-1 · AFDD investigation triage", "code computes flags → Jev judges → policy overrides → engineer")
    incidents = load()

    step(1, "what code computes BEFORE the model sees anything")
    first = incidents[0]["state"]
    print(f"   raw reading: zone {first['zone_c']} °C, setpoint {first['setpoint_c']} °C, "
          f"age {first['age_seconds']} s, quality_ok {first['quality_ok']}")
    st = prepare(first)
    print(f"✓ computed:   {json.dumps(st['computed'])}")
    print("   Jev gets these as facts. It is never asked 'is 29.1 more than 2 above 24.0?'")

    step(2, "ask Jev about that incident")
    resp = ask(st, QUESTIONS)
    validate(resp, QUESTIONS)
    show(resp, top=4)

    step(3, f"triage all {len(incidents)} incidents")
    rows = []
    for inc in incidents:
        st = prepare(inc["state"])
        a = validate(ask(st, QUESTIONS, quiet=True), QUESTIONS)
        rec = policy(st, a)
        f = st["computed"]
        flags = ",".join(k for k in ("stale", "bad_quality", "warm_deviation") if f[k]) or "-"
        rows.append([inc["id"], f"{f['deviation_c']:+.1f}", flags, a["queue"]["choice"],
                     f"{a['safety_concern']['noul']:.2f}", f"{a['severity']['score']:.2f}",
                     rec["proposed_queue"], "YES" if rec["safety_review"] else "-"])
    table(rows, ["id", "Δ°C", "code flags", "jev queue", "safety", "sev", "proposed queue", "safety rev"])

    step(4, "read the overrides")
    print("→ fault-2: Jev may say 'connectivity' anyway — but the override does not depend on it")
    print("→ fault-4: bad_quality=True → data-quality queue, whatever the reading claims")
    print("⚠ existing fire/safety alarms run independently of this classifier — always")
    print("═ execute: False · route: engineer_review · no_bacnet_write: True · no_setpoint_change: True")


if __name__ == "__main__":
    main()
