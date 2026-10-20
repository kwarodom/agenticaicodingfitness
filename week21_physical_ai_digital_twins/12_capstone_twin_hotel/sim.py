#!/usr/bin/env python3
"""Capstone simulator — canned LLM streams + shared constants, no GPU needed.

Pure stdlib. The demos and the twin/ package import this for:
  • installed_models / tok_s / stream_generate — the standard sim LLM interface
    (stream_generate picks a canned answer that matches the prompt, so the SIM
    reads like a real agent-reasoning moment)
  • VARIANTS — the one-line capstone variant briefs (hospital / office / factory / city)
"""
from __future__ import annotations

import time

# ── the standard sim LLM interface ────────────────────────────────────────────
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


# canned answers, keyed by a keyword the prompt will contain — so each
# agent-reasoning moment in the demos streams a RELEVANT simulated answer.
_CANNED = [
    ("postmortem",
     "[simulated] Postmortem, sensor freeze on VAV-05-02: the flow sensor reported the same "
     "value for 3 consecutive polls, the watchdog tripped, and the operator correctly fell "
     "back to the Guideline 36 sequence — zero comfort excursions. Root cause is likely a "
     "stuck transducer (4 similar episodes in the flywheel). Recommend: keep the watchdog at "
     "3 stale polls, open a ROUTINE work order for the sensor, and hold the SUPERVISED rung — "
     "the fallback worked exactly as designed."),
    ("morning brief",
     "[simulated] Morning brief, 06:00: the twin ran clean overnight — floor-5 zones held "
     "23±0.4°C, chillers alternated CH-1/CH-2 per schedule, and the pre-cool the surrogate "
     "recommended saved ~9% on the 06:00–09:00 peak. One watch item: room 1203 logged its "
     "5th thermostat override this month — the flywheel's consolidated skill says check "
     "sensor calibration before rolling a truck. VIP suite 1512 arrives 09:30; pre-condition "
     "within the guardrail band only."),
    ("chiller",
     "[simulated] Diagnosis, CH-2 low delta-P alarm: the graph shows CH-2 feeds AHU-3/-4 via "
     "CHWP-2; the TSDB shows flow dropped 34% over 20 min while pump speed held — consistent "
     "with a fouled strainer per the O&M note, not a refrigerant fault. Confidence high: the "
     "twin's simulation reproduces the AHU-3 supply-temp drift only under a flow restriction. "
     "Action: CRITICAL work order on CH-2 strainer, shift load to CH-1, verify recovery in "
     "the session layer within 30 min."),
]
_DEFAULT = ("[simulated] The living twin closes the loop: the SCENE mirrors the building, the "
            "STATE binds live telemetry to it, the SIMULATION answers what-if before anything "
            "is touched, and the AGENTS act through guardrails and learn from every episode — "
            "one system, running on hardware you own, $0 per token.")


def _pick(prompt: str) -> str:
    p = prompt.lower()
    for key, text in _CANNED:
        if key in p:
            return text
    return _DEFAULT


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _pick(prompt).split(" "):
        yield w + " "
        time.sleep(delay)


# ── the capstone variant briefs (one line each — see README / Ch 7) ──────────
VARIANTS = [
    ("Hospital",  "same stack, harder gates — N+1 redundancy in the graph, IAQ/pressure-cascade "
                  "zones in the RC model, compliance evidence exported from the episode store"),
    ("Office",    "tenant-comfort SLAs per lease in the policy, after-hours billing from the "
                  "TSDB join, the copilot answers facility tickets against the twin"),
    ("Factory",   "swap the room grid for lines and cells — the Mega blueprint pattern: train "
                  "robot fleets in the twin before they touch the real floor"),
    ("City district", "many buildings, one grid signal — CityLearn-style demand response, the "
                  "autonomy ladder now gates a fleet of building operators"),
]
