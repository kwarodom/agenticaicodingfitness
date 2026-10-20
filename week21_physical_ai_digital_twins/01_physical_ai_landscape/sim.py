#!/usr/bin/env python3
"""Physical AI landscape simulator — learn the NVIDIA digital-twin map with no GPU."""
from __future__ import annotations

import time

# ── the three-computer model (Ch 2) ───────────────────────────────────────────
# (computer, role, where it lives, what runs on it, week-21 building-twin example)
THREE_COMPUTERS = [
    ("DGX", "TRAIN the models",
     "data center / your desk (Spark)",
     "model training · RL policy training · fine-tuning",
     "train the HVAC control policy (App 09) and the copilot model (App 11)"),
    ("OVX / Omniverse", "SIMULATE the world (sim2real)",
     "data center / on-prem workstation",
     "Omniverse Kit apps · OpenUSD stages · physics + energy sims",
     "the hotel twin: USD scene (Apps 02-04) + what-if energy sims (Apps 06-08)"),
    ("Jetson / AGX / IGX", "DEPLOY at the edge — ACT",
     "in the building / on the robot",
     "inference · NIM microservices · control agents",
     "agents watching and actuating AHUs/chillers on-site (Apps 09-11)"),
]

# ── Omniverse's 2026 shape (Ch 3): libraries + microservices, not an app suite ─
OMNIVERSE_PIECES = [
    ("Omniverse Kit SDK", "SDK", "build your OWN twin app — the twin viewer is yours to make"),
    ("OpenUSD", "format", "the core scene description everything composes into"),
    ("Kit App Streaming", "microservice", "stream a Kit app's viewport to any browser"),
    ("USD Code NIM", "microservice", "LLM microservice that writes/explains USD Python"),
    ("USD Search NIM", "microservice", "natural-language search over huge USD asset libraries"),
]

# ── Omniverse vs Cosmos (Ch 4): deterministic sim vs generative world model ───
OMNI_VS_COSMOS = [
    ("what it is", "deterministic physics simulation", "generative world foundation models (WFMs)"),
    ("answers", "what WILL happen, given physics", "what COULD plausibly happen (video/worlds)"),
    ("same input twice", "same answer — reproducible", "different plausible samples"),
    ("building twin use", "the twin itself: what-if you can act on", "synthetic data: rare events to train on"),
    ("feeds", "validation, control decisions", "policy training, perception training"),
]

# ── deployments, honestly tiered (Ch 5) ───────────────────────────────────────
# (deployment, domain, tier, what actually happened)
DEPLOYMENTS = [
    ("BMW Debrecen plant", "factory", "PROVEN",
     "planned entirely virtually before construction — ~30% planning-efficiency gain"),
    ("Foxconn AI-server plants", "factory", "PROVEN",
     "Guadalajara + Houston plants twinned to plan/validate production"),
    ("Amazon Robotics warehouses", "warehouse", "PROVEN",
     "robot fleets trained/validated in twins before deployment (the Mega pattern)"),
    ("Lowe's retail stores", "retail", "PROVEN",
     "store twins for layout and operations"),
    ("Data centers", "data center", "PROVEN",
     "Cadence Reality DC · Schneider Electric · ETAP · Vertiv — power/cooling twins"),
    ("Hotels / hospitals / offices", "buildings-ops", "FRONTIER",
     "no established Omniverse reference deployments — this is the gap Week 21 targets"),
]

_MODELS = ["nemotron-3-super:120b-a12b", "nemotron-3-nano:30b-a3b"]
_TOK = {"nemotron-3-super:120b-a12b": 20.0, "nemotron-3-nano:30b-a3b": 54.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated twin copilot] For a 30-floor Bangkok hotel, the DSX 'twin as operating "
           "system' idea maps straight onto the four Week-21 layers: SCENE is the OpenUSD model "
           "of every floor, riser and AHU, and STATE binds live BACnet/IoT telemetry to those "
           "prims so the twin mirrors the building minute by minute. SIMULATION — a calibrated "
           "energy model plus fast surrogates — answers what-if questions like pre-cooling before "
           "the afternoon check-in peak without experimenting on guests, and AGENTS act on the "
           "verified answers, turning the twin from a 3D viewer into the hotel's operating system.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
