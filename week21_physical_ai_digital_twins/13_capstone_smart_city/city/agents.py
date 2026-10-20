#!/usr/bin/env python3
"""City AGENTS — a sovereign, self-evolving fleet (Apps 9–11 + Weeks 18/20, city-scale).

Three things live here:
  • the SOVEREIGNTY AUDIT — what data exists, where it may flow, what stays on-box
    (video NEVER leaves the city's own GPUs; the blueprint's local/local_shared/
    remote-NIM modes made concrete);
  • the AUTONOMY LADDER for city actions — a city is not a hotel: the blast radius
    of a bad action is bigger, so rungs are earned per ACTION TYPE, not per agent;
  • the FLYWHEEL — incidents become episodes, episodes consolidate into semantic
    facts and procedural playbooks the fleet reuses next time.
"""
from __future__ import annotations

# ── sovereignty audit ─────────────────────────────────────────────────────────
DATA_CLASSES = [
    ("camera video (24 streams)",   "SOVEREIGN", "never leaves city DGX — VLM verifies on-box"),
    ("CV events / alerts",          "SOVEREIGN", "Kafka on-prem; plates/faces never extracted"),
    ("signal-timing plans",         "SOVEREIGN", "operational infrastructure data"),
    ("district energy telemetry",   "SOVEREIGN", "grid-critical; on-prem historian"),
    ("weather forecasts (inbound)", "PUBLIC",    "pulled from met service / Earth-2 — inbound only"),
    ("anonymized congestion stats", "SHAREABLE", "aggregates only, k-anonymized, opt-in publish"),
]

MODEL_MODES = [
    ("local",        "dedicated city GPUs per model",         "max sovereignty · max hardware"),
    ("local_shared", "all models share one GPU",              "small city / pilot — still sovereign"),
    ("remote NIM",   "hosted endpoints (build.nvidia.com)",   "fastest start — video-derived data leaves site: NOT the sovereign path"),
]

# ── autonomy ladder, per action type (rung 1=offline 2=shadow 3=advisory 4=supervised) ──
ACTION_LADDER = [
    ("publish verified alert to ops map",  4, "bounded: verified alerts only, human dashboard"),
    ("notify hospital of inbound trauma",  4, "bounded: template message, confirmed incidents only"),
    ("retime traffic signals (corridor)",  3, "ADVISORY — engineer approves each plan, for now"),
    ("dispatch road crew / tow",           3, "ADVISORY — dispatcher clicks approve"),
    ("district demand-response call",      4, "bounded: opt-in buildings, hospital excluded, ≤2h"),
    ("substation switching",               1, "OFFLINE ONLY — humans + interlocks, agent observes"),
]

GUARDRAILS = [
    "signal plans clamped to engineer-approved envelopes (min green, ped phases)",
    "DR calls: opt-in buildings only · hospital hard-excluded · auto-restore ≤ 2 h",
    "any watchdog trip → previous timing plan / grid schedule restored",
    "kill switch in the traffic-management centre, outside the agent's reach",
]


# ── the flywheel: episodes → facts → playbooks ────────────────────────────────
class Flywheel:
    def __init__(self):
        self.episodic: list[dict] = []
        self.semantic: list[str] = [
            "Asoke interchange (S-06) jams first when Sukhumvit demand > 0.95 v/c",
            "bus dwell at S-06 stop 3 reads as stop-anomaly ~nightly (known FP source)",
        ]
        self.procedural: list[str] = ["evening-peak-DR-playbook (v3)"]

    def record(self, episode: dict) -> None:
        self.episodic.append(episode)

    def consolidate(self) -> list[str]:
        """Distil tonight's episodes into durable memory (Week 18's loop, city-scale)."""
        new = []
        for ep in self.episodic:
            if ep.get("kind") == "incident" and ep.get("reroute_worked"):
                new.append(f"playbook: incident at {ep['where']} → reroute {ep['route']} "
                           f"holds v/c under {ep['vc_after']}")
                self.procedural.append(f"incident-reroute-{ep['where']}")
            if ep.get("kind") == "grid" and ep.get("shortfall_mw", 1) == 0:
                new.append("fact: D2 flex + storage rides through a SUB-B partial trip unaided")
                self.semantic.append("SUB-B partial trip coverable by D2 flex (verified 1×)")
        return new
