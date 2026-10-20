#!/usr/bin/env python3
"""Capstone II simulator — canned LLM streams + the standard sim interface.

Pure stdlib. In SIM mode, `stream_generate` picks the canned answer that matches
the prompt so every agent-reasoning moment reads like a real one; in REAL mode
view.py streams from your endpoint instead and these strings are never used.
"""
from __future__ import annotations

import time

_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = {
    "vss": ("[VSS agent] Between 22:00 and 01:00 the network raised 5 raw events; the "
            "verification VLM confirmed 3 and rejected 2 (a scheduled bus dwell and a "
            "lighting artifact). The only multi-camera incident is the 23:15 sideswipe at "
            "Sukhumvit × Asoke — CAM-13 and CAM-14 corroborate. No plates or faces were "
            "extracted; clips never left the city DGX."),
    "brief": ("[city operator] Overnight summary for the duty engineer: one verified "
              "sideswipe at I-11 (23:15, cleared 23:52, reroute via S-01/S-02 held v/c at "
              "0.89), one verified stop-anomaly in Old Town (00:00, tow dispatched), and a "
              "SUB-B partial trip at 01:00 ridden through with D2 flexibility — hospital "
              "untouched. Two false alerts were filtered before any human saw them. "
              "Consolidation stored 1 new fact and 1 reroute playbook."),
    "postmortem": ("[postmortem note → semantic memory] The 23:15 I-11 sideswipe was "
                   "detected by CV in 2 s, VLM-verified in 9 s, and the corridor reroute "
                   "was approved by the duty engineer in 3 min (advisory rung). Diversion "
                   "kept Sukhumvit under jam threshold. Store: reroute playbook v1 for "
                   "I-11; keep signal retiming at ADVISORY until 5 clean approvals."),
}


def canned_for(prompt: str) -> str:
    p = prompt.lower()
    if "verified" in p or "camera" in p or "vss" in p:
        return _CANNED["vss"]
    if "postmortem" in p or "consolidat" in p:
        return _CANNED["postmortem"]
    return _CANNED["brief"]


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in canned_for(prompt).split(" "):
        yield w + " "
        time.sleep(delay)
