#!/usr/bin/env python3
"""The Smart City Blueprint pipeline, simulated faithfully (NVIDIA VSS 3.0).

Real components mirrored here (docs.nvidia.com/vss/latest/smartcity-docs/):
  • RTVI CV  — real-time detection/tracking with RTDETR or GDINO
  • Kafka    — every CV event is a message on a topic
  • Alert Verification Service — a VLM re-watches the clip and confirms/rejects
  • VSS Agents — the natural-language interface over everything
  • VIOS/NVStreamer — stream management (we just count streams)

Stream budgets are the documented per-GPU numbers; the SIM enforces them so the
capacity math you learn is the real math.
"""
from __future__ import annotations

from . import world

# Documented streams-per-GPU (Smart City Blueprint quickstart).
STREAM_BUDGET = {
    "H100":                    {"rtdetr": 30, "gdino": 12},
    "L40S":                    {"rtdetr": 30, "gdino": 6},
    "RTX PRO 6000 Blackwell":  {"rtdetr": 30, "gdino": 12},
}

# Real endpoint map from the quickstart (what you'd see on a live deployment).
ENDPOINTS = [
    ("VSS-UI (chat)",        ":7777"),   ("Video-Analytics-UI (map)", ":3002"),
    ("Video-Analytics-API",  ":8081"),   ("VST-MCP",                  ":8001"),
    ("VA-MCP",               ":9901"),   ("LLM-NIM",                  ":30081"),
    ("VLM-NIM",              ":30082"),  ("NvStreamer",               ":31000"),
]

INCIDENT_TYPES = ["collision:rear-end", "collision:sideswipe", "collision:head-on",
                  "stop-anomaly", "anomalous-movement", "congestion"]


def plan_deployment(gpu: str) -> dict:
    """Fit the 24-camera network onto one GPU: how many of each detector fit?"""
    budget = STREAM_BUDGET[gpu]
    want = {"rtdetr": sum(1 for c in world.CAMERAS if c["model"] == "rtdetr"),
            "gdino":  sum(1 for c in world.CAMERAS if c["model"] == "gdino")}
    # gdino streams are the scarce resource; rtdetr rides in the remaining share.
    fit_g = min(want["gdino"], budget["gdino"])
    frac_left = 1.0 - fit_g / budget["gdino"] if budget["gdino"] else 1.0
    fit_r = min(want["rtdetr"], int(budget["rtdetr"] * max(frac_left, 0)))
    return {"gpu": gpu, "want": want, "fit": {"gdino": fit_g, "rtdetr": fit_r},
            "ok": fit_g >= want["gdino"] and fit_r >= want["rtdetr"]}


# A deterministic night of CV events (the raw, UNVERIFIED detections).
RAW_EVENTS = [
    {"t": "22:47", "cam": "CAM-11", "watch": "S-06", "type": "stop-anomaly",
     "note": "vehicle stationary 40 s in lane 3", "real": False},   # bus at a stop — FP
    {"t": "23:15", "cam": "CAM-13", "watch": "I-11", "type": "collision:sideswipe",
     "note": "two tracks merge then diverge, debris", "real": True},
    {"t": "23:16", "cam": "CAM-14", "watch": "I-11", "type": "anomalous-movement",
     "note": "pedestrians converge on lane 1", "real": True},        # same incident, 2nd view
    {"t": "23:58", "cam": "CAM-07", "watch": "S-04", "type": "stop-anomaly",
     "note": "vehicle stationary 90 s, hazards on", "real": True},
    {"t": "00:31", "cam": "CAM-19", "watch": "I-14", "type": "collision:rear-end",
     "note": "shadow flicker at gate arch", "real": False},          # headlight shadows — FP
]


def verify(event: dict) -> dict:
    """The Alert Verification Service: a VLM re-watches the clip.

    This is the blueprint's core economics: CV casts a wide net (cheap, per-frame),
    the VLM confirms (expensive, per-alert) — so humans only ever see verified alerts.
    """
    verdict = "CONFIRMED" if event["real"] else "REJECTED"
    reason = {
        False: "VLM: matches a scheduled bus dwell / lighting artifact — no incident.",
        True:  "VLM: clip shows a genuine incident consistent with the CV track.",
    }[event["real"]]
    return {**event, "verdict": verdict, "reason": reason}


def kafka_trace(event: dict) -> list[str]:
    """One event's trip through the pipeline, as a wire trace."""
    return [
        f"RTVI CV ({event['cam']})     detect+track → {event['type']}",
        f"kafka topic cv.events        ← {{cam:{event['cam']}, watch:{event['watch']}, t:{event['t']}}}",
        "alert-verification (VLM)     ← consumes event, fetches 12 s clip via VIOS",
        f"kafka topic alerts.verified  ← verdict published",
        "VSS agent / map UI (:3002)   ← only VERIFIED alerts reach a human",
    ]
